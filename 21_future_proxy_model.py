import gc
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

T = "avg_delay_minutes_next_30m"
N_SPLITS = 5
SEED = 42
t0 = time.time()


def ts():
    m, s = divmod(int(time.time() - t0), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def finite32(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def try_lgb_gpu():
    try:
        d = lgb.Dataset(np.random.rand(256, 4), label=np.random.rand(256))
        lgb.train({"objective": "regression", "metric": "l1", "device": "gpu", "gpu_use_dp": False, "verbose": -1}, d, 2)
        return True
    except Exception:
        return False


def prepare_raw():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train["row_idx"] = np.arange(len(train))
    test["row_idx"] = np.arange(len(test))
    train = train.sort_values(["scenario_id", "row_idx"]).reset_index(drop=True)
    test = test.sort_values(["scenario_id", "row_idx"]).reset_index(drop=True)
    train["timeslot"] = train.groupby("scenario_id").cumcount().astype(np.int16)
    test["timeslot"] = test.groupby("scenario_id").cumcount().astype(np.int16)
    return train, test


def select_top_columns(train, top_n=38):
    num = [
        c
        for c in train.columns
        if c not in ["ID", "layout_id", "scenario_id", T, "row_idx"]
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    rows = []
    y = train[T]
    for c in num:
        corr = train[c].corr(y)
        if pd.notna(corr):
            rows.append((c, abs(corr), corr))
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    cols = [r[0] for r in rows[:top_n]]
    log("Top future-proxy columns:")
    for c, a, corr in rows[:top_n]:
        log(f"  {c:32s} corr={corr:+.4f}")
    return cols


def add_future_proxy(df, cols):
    g = df.groupby("scenario_id", sort=False)
    feats = {}
    for c in cols:
        s = df[c]
        lead1 = g[c].shift(-1)
        lead2 = g[c].shift(-2)
        lead3 = g[c].shift(-3)
        feats[f"fp_{c}_lead1"] = lead1
        feats[f"fp_{c}_lead2"] = lead2
        feats[f"fp_{c}_lead3"] = lead3
        feats[f"fp_{c}_next2_mean"] = pd.concat([lead1, lead2], axis=1).mean(axis=1)
        feats[f"fp_{c}_next3_mean"] = pd.concat([lead1, lead2, lead3], axis=1).mean(axis=1)
        feats[f"fp_{c}_now_next2_mean"] = pd.concat([s, lead1, lead2], axis=1).mean(axis=1)
        feats[f"fp_{c}_delta_lead1"] = lead1 - s
        feats[f"fp_{c}_delta_lead2"] = lead2 - s
    out = pd.DataFrame(feats, index=df.index)
    return out


def train_lgb_cv(X, Xte, y_model, y_raw, groups, params, rounds, name, post):
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.arange(len(groups)), groups=groups))
    oof = np.zeros(X.shape[0], dtype=np.float32)
    pred = np.zeros(Xte.shape[0], dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        log(f"  {name} fold {fi + 1}/{N_SPLITS}")
        dtr = lgb.Dataset(X[tri], y_model[tri])
        dva = lgb.Dataset(X[vai], y_model[vai])
        model = lgb.train(
            params,
            dtr,
            num_boost_round=rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        oof[vai] = post(model.predict(X[vai], num_iteration=model.best_iteration)).astype(np.float32)
        pred += post(model.predict(Xte, num_iteration=model.best_iteration)).astype(np.float32) / N_SPLITS
        del model, dtr, dva
        gc.collect()
    log(f"  {name} OOF={mean_absolute_error(y_raw, oof):.5f} mean={oof.mean():.4f} std={oof.std():.4f}")
    return oof, pred


def save(sample, pred, fname):
    out = sample.copy()
    out[T] = np.clip(pred, 0, None).astype(np.float32)
    out.to_csv(fname, index=False)
    log(f"  {fname:32s} mean={out[T].mean():.4f} std={out[T].std():.4f} max={out[T].max():.2f}")


def slsqp_weights(oof_stack, y):
    def obj(w):
        w = np.clip(w, 0, None)
        if w.sum() < 1e-12:
            return 1e9
        return mean_absolute_error(y, oof_stack @ (w / w.sum()))

    w0 = np.ones(oof_stack.shape[1]) / oof_stack.shape[1]
    res = minimize(
        obj,
        w0,
        method="SLSQP",
        bounds=[(0, 1)] * len(w0),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 2000, "ftol": 1e-11},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def main():
    log("Loading base features")
    d = np.load("features.npz", allow_pickle=True)
    X = finite32(d["X"])
    Xte = finite32(d["X_te"])
    y_raw = d["y_raw"].astype(np.float64)
    y_log = d["y_log"].astype(np.float32)
    y_sqrt = d["y_sqrt"].astype(np.float32)
    groups = d["groups"]

    train, test = prepare_raw()
    cols = select_top_columns(train, top_n=38)
    log("Building future-proxy blocks")
    fp_tr = add_future_proxy(train, cols)
    fp_te = add_future_proxy(test, cols)
    fp_tr = fp_tr.sort_index()
    fp_te = fp_te.sort_index()
    # restore original row order
    train_order = train["row_idx"].values
    test_order = test["row_idx"].values
    fp_tr["row_idx"] = train_order
    fp_te["row_idx"] = test_order
    fp_tr = fp_tr.sort_values("row_idx").drop(columns=["row_idx"])
    fp_te = fp_te.sort_values("row_idx").drop(columns=["row_idx"])
    Xfp = finite32(fp_tr.values)
    Xfp_te = finite32(fp_te.values)
    Xall = np.hstack([X, Xfp]).astype(np.float32)
    Xall_te = np.hstack([Xte, Xfp_te]).astype(np.float32)
    log(f"Xall={Xall.shape}, added={Xfp.shape[1]}")

    gpu = try_lgb_gpu()
    dev = {"device": "gpu", "gpu_use_dp": False} if gpu else {}
    log(f"LightGBM device: {'GPU' if gpu else 'CPU'}")

    common = dict(
        metric="mae",
        learning_rate=0.03,
        feature_fraction=0.70,
        bagging_fraction=0.85,
        bagging_freq=1,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbose=-1,
        n_jobs=-1,
        seed=SEED,
        **dev,
    )
    specs = [
        ("fp_q50", dict(common, objective="quantile", alpha=0.50, num_leaves=127), 4000, y_log, lambda z: np.expm1(z)),
        ("fp_q55", dict(common, objective="quantile", alpha=0.55, num_leaves=127, reg_lambda=1.5), 4000, y_log, lambda z: np.expm1(z)),
        ("fp_mae", dict(common, objective="regression_l1", num_leaves=95, reg_lambda=1.2), 3500, y_log, lambda z: np.expm1(z)),
        ("fp_sqrt", dict(common, objective="quantile", alpha=0.50, num_leaves=95, learning_rate=0.04), 1500, y_sqrt, lambda z: np.square(np.clip(z, 0, None))),
    ]

    oofs, preds, names = [], [], []
    for name, params, rounds, target, post in specs:
        o, p = train_lgb_cv(Xall, Xall_te, target, y_raw, groups, params, rounds, name, post)
        oofs.append(o)
        preds.append(p)
        names.append(name)

    # Simple OOF-weight heuristic and direct SLSQP.  Direct SLSQP is less
    # conservative but this file is meant to create candidates, not one safe
    # final answer.
    maes = np.array([mean_absolute_error(y_raw, o) for o in oofs])
    oof_stack = np.column_stack(oofs).astype(np.float64)
    pred_stack = np.column_stack(preds).astype(np.float64)
    w_heur = 1.0 / np.maximum(maes - maes.min() + 0.02, 0.02)
    w_heur = w_heur / w_heur.sum()
    w_slsqp = slsqp_weights(oof_stack, y_raw)
    ens_oof = (oof_stack @ w_slsqp).astype(np.float32)
    ens_pred = (pred_stack @ w_slsqp).astype(np.float32)
    heur_oof = (oof_stack @ w_heur).astype(np.float32)
    heur_pred = (pred_stack @ w_heur).astype(np.float32)
    log(f"fp heuristic OOF={mean_absolute_error(y_raw, heur_oof):.5f}")
    log(f"fp SLSQP OOF={mean_absolute_error(y_raw, ens_oof):.5f}")
    for n, wh, ws, mm in zip(names, w_heur, w_slsqp, maes):
        log(f"  {n:8s} mae={mm:.5f} w_heur={wh:.4f} w_slsqp={ws:.4f}")

    np.savez_compressed(
        "future_proxy_oof.npz",
        oof_stack=oof_stack.astype(np.float32),
        pred_stack=pred_stack.astype(np.float32),
        ens_oof=ens_oof,
        ens_pred=ens_pred,
        heur_oof=heur_oof,
        heur_pred=heur_pred,
        weights=w_slsqp,
        weights_heur=w_heur,
        names=np.array(names),
        y_raw=y_raw,
        fp_cols=np.array(fp_tr.columns),
    )

    sample = pd.read_csv("data/sample_submission.csv")
    base = pd.read_csv("sub_v12_main.csv")[T].values.astype(np.float64)
    raw = ens_pred.astype(np.float64)
    scaled = np.clip(raw * (base.mean() / max(raw.mean(), 1e-9)), 0, None)
    save(sample, scaled, "sub_v21_future_proxy_scaled.csv")
    for wb in [0.20, 0.35, 0.50, 0.65]:
        pred = (1 - wb) * base + wb * scaled
        save(sample, pred, f"sub_v21_v12{int((1-wb)*100):02d}_fp{int(wb*100):02d}.csv")

    raw_h = heur_pred.astype(np.float64)
    scaled_h = np.clip(raw_h * (base.mean() / max(raw_h.mean(), 1e-9)), 0, None)
    save(sample, scaled_h, "sub_v21_future_proxy_heur_scaled.csv")
    for wb in [0.20, 0.35, 0.50]:
        pred = (1 - wb) * base + wb * scaled_h
        save(sample, pred, f"sub_v21h_v12{int((1-wb)*100):02d}_fp{int(wb*100):02d}.csv")

    log("Primary probes: sub_v21_v1265_fp35.csv, sub_v21_v1250_fp50.csv")


if __name__ == "__main__":
    main()
