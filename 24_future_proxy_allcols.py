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
t0 = time.time()


def ts():
    m, s = divmod(int(time.time() - t0), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def finite32(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def try_gpu():
    try:
        ds = lgb.Dataset(np.random.rand(128, 4), np.random.rand(128))
        lgb.train({"objective": "regression", "device": "gpu", "gpu_use_dp": False, "verbose": -1}, ds, 2)
        return True
    except Exception:
        return False


def add_future_proxy(df, cols):
    g = df.groupby("scenario_id", sort=False)
    feats = {}
    for c in cols:
        s = df[c]
        lead1 = g[c].shift(-1)
        lead2 = g[c].shift(-2)
        lead3 = g[c].shift(-3)
        feats[f"afp_{c}_lead1"] = lead1
        feats[f"afp_{c}_lead2"] = lead2
        feats[f"afp_{c}_lead3"] = lead3
        feats[f"afp_{c}_next2_mean"] = pd.concat([lead1, lead2], axis=1).mean(axis=1)
        feats[f"afp_{c}_next3_mean"] = pd.concat([lead1, lead2, lead3], axis=1).mean(axis=1)
        feats[f"afp_{c}_delta1"] = lead1 - s
        feats[f"afp_{c}_delta2"] = lead2 - s
    return pd.DataFrame(feats, index=df.index)


def build_matrix():
    d = np.load("features.npz", allow_pickle=True)
    X = finite32(d["X"])
    Xte = finite32(d["X_te"])
    y_raw = d["y_raw"].astype(np.float64)
    y_log = d["y_log"].astype(np.float32)
    groups = d["groups"]

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    train["row_idx"] = np.arange(len(train))
    test["row_idx"] = np.arange(len(test))
    train = train.sort_values(["scenario_id", "row_idx"]).reset_index(drop=True)
    test = test.sort_values(["scenario_id", "row_idx"]).reset_index(drop=True)
    train["timeslot"] = train.groupby("scenario_id").cumcount().astype(np.int16)
    test["timeslot"] = test.groupby("scenario_id").cumcount().astype(np.int16)

    cols = [
        c
        for c in train.columns
        if c not in ["ID", "layout_id", "scenario_id", T, "row_idx"]
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    log(f"all numeric future columns={len(cols)}")
    fp_tr = add_future_proxy(train, cols)
    fp_te = add_future_proxy(test, cols)
    fp_tr["row_idx"] = train["row_idx"].values
    fp_te["row_idx"] = test["row_idx"].values
    fp_tr = fp_tr.sort_values("row_idx").drop(columns=["row_idx"])
    fp_te = fp_te.sort_values("row_idx").drop(columns=["row_idx"])
    Xall = np.hstack([X, finite32(fp_tr.values)]).astype(np.float32)
    Xall_te = np.hstack([Xte, finite32(fp_te.values)]).astype(np.float32)
    return Xall, Xall_te, y_raw, y_log, groups, fp_tr.columns


def train_lgb_cv(X, Xte, y_log, y_raw, groups, params, rounds, name):
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.arange(len(groups)), groups=groups))
    oof = np.zeros(X.shape[0], dtype=np.float32)
    pred = np.zeros(Xte.shape[0], dtype=np.float32)
    for fi, (tri, vai) in enumerate(folds):
        log(f"  {name} fold {fi + 1}/{N_SPLITS}")
        dtr = lgb.Dataset(X[tri], y_log[tri])
        dva = lgb.Dataset(X[vai], y_log[vai])
        model = lgb.train(
            params,
            dtr,
            num_boost_round=rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        oof[vai] = np.expm1(model.predict(X[vai], num_iteration=model.best_iteration)).astype(np.float32)
        pred += np.expm1(model.predict(Xte, num_iteration=model.best_iteration)).astype(np.float32) / N_SPLITS
        del model, dtr, dva
        gc.collect()
    log(f"  {name} OOF={mean_absolute_error(y_raw, oof):.5f} mean={oof.mean():.4f} std={oof.std():.4f}")
    return oof, pred


def slsqp(oof_stack, y):
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
        options={"maxiter": 2500, "ftol": 1e-11},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def save(sample, pred, fname):
    out = sample.copy()
    out[T] = np.clip(pred, 0, None).astype(np.float32)
    out.to_csv(fname, index=False)
    log(f"  {fname:34s} mean={out[T].mean():.4f} std={out[T].std():.4f} max={out[T].max():.2f}")


def main():
    X, Xte, y_raw, y_log, groups, fp_cols = build_matrix()
    log(f"X={X.shape} Xte={Xte.shape}")
    dev = {"device": "gpu", "gpu_use_dp": False} if try_gpu() else {}
    log(f"LGB device={'GPU' if dev else 'CPU'}")

    common = dict(
        metric="mae",
        learning_rate=0.03,
        feature_fraction=0.62,
        bagging_fraction=0.85,
        bagging_freq=1,
        min_child_samples=35,
        reg_alpha=0.15,
        reg_lambda=1.5,
        verbose=-1,
        n_jobs=-1,
        seed=42,
        **dev,
    )
    specs = [
        ("afp_q50", dict(common, objective="quantile", alpha=0.50, num_leaves=127), 3500),
        ("afp_q55", dict(common, objective="quantile", alpha=0.55, num_leaves=127, reg_lambda=2.0), 3500),
        ("afp_mae", dict(common, objective="regression_l1", num_leaves=95, reg_lambda=2.0), 3000),
    ]

    oofs, preds, names = [], [], []
    for name, params, rounds in specs:
        o, p = train_lgb_cv(X, Xte, y_log, y_raw, groups, params, rounds, name)
        oofs.append(o)
        preds.append(p)
        names.append(name)

    old = np.load("xgb_future_proxy_oof.npz", allow_pickle=True)
    oof_stack = np.column_stack([old["combo_oof"].astype(np.float64)] + [o.astype(np.float64) for o in oofs])
    pred_stack = np.column_stack([old["combo_pred"].astype(np.float64)] + [p.astype(np.float64) for p in preds])
    w = slsqp(oof_stack, y_raw)
    combo_oof = (oof_stack @ w).astype(np.float32)
    combo_pred = (pred_stack @ w).astype(np.float64)
    log(f"allcols combo OOF={mean_absolute_error(y_raw, combo_oof):.5f}")
    for n, ww in zip(["old_xgbfp_combo"] + names, w):
        log(f"  {n:16s} {ww:.4f}")

    np.savez_compressed(
        "future_proxy_allcols_oof.npz",
        oof_stack=np.column_stack(oofs).astype(np.float32),
        pred_stack=np.column_stack(preds).astype(np.float32),
        names=np.array(names),
        combo_component_names=np.array(["old_xgbfp_combo"] + names),
        combo_oof=combo_oof,
        combo_pred=combo_pred,
        weights=w,
        y_raw=y_raw,
        fp_cols=np.array(fp_cols),
    )

    sample = pd.read_csv("data/sample_submission.csv")
    base = pd.read_csv("sub_v12_main.csv")[T].values.astype(np.float64)
    scaled = np.clip(combo_pred * (base.mean() / max(combo_pred.mean(), 1e-9)), 0, None)
    save(sample, scaled, "sub_v24_allfp_scaled.csv")
    for wb in [0.25, 0.35, 0.42, 0.50, 0.60]:
        pred = (1 - wb) * base + wb * scaled
        save(sample, pred, f"sub_v24_v12{int(round((1-wb)*100)):02d}_fp{int(round(wb*100)):02d}.csv")

    log("Primary probes: sub_v24_v1265_fp35.csv, sub_v24_v1258_fp42.csv")


if __name__ == "__main__":
    main()
