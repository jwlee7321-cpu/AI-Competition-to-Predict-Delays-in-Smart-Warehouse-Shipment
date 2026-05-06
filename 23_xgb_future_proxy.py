import gc
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
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
            rows.append((c, abs(corr)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in rows[:top_n]]


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

    cols = select_top_columns(train, top_n=38)
    log(f"future proxy columns={len(cols)}")
    fp_tr = add_future_proxy(train, cols)
    fp_te = add_future_proxy(test, cols)
    fp_tr["row_idx"] = train["row_idx"].values
    fp_te["row_idx"] = test["row_idx"].values
    fp_tr = fp_tr.sort_values("row_idx").drop(columns=["row_idx"])
    fp_te = fp_te.sort_values("row_idx").drop(columns=["row_idx"])

    Xall = np.hstack([X, finite32(fp_tr.values)]).astype(np.float32)
    Xall_te = np.hstack([Xte, finite32(fp_te.values)]).astype(np.float32)
    return Xall, Xall_te, y_raw, y_log, groups


def train_xgb_cv(X, Xte, y_log, y_raw, groups, params, rounds, name):
    folds = list(GroupKFold(n_splits=N_SPLITS).split(np.arange(len(groups)), groups=groups))
    oof = np.zeros(X.shape[0], dtype=np.float32)
    pred = np.zeros(Xte.shape[0], dtype=np.float32)
    dte = xgb.DMatrix(Xte)
    for fi, (tri, vai) in enumerate(folds):
        log(f"  {name} fold {fi + 1}/{N_SPLITS}")
        dtr = xgb.DMatrix(X[tri], label=y_log[tri])
        dva = xgb.DMatrix(X[vai], label=y_log[vai])
        model = xgb.train(
            params,
            dtr,
            num_boost_round=rounds,
            evals=[(dva, "val")],
            early_stopping_rounds=80,
            verbose_eval=False,
        )
        oof[vai] = np.expm1(model.predict(dva, iteration_range=(0, model.best_iteration + 1))).astype(np.float32)
        pred += np.expm1(model.predict(dte, iteration_range=(0, model.best_iteration + 1))).astype(np.float32) / N_SPLITS
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
        options={"maxiter": 3000, "ftol": 1e-11},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def save(sample, pred, fname):
    out = sample.copy()
    out[T] = np.clip(pred, 0, None).astype(np.float32)
    out.to_csv(fname, index=False)
    log(f"  {fname:34s} mean={out[T].mean():.4f} std={out[T].std():.4f} max={out[T].max():.2f}")


def main():
    X, Xte, y_raw, y_log, groups = build_matrix()
    log(f"X={X.shape} Xte={Xte.shape}")

    common = dict(
        eval_metric="mae",
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=35,
        subsample=0.82,
        colsample_bytree=0.72,
        reg_alpha=0.15,
        reg_lambda=3.0,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        seed=42,
        verbosity=0,
    )
    specs = [
        ("xgb_fp_abs", dict(common, objective="reg:absoluteerror"), 1800),
        ("xgb_fp_q55", dict(common, objective="reg:quantileerror", quantile_alpha=0.55), 1800),
        ("xgb_fp_q60", dict(common, objective="reg:quantileerror", quantile_alpha=0.60, reg_lambda=4.0), 1800),
    ]

    x_oofs, x_preds, x_names = [], [], []
    for name, params, rounds in specs:
        o, p = train_xgb_cv(X, Xte, y_log, y_raw, groups, params, rounds, name)
        x_oofs.append(o)
        x_preds.append(p)
        x_names.append(name)

    fp = np.load("future_proxy_oof.npz", allow_pickle=True)
    names = list(fp["names"].astype(str)) + x_names
    oof_stack = np.column_stack([fp["oof_stack"].astype(np.float64)] + [o.astype(np.float64) for o in x_oofs])
    pred_stack = np.column_stack([fp["pred_stack"].astype(np.float64)] + [p.astype(np.float64) for p in x_preds])
    w = slsqp(oof_stack, y_raw)
    combo_oof = (oof_stack @ w).astype(np.float32)
    combo_pred = (pred_stack @ w).astype(np.float64)
    log(f"combo OOF={mean_absolute_error(y_raw, combo_oof):.5f}")
    for n, ww in zip(names, w):
        log(f"  {n:10s} {ww:.4f}")

    np.savez_compressed(
        "xgb_future_proxy_oof.npz",
        xgb_oof_stack=np.column_stack(x_oofs).astype(np.float32),
        xgb_pred_stack=np.column_stack(x_preds).astype(np.float32),
        combo_oof=combo_oof,
        combo_pred=combo_pred,
        weights=w,
        names=np.array(names),
        y_raw=y_raw,
    )

    sample = pd.read_csv("data/sample_submission.csv")
    base = pd.read_csv("sub_v12_main.csv")[T].values.astype(np.float64)
    scaled = np.clip(combo_pred * (base.mean() / max(combo_pred.mean(), 1e-9)), 0, None)
    save(sample, scaled, "sub_v23_xgbfp_scaled.csv")
    for wb in [0.20, 0.30, 0.37, 0.45, 0.55]:
        pred = (1 - wb) * base + wb * scaled
        save(sample, pred, f"sub_v23_v12{int(round((1-wb)*100)):02d}_fp{int(round(wb*100)):02d}.csv")

    log("Primary probes: sub_v23_v1263_fp37.csv, sub_v23_v1270_fp30.csv")


if __name__ == "__main__":
    main()
