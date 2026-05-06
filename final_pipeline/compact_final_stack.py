import argparse
import gc
import os
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


def make_folds(groups, seed=None):
    if seed is None:
        return list(GroupKFold(n_splits=N_SPLITS).split(np.arange(len(groups)), groups=groups))
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups).copy()
    rng.shuffle(uniq)
    fold_map = {g: i % N_SPLITS for i, g in enumerate(uniq)}
    fid = np.array([fold_map[g] for g in groups])
    return [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(N_SPLITS)]


def load_base():
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
    return X, Xte, y_raw, y_log, groups, train, test, cols


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


def add_scenario_context(df, cols):
    g = df.groupby("scenario_id", sort=False)
    feats = {}
    eps = 1e-3
    for c in cols:
        cur = df[c].astype(np.float32)
        mean = g[c].transform("mean")
        std = g[c].transform("std").fillna(0)
        mn = g[c].transform("min")
        mx = g[c].transform("max")
        first = g[c].transform("first")
        last = g[c].transform("last")
        feats[f"sctx_{c}_mean"] = mean
        feats[f"sctx_{c}_std"] = std
        feats[f"sctx_{c}_min"] = mn
        feats[f"sctx_{c}_max"] = mx
        feats[f"sctx_{c}_range"] = mx - mn
        feats[f"sctx_{c}_last_first"] = last - first
        feats[f"sctx_{c}_cur_minus_mean"] = cur - mean
        feats[f"sctx_{c}_cur_z"] = (cur - mean) / (std + eps)
        feats[f"sctx_{c}_cur_to_max"] = cur / (mx.abs() + 1.0)
        feats[f"sctx_{c}_cur_to_mean"] = cur / (mean.abs() + 1.0)
    return pd.DataFrame(feats, index=df.index)


def add_future_shape(df, cols):
    g = df.groupby("scenario_id", sort=False)
    feats = {}
    eps = 1.0
    for c in cols:
        cur = df[c].astype(np.float32)
        lead1 = g[c].shift(-1)
        lead2 = g[c].shift(-2)
        lead3 = g[c].shift(-3)
        lead4 = g[c].shift(-4)
        nxt2 = pd.concat([lead1, lead2], axis=1)
        nxt3 = pd.concat([lead1, lead2, lead3], axis=1)
        nxt4 = pd.concat([lead1, lead2, lead3, lead4], axis=1)
        cur2 = pd.concat([cur, lead1, lead2], axis=1)
        feats[f"fsp_{c}_lead1"] = lead1
        feats[f"fsp_{c}_lead2"] = lead2
        feats[f"fsp_{c}_lead3"] = lead3
        feats[f"fsp_{c}_lead4"] = lead4
        feats[f"fsp_{c}_next2_mean"] = nxt2.mean(axis=1)
        feats[f"fsp_{c}_next3_mean"] = nxt3.mean(axis=1)
        feats[f"fsp_{c}_next4_mean"] = nxt4.mean(axis=1)
        feats[f"fsp_{c}_next2_max"] = nxt2.max(axis=1)
        feats[f"fsp_{c}_next2_min"] = nxt2.min(axis=1)
        feats[f"fsp_{c}_next3_max"] = nxt3.max(axis=1)
        feats[f"fsp_{c}_next3_min"] = nxt3.min(axis=1)
        feats[f"fsp_{c}_next3_std"] = nxt3.std(axis=1)
        feats[f"fsp_{c}_cur_next2_mean"] = cur2.mean(axis=1)
        feats[f"fsp_{c}_delta1"] = lead1 - cur
        feats[f"fsp_{c}_delta2"] = lead2 - cur
        feats[f"fsp_{c}_delta3"] = lead3 - cur
        feats[f"fsp_{c}_slope12"] = lead2 - lead1
        feats[f"fsp_{c}_slope23"] = lead3 - lead2
        feats[f"fsp_{c}_accel012"] = lead2 - 2.0 * lead1 + cur
        feats[f"fsp_{c}_reldelta2"] = (lead2 - cur) / (cur.abs() + eps)
    return pd.DataFrame(feats, index=df.index)


def append_ordered_features(X, Xte, train, test, feat_fn, cols):
    tr = feat_fn(train, cols)
    te = feat_fn(test, cols)
    tr["row_idx"] = train["row_idx"].values
    te["row_idx"] = test["row_idx"].values
    tr = tr.sort_values("row_idx").drop(columns=["row_idx"])
    te = te.sort_values("row_idx").drop(columns=["row_idx"])
    Xall = np.hstack([X, finite32(tr.values)]).astype(np.float32)
    Xall_te = np.hstack([Xte, finite32(te.values)]).astype(np.float32)
    return Xall, Xall_te, np.array(tr.columns)


def train_lgb_cv(X, Xte, y_log, y_raw, groups, params, rounds, name, fold_seeds=(None,)):
    n_tr, n_te = X.shape[0], Xte.shape[0]
    oof_sum = np.zeros(n_tr, dtype=np.float64)
    oof_count = np.zeros(n_tr, dtype=np.int16)
    pred_sum = np.zeros(n_te, dtype=np.float64)
    runs = 0
    for fs in fold_seeds:
        folds = make_folds(groups, seed=fs)
        seed_val = 42 if fs is None else int(fs)
        p = dict(params, seed=seed_val, bagging_seed=seed_val, feature_fraction_seed=seed_val)
        tag = "main" if fs is None else str(fs)
        for fi, (tri, vai) in enumerate(folds):
            log(f"  {name} foldseed={tag} fold {fi + 1}/{N_SPLITS}")
            dtr = lgb.Dataset(X[tri], y_log[tri])
            dva = lgb.Dataset(X[vai], y_log[vai])
            model = lgb.train(
                p,
                dtr,
                num_boost_round=rounds,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
            )
            pv = np.expm1(model.predict(X[vai], num_iteration=model.best_iteration)).astype(np.float32)
            pt = np.expm1(model.predict(Xte, num_iteration=model.best_iteration)).astype(np.float32)
            oof_sum[vai] += pv
            oof_count[vai] += 1
            pred_sum += pt
            runs += 1
            del model, dtr, dva, pv, pt
            gc.collect()
    oof = (oof_sum / np.maximum(oof_count, 1)).astype(np.float32)
    pred = (pred_sum / runs).astype(np.float32)
    log(f"  {name} OOF={mean_absolute_error(y_raw, oof):.6f} mean={oof.mean():.4f} std={oof.std():.4f} runs={runs}")
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


def save_submission(sample, pred, fname):
    out = sample.copy()
    out[T] = np.clip(pred, 0, None).astype(np.float32)
    out.to_csv(fname, index=False)
    log(f"  {fname:34s} mean={out[T].mean():.4f} std={out[T].std():.4f} max={out[T].max():.2f}")


def blend_and_save(name, comp_names, oofs, preds, y_raw, extra_npz, out_npz):
    oof_stack = np.column_stack([o.astype(np.float64) for o in oofs])
    pred_stack = np.column_stack([p.astype(np.float64) for p in preds])
    w = slsqp(oof_stack, y_raw)
    combo_oof = (oof_stack @ w).astype(np.float32)
    combo_pred = (pred_stack @ w).astype(np.float64)
    log(f"{name} combo OOF={mean_absolute_error(y_raw, combo_oof):.6f}")
    for n, ww in zip(comp_names, w):
        log(f"  {n:22s} {ww:.4f}")
    np.savez_compressed(
        out_npz,
        component_names=np.array(comp_names),
        combo_oof=combo_oof,
        combo_pred=combo_pred,
        weights=w,
        y_raw=y_raw,
        **extra_npz,
    )
    return combo_oof, combo_pred


def stage_v25(force, dev):
    out = "future_proxy_allcols_multifold_oof.npz"
    if os.path.exists(out) and not force:
        log(f"skip v25: {out} exists")
        return
    X, Xte, y_raw, y_log, groups, train, test, cols = load_base()
    Xall, Xall_te, fp_cols = append_ordered_features(X, Xte, train, test, add_future_proxy, cols)
    common = dict(metric="mae", learning_rate=0.03, feature_fraction=0.62, bagging_fraction=0.85,
                  bagging_freq=1, min_child_samples=35, reg_alpha=0.15, reg_lambda=1.8,
                  verbose=-1, n_jobs=-1, **dev)
    specs = [
        ("afpmf_q55", dict(common, objective="quantile", alpha=0.55, num_leaves=127, reg_lambda=2.0), 3500),
        ("afpmf_q60", dict(common, objective="quantile", alpha=0.60, num_leaves=127, reg_lambda=2.5), 3500),
        ("afpmf_q50", dict(common, objective="quantile", alpha=0.50, num_leaves=127, reg_lambda=1.8), 3200),
    ]
    oofs, preds, names = [], [], []
    for n, p, r in specs:
        o, pr = train_lgb_cv(Xall, Xall_te, y_log, y_raw, groups, p, r, n, fold_seeds=(2025, 2028, 2031))
        oofs.append(o); preds.append(pr); names.append(n)
    v23 = np.load("xgb_future_proxy_oof.npz", allow_pickle=True)
    v24 = np.load("future_proxy_allcols_oof.npz", allow_pickle=True)
    comp_names = ["v23_xgbfp_combo", "v24_allcols_combo"] + names
    combo_oof, combo_pred = blend_and_save(
        "v25", comp_names,
        [v23["combo_oof"], v24["combo_oof"]] + oofs,
        [v23["combo_pred"], v24["combo_pred"]] + preds,
        y_raw,
        dict(oof_stack=np.column_stack(oofs).astype(np.float32), pred_stack=np.column_stack(preds).astype(np.float32),
             names=np.array(names), fp_cols=fp_cols, fold_seeds=np.array([2025, 2028, 2031])),
        out,
    )
    del Xall, Xall_te, oofs, preds, combo_oof, combo_pred
    gc.collect()


def stage_v27(force, dev):
    out = "scenario_context_oof.npz"
    if os.path.exists(out) and not force:
        log(f"skip v27: {out} exists")
        return
    X, Xte, y_raw, y_log, groups, train, test, cols = load_base()
    Xall, Xall_te, ctx_cols = append_ordered_features(X, Xte, train, test, add_scenario_context, cols)
    common = dict(metric="mae", learning_rate=0.03, feature_fraction=0.58, bagging_fraction=0.86,
                  bagging_freq=1, min_child_samples=38, reg_alpha=0.18, reg_lambda=2.2,
                  verbose=-1, n_jobs=-1, **dev)
    specs = [
        ("sctx_q55", dict(common, objective="quantile", alpha=0.55, num_leaves=127), 3500),
        ("sctx_q60", dict(common, objective="quantile", alpha=0.60, num_leaves=127, reg_lambda=2.7), 3500),
        ("sctx_mae", dict(common, objective="regression_l1", num_leaves=95, reg_lambda=2.8), 3000),
    ]
    oofs, preds, names = [], [], []
    for n, p, r in specs:
        o, pr = train_lgb_cv(Xall, Xall_te, y_log, y_raw, groups, p, r, n)
        oofs.append(o); preds.append(pr); names.append(n)
    v23 = np.load("xgb_future_proxy_oof.npz", allow_pickle=True)
    v24 = np.load("future_proxy_allcols_oof.npz", allow_pickle=True)
    v25 = np.load("future_proxy_allcols_multifold_oof.npz", allow_pickle=True)
    comp_names = ["v23_xgbfp_combo", "v24_allcols_combo", "v25_multifold_combo"] + names
    blend_and_save(
        "v27", comp_names,
        [v23["combo_oof"], v24["combo_oof"], v25["combo_oof"]] + oofs,
        [v23["combo_pred"], v24["combo_pred"], v25["combo_pred"]] + preds,
        y_raw,
        dict(oof_stack=np.column_stack(oofs).astype(np.float32), pred_stack=np.column_stack(preds).astype(np.float32),
             names=np.array(names), ctx_cols=ctx_cols),
        out,
    )
    del Xall, Xall_te, oofs, preds
    gc.collect()


def stage_v26(force, dev):
    out = "future_proxy_shape_oof.npz"
    if os.path.exists(out) and not force:
        log(f"skip v26: {out} exists")
        return
    X, Xte, y_raw, y_log, groups, train, test, cols = load_base()
    Xall, Xall_te, fp_cols = append_ordered_features(X, Xte, train, test, add_future_shape, cols)
    common = dict(metric="mae", learning_rate=0.027, feature_fraction=0.54, bagging_fraction=0.86,
                  bagging_freq=1, min_child_samples=45, reg_alpha=0.20, reg_lambda=2.4,
                  verbose=-1, n_jobs=-1, **dev)
    specs = [
        ("fsp_q55", dict(common, objective="quantile", alpha=0.55, num_leaves=127), 3800),
        ("fsp_q60", dict(common, objective="quantile", alpha=0.60, num_leaves=127, reg_lambda=2.8), 3800),
        ("fsp_mae", dict(common, objective="regression_l1", num_leaves=95, reg_lambda=3.0), 3300),
    ]
    oofs, preds, names = [], [], []
    for n, p, r in specs:
        o, pr = train_lgb_cv(Xall, Xall_te, y_log, y_raw, groups, p, r, n)
        oofs.append(o); preds.append(pr); names.append(n)
    v23 = np.load("xgb_future_proxy_oof.npz", allow_pickle=True)
    v24 = np.load("future_proxy_allcols_oof.npz", allow_pickle=True)
    v25 = np.load("future_proxy_allcols_multifold_oof.npz", allow_pickle=True)
    v27 = np.load("scenario_context_oof.npz", allow_pickle=True)
    comp_names = ["v23_xgbfp_combo", "v24_allcols_combo", "v25_multifold_combo", "v27_sctx_combo"] + names
    blend_and_save(
        "v26", comp_names,
        [v23["combo_oof"], v24["combo_oof"], v25["combo_oof"], v27["combo_oof"]] + oofs,
        [v23["combo_pred"], v24["combo_pred"], v25["combo_pred"], v27["combo_pred"]] + preds,
        y_raw,
        dict(oof_stack=np.column_stack(oofs).astype(np.float32), pred_stack=np.column_stack(preds).astype(np.float32),
             names=np.array(names), fp_cols=fp_cols),
        out,
    )
    del Xall, Xall_te, oofs, preds
    gc.collect()


def stage_final():
    sources = [
        ("xgb_future_proxy_oof.npz", "v23_xgbfp"),
        ("future_proxy_allcols_oof.npz", "v24_allcols"),
        ("future_proxy_allcols_multifold_oof.npz", "v25_multifold"),
        ("future_proxy_shape_oof.npz", "v26_shape"),
        ("scenario_context_oof.npz", "v27_sctx"),
    ]
    names, oofs, preds, y = [], [], [], None
    for fname, name in sources:
        d = np.load(fname, allow_pickle=True)
        names.append(name)
        oofs.append(d["combo_oof"].astype(np.float64))
        preds.append(d["combo_pred"].astype(np.float64))
        if y is None:
            y = d["y_raw"].astype(np.float64)
    oof_stack = np.column_stack(oofs)
    pred_stack = np.column_stack(preds)
    w = slsqp(oof_stack, y)
    combo_oof = (oof_stack @ w).astype(np.float32)
    combo_pred = (pred_stack @ w).astype(np.float64)
    log(f"final future stack OOF={mean_absolute_error(y, combo_oof):.6f}")
    for n, ww in zip(names, w):
        log(f"  {n:16s} {ww:.5f}")
    np.savez_compressed("final_future_stack_oof.npz", names=np.array(names), weights=w, combo_oof=combo_oof, combo_pred=combo_pred, y_raw=y)

    sample = pd.read_csv("data/sample_submission.csv")
    base = pd.read_csv("sub_v12_main.csv")[T].values.astype(np.float64)
    scaled = np.clip(combo_pred * (base.mean() / max(combo_pred.mean(), 1e-9)), 0, None)
    save_submission(sample, scaled, "sub_v28_future_stack_scaled.csv")
    for wb in [0.25, 0.30, 0.35, 0.40, 0.42, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        pred = (1 - wb) * base + wb * scaled
        save_submission(sample, pred, f"sub_v28_v12{int(round((1-wb)*100)):02d}_fp{int(round(wb*100)):02d}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain stages even if their npz artifacts exist")
    parser.add_argument("--only-final", action="store_true", help="only rebuild final_future_stack_oof and submissions")
    args = parser.parse_args()
    dev = {"device": "gpu", "gpu_use_dp": False} if try_gpu() else {}
    log(f"LGB device={'GPU' if dev else 'CPU'}")
    if not args.only_final:
        stage_v25(args.force, dev)
        stage_v27(args.force, dev)
        stage_v26(args.force, dev)
    stage_final()


if __name__ == "__main__":
    main()
