from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

T = "avg_delay_minutes_next_30m"


def rank_values(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    r = np.empty_like(x, dtype=np.float64)
    r[order] = (np.arange(len(x), dtype=np.float64) + 0.5) / len(x)
    return r


def centered(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    out = values.astype(np.float64).copy()
    if weights is None:
        out -= out.mean()
    else:
        out -= np.average(out, weights=np.maximum(weights, 1.0))
    return out


def make_rank_curve(pred: np.ndarray, y: np.ndarray, n_bins: int = 160, smooth: int = 9):
    r = rank_values(pred)
    bins = np.minimum((r * n_bins).astype(np.int32), n_bins - 1)
    resid = y - pred
    global_med = float(np.median(resid))
    corr = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        m = bins == b
        counts[b] = m.sum()
        corr[b] = float(np.median(resid[m])) if m.any() else global_med
    corr = uniform_filter1d(corr, size=smooth, mode="nearest")
    corr = centered(corr, counts)
    xs = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    return xs, corr


def apply_rank_curve(pred: np.ndarray, xs: np.ndarray, corr: np.ndarray) -> np.ndarray:
    return np.interp(rank_values(pred), xs, corr)


def make_segment_rank_curves(
    pred: np.ndarray,
    y: np.ndarray,
    segment: np.ndarray,
    n_bins: int = 80,
    smooth: int = 7,
    min_rows: int = 700,
):
    curves = {}
    fallback = make_rank_curve(pred, y, n_bins=n_bins, smooth=smooth)
    for s in np.unique(segment):
        m = segment == s
        curves[int(s)] = make_rank_curve(pred[m], y[m], n_bins=n_bins, smooth=smooth) if m.sum() >= min_rows else fallback
    return curves, fallback


def apply_segment_rank_curves(pred: np.ndarray, segment: np.ndarray, curves, fallback) -> np.ndarray:
    out = np.zeros(len(pred), dtype=np.float64)
    for s in np.unique(segment):
        m = segment == s
        xs, corr = curves.get(int(s), fallback)
        out[m] = apply_rank_curve(pred[m], xs, corr)
    return out


def make_bias(resid: np.ndarray, key: np.ndarray, min_rows: int = 500, shrink: float = 1200.0):
    vals = {}
    global_med = float(np.median(resid))
    for k in np.unique(key):
        m = key == k
        n = int(m.sum())
        raw = float(np.median(resid[m])) if n else global_med
        lam = n / (n + shrink)
        vals[k] = lam * raw + (1.0 - lam) * global_med if n >= min_rows else global_med
    mean_val = float(np.mean(list(vals.values()))) if vals else 0.0
    for k in list(vals):
        vals[k] -= mean_val
    return vals


def apply_bias(key: np.ndarray, vals: dict, default: float = 0.0) -> np.ndarray:
    return np.array([vals.get(k, default) for k in key], dtype=np.float64)


def build_train_test_keys():
    train = pd.read_csv("data/train.csv", usecols=["scenario_id", "layout_id", "shift_hour"])
    test = pd.read_csv("data/test.csv", usecols=["scenario_id", "layout_id", "shift_hour"])
    train["timeslot"] = train.groupby("scenario_id").cumcount().astype(np.int16)
    test["timeslot"] = test.groupby("scenario_id").cumcount().astype(np.int16)
    train["time3"] = np.digitize(train["timeslot"].to_numpy(), [8, 17]).astype(np.int16)
    test["time3"] = np.digitize(test["timeslot"].to_numpy(), [8, 17]).astype(np.int16)
    train["hour"] = train["shift_hour"].fillna(-1).round().astype(np.int16)
    test["hour"] = test["shift_hour"].fillna(-1).round().astype(np.int16)
    return train, test


def crossfit_corrections(base_oof: np.ndarray, y: np.ndarray, train: pd.DataFrame):
    groups = train["scenario_id"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(np.arange(len(y)), groups=groups))
    time3 = train["time3"].to_numpy()
    timeslot = train["timeslot"].to_numpy()
    hour = train["hour"].to_numpy()
    layout = train["layout_id"].to_numpy()

    out = {
        "rank": np.zeros(len(y), dtype=np.float64),
        "rank_time3": np.zeros(len(y), dtype=np.float64),
        "rank_timeslot": np.zeros(len(y), dtype=np.float64),
        "rank_hour": np.zeros(len(y), dtype=np.float64),
        "seg_time3": np.zeros(len(y), dtype=np.float64),
        "seg_hour": np.zeros(len(y), dtype=np.float64),
        "rank_layout": np.zeros(len(y), dtype=np.float64),
    }
    for tri, vai in folds:
        xs, corr = make_rank_curve(base_oof[tri], y[tri])
        c_rank_tr = apply_rank_curve(base_oof[tri], xs, corr)
        c_rank_va = apply_rank_curve(base_oof[vai], xs, corr)
        out["rank"][vai] = c_rank_va

        resid_after_rank = y[tri] - (base_oof[tri] + c_rank_tr)
        for name, key in [
            ("rank_time3", time3),
            ("rank_timeslot", timeslot),
            ("rank_hour", hour),
            ("rank_layout", layout),
        ]:
            vals = make_bias(resid_after_rank, key[tri])
            out[name][vai] = c_rank_va + apply_bias(key[vai], vals)

        curves, fallback = make_segment_rank_curves(base_oof[tri], y[tri], time3[tri])
        out["seg_time3"][vai] = apply_segment_rank_curves(base_oof[vai], time3[vai], curves, fallback)
        curves, fallback = make_segment_rank_curves(base_oof[tri], y[tri], hour[tri], min_rows=500)
        out["seg_hour"][vai] = apply_segment_rank_curves(base_oof[vai], hour[vai], curves, fallback)
    return out


def fit_full_corrections(base_oof: np.ndarray, y: np.ndarray, best_pred: np.ndarray, train: pd.DataFrame, test: pd.DataFrame):
    time3 = train["time3"].to_numpy()
    timeslot = train["timeslot"].to_numpy()
    hour = train["hour"].to_numpy()
    layout = train["layout_id"].to_numpy()
    time3_te = test["time3"].to_numpy()
    timeslot_te = test["timeslot"].to_numpy()
    hour_te = test["hour"].to_numpy()
    layout_te = test["layout_id"].to_numpy()

    xs, corr = make_rank_curve(base_oof, y)
    c_train = apply_rank_curve(base_oof, xs, corr)
    c_test = apply_rank_curve(best_pred, xs, corr)
    resid_after_rank = y - (base_oof + c_train)

    out = {"rank": c_test}
    for name, key, key_te in [
        ("rank_time3", time3, time3_te),
        ("rank_timeslot", timeslot, timeslot_te),
        ("rank_hour", hour, hour_te),
        ("rank_layout", layout, layout_te),
    ]:
        vals = make_bias(resid_after_rank, key)
        out[name] = c_test + apply_bias(key_te, vals)

    curves, fallback = make_segment_rank_curves(base_oof, y, time3)
    out["seg_time3"] = apply_segment_rank_curves(best_pred, time3_te, curves, fallback)
    curves, fallback = make_segment_rank_curves(base_oof, y, hour, min_rows=500)
    out["seg_hour"] = apply_segment_rank_curves(best_pred, hour_te, curves, fallback)
    return out


def save_submission(sample: pd.DataFrame, base: np.ndarray, corr: np.ndarray, alpha: float, name: str):
    pred = np.clip(base + alpha * corr, 0, None)
    pred += base.mean() - pred.mean()
    pred = np.clip(pred, 0, None)
    out = sample.copy()
    out[T] = pred.astype(np.float32)
    out.to_csv(name, index=False)
    d = pred - base
    return {
        "file": name,
        "alpha": alpha,
        "mean": float(pred.mean()),
        "std": float(pred.std()),
        "mad_vs_best": float(np.mean(np.abs(d))),
        "maxabs_vs_best": float(np.max(np.abs(d))),
        "n_gt1": int((np.abs(d) > 1).sum()),
        "n_gt2": int((np.abs(d) > 2).sum()),
    }


def main():
    v12 = np.load("v12_full_oof_bundle.npz", allow_pickle=True)
    future = np.load("final_future_stack_oof.npz", allow_pickle=True)
    y = future["y_raw"].astype(np.float64)
    v12_oof = v12["best_hedge_oof"].astype(np.float64)
    future_oof = future["combo_oof"].astype(np.float64)
    base_oof = np.clip(0.30 * v12_oof + 0.70 * future_oof, 0, None)

    best = pd.read_csv("sub_port_v28_w70.csv")
    best_pred = best[T].to_numpy(dtype=np.float64)
    sample = pd.read_csv("data/sample_submission.csv")
    train, test = build_train_test_keys()

    print(f"base w70 OOF={mean_absolute_error(y, base_oof):.6f} mean={base_oof.mean():.5f}")
    cf = crossfit_corrections(base_oof, y, train)
    full = fit_full_corrections(base_oof, y, best_pred, train, test)

    rows = []
    alphas = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80]
    for name, corr in cf.items():
        for alpha in alphas:
            p = np.clip(base_oof + alpha * corr, 0, None)
            rows.append(
                {
                    "name": name,
                    "alpha": alpha,
                    "oof_mae": mean_absolute_error(y, p),
                    "corr_std": float(np.std(corr)),
                }
            )
    report = pd.DataFrame(rows).sort_values("oof_mae")
    report.to_csv("w70_residual_calibration_report.csv", index=False)
    print(report.head(20).to_string(index=False))

    # Emit conservative and attack probes from the best cross-fit variants.
    chosen = []
    for _, row in report.iterrows():
        name = row["name"]
        alpha = float(row["alpha"])
        if alpha <= 0:
            continue
        if name not in [x[0] for x in chosen]:
            chosen.append((name, alpha))
        if len(chosen) >= 4:
            break

    out_rows = []
    for name, best_alpha in chosen:
        for alpha in sorted(
            set([0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, best_alpha])
        ):
            fname = f"sub_v40_w70cal_{name}_a{int(round(alpha * 100)):03d}.csv"
            out_rows.append(save_submission(sample, best_pred, full[name], alpha, fname))

    out_report = pd.DataFrame(out_rows).sort_values(["mad_vs_best", "maxabs_vs_best"])
    out_report.to_csv("w70_residual_submission_report.csv", index=False)
    print("\nSaved candidates:")
    print(out_report.to_string(index=False))


if __name__ == "__main__":
    main()
