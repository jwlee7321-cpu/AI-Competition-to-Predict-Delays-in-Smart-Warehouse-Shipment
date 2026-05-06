from __future__ import annotations

import os

import numpy as np
import pandas as pd

T = "avg_delay_minutes_next_30m"


def load_pred(path: str) -> np.ndarray:
    return pd.read_csv(path)[T].to_numpy(dtype=np.float64)


def save(sample: pd.DataFrame, base: np.ndarray, pred: np.ndarray, name: str):
    pred = np.clip(pred, 0, None)
    pred += base.mean() - pred.mean()
    pred = np.clip(pred, 0, None)
    out = sample.copy()
    out[T] = pred.astype(np.float32)
    out.to_csv(name, index=False)
    d = pred - base
    return {
        "file": name,
        "mean": float(pred.mean()),
        "std": float(pred.std()),
        "mad_vs_best": float(np.mean(np.abs(d))),
        "maxabs_vs_best": float(np.max(np.abs(d))),
        "corr_vs_best": float(np.corrcoef(base, pred)[0, 1]),
        "n_gt1": int((np.abs(d) > 1).sum()),
        "n_gt2": int((np.abs(d) > 2).sum()),
        "n_gt5": int((np.abs(d) > 5).sum()),
    }


def ensure_w65_line(sample: pd.DataFrame, best: np.ndarray):
    path = "sub_v39_v28line_w065.csv"
    if os.path.exists(path):
        return
    v12_path = "sub_v12_main_pre_rerun.csv" if os.path.exists("sub_v12_main_pre_rerun.csv") else "sub_v12_main.csv"
    future_path = "sub_v28_future_stack_scaled.csv"
    if not os.path.exists(v12_path) or not os.path.exists(future_path):
        return
    v12 = load_pred(v12_path)
    future = load_pred(future_path)
    save(sample, best, 0.35 * v12 + 0.65 * future, path)


def main():
    sample = pd.read_csv("data/sample_submission.csv")
    best = load_pred("sub_port_v28_w70.csv")
    rows = []
    ensure_w65_line(sample, best)

    refs = {
        "w65": "sub_v39_v28line_w065.csv",
        "v34g70": "sub_port_v34global_w70.csv",
        "v34g75": "sub_port_v34global_w75.csv",
        "v34t70": "sub_port_v34time_w70.csv",
    }
    ref_delta = {}
    for tag, path in refs.items():
        if os.path.exists(path):
            ref_delta[tag] = load_pred(path) - best

    rank_files = [
        "sub_v40_w70cal_rank_a015.csv",
        "sub_v40_w70cal_rank_a025.csv",
        "sub_v40_w70cal_rank_a035.csv",
        "sub_v40_w70cal_rank_a050.csv",
        "sub_v40_w70cal_rank_a065.csv",
        "sub_v40_w70cal_rank_a080.csv",
        "sub_v40_w70cal_rank_layout_a025.csv",
        "sub_v40_w70cal_rank_layout_a035.csv",
        "sub_v40_w70cal_rank_layout_a050.csv",
        "sub_v40_w70cal_rank_layout_a065.csv",
        "sub_v40_w70cal_rank_layout_a080.csv",
        "sub_v40_w70cal_rank_time3_a050.csv",
        "sub_v40_w70cal_rank_time3_a065.csv",
        "sub_v40_w70cal_rank_timeslot_a050.csv",
        "sub_v40_w70cal_rank_timeslot_a065.csv",
    ]

    for path in rank_files:
        if not os.path.exists(path):
            continue
        tag = path.replace("sub_v40_w70cal_", "").replace(".csv", "")
        d_rank = load_pred(path) - best

        if "w65" in ref_delta:
            rows.append(save(sample, best, best + d_rank + ref_delta["w65"], f"sub_v41_{tag}_add_w65.csv"))
            rows.append(save(sample, best, best + d_rank + 0.5 * ref_delta["w65"], f"sub_v41_{tag}_add_w65half.csv"))

        for ref_tag in ["v34g70", "v34g75", "v34t70"]:
            if ref_tag in ref_delta:
                rows.append(
                    save(sample, best, best + d_rank + 0.5 * ref_delta[ref_tag], f"sub_v41_{tag}_add_{ref_tag}half.csv")
                )

        if tag in ["rank_a025", "rank_a035"] and "v34g70" in ref_delta:
            rows.append(save(sample, best, best + d_rank + ref_delta["v34g70"], f"sub_v41_{tag}_add_v34g70.csv"))

    report = pd.DataFrame(rows).sort_values(["mad_vs_best", "maxabs_vs_best"])
    report.to_csv("v41_combo_candidate_report.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
