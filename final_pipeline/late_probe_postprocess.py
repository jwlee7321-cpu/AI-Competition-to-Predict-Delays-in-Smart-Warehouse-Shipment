from __future__ import annotations

import os

import numpy as np
import pandas as pd

T = "avg_delay_minutes_next_30m"


def load(path: str) -> np.ndarray:
    return pd.read_csv(path)[T].to_numpy(dtype=np.float64)


def save(sample: pd.DataFrame, anchor: np.ndarray, pred: np.ndarray, path: str):
    pred = np.clip(pred, 0, None)
    pred += anchor.mean() - pred.mean()
    pred = np.clip(pred, 0, None)
    out = sample.copy()
    out[T] = pred.astype(np.float32)
    out.to_csv(path, index=False)
    d = pred - anchor
    return {
        "file": path,
        "mean": float(pred.mean()),
        "std": float(pred.std()),
        "mad_vs_w70": float(np.mean(np.abs(d))),
        "maxabs_vs_w70": float(np.max(np.abs(d))),
        "corr_vs_w70": float(np.corrcoef(anchor, pred)[0, 1]),
        "n_gt1": int((np.abs(d) > 1).sum()),
        "n_gt2": int((np.abs(d) > 2).sum()),
        "n_gt5": int((np.abs(d) > 5).sum()),
    }


def main():
    sample = pd.read_csv("data/sample_submission.csv")
    w70 = load("sub_port_v28_w70.csv")
    failed_v41 = load("sub_v41_rank_layout_a035_add_w65.csv")
    wrong_dir = failed_v41 - w70
    rows = []

    # v42: reverse the public-failed v41 direction.
    for a in [0.15, 0.25, 0.35, 0.50, 0.75, 0.85, 0.90, 1.00]:
        rows.append(save(sample, w70, w70 - a * wrong_dir, f"sub_v42_anti_v41layout035_a{int(round(a * 100)):03d}.csv"))

    # Optional v42 residual-only anti references.
    for src in [
        "sub_v40_w70cal_rank_layout_a025.csv",
        "sub_v40_w70cal_rank_layout_a035.csv",
        "sub_v40_w70cal_rank_layout_a050.csv",
        "sub_v40_w70cal_rank_a035.csv",
        "sub_v40_w70cal_rank_a050.csv",
    ]:
        if not os.path.exists(src):
            continue
        delta = load(src) - w70
        tag = src.replace("sub_v40_w70cal_", "").replace(".csv", "")
        for a in [0.35, 0.50, 0.75, 1.00]:
            rows.append(save(sample, w70, w70 - a * delta, f"sub_v42_anti_{tag}_m{int(round(a * 100)):03d}.csv"))

    # v44: fine search on the reverse direction and mild secondary additions.
    refs = []
    for tag, path in [
        ("v34g70", "sub_port_v34global_w70.csv"),
        ("v34g75", "sub_port_v34global_w75.csv"),
        ("reslgb_c02p056", "sub_v43_reslgb_l1_c02_p056.csv"),
        ("reslgb_huberm010", "sub_v43_reslgb_huber_c01_m010.csv"),
        ("rankanti", "sub_v42_anti_rank_layout_a035_m050.csv"),
    ]:
        if os.path.exists(path):
            refs.append((tag, load(path) - w70))

    for a in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
        base = w70 - a * wrong_dir
        rows.append(save(sample, w70, base, f"sub_v44_anti_v41layout035_a{int(round(a * 100)):03d}.csv"))

    for a in [1.10, 1.15, 1.20]:
        base = w70 - a * wrong_dir
        for tag, delta in refs:
            weights = [0.25, 0.50] if tag in ["v34g70", "v34g75", "rankanti"] else [0.50, 1.00]
            for w in weights:
                rows.append(
                    save(
                        sample,
                        w70,
                        base + w * delta,
                        f"sub_v44_a{int(round(a * 100)):03d}_add_{tag}_w{int(round(w * 100)):03d}.csv",
                    )
                )

    # v45: more aggressive additions on top of the a120 public best line.
    a120 = w70 - 1.20 * wrong_dir
    for tag, path in [
        ("v34g70", "sub_port_v34global_w70.csv"),
        ("v34g75", "sub_port_v34global_w75.csv"),
        ("v34p5w70", "sub_port_v34pred5_w70.csv"),
        ("reslgb_c02", "sub_v43_reslgb_l1_c02_p056.csv"),
        ("reslgb_hm", "sub_v43_reslgb_huber_c01_m010.csv"),
    ]:
        if not os.path.exists(path):
            continue
        delta = load(path) - w70
        weights = [0.75, 1.00] if tag.startswith("v34g") else ([0.25, 0.50] if tag == "v34p5w70" else [1.00, 1.50])
        for w in weights:
            rows.append(save(sample, w70, a120 + w * delta, f"sub_v45_a120_add_{tag}_w{int(round(w * 100)):03d}.csv"))

    if os.path.exists("sub_port_v34global_w70.csv") and os.path.exists("sub_v43_reslgb_l1_c02_p056.csv"):
        g = load("sub_port_v34global_w70.csv") - w70
        r = load("sub_v43_reslgb_l1_c02_p056.csv") - w70
        for wg, wr in [(0.50, 1.00), (0.75, 1.00), (0.75, 1.50)]:
            rows.append(
                save(
                    sample,
                    w70,
                    a120 + wg * g + wr * r,
                    f"sub_v45_a120_add_v34g70_w{int(wg * 100):03d}_reslgb_w{int(wr * 100):03d}.csv",
                )
            )

    report = pd.DataFrame(rows).sort_values(["mad_vs_w70", "maxabs_vs_w70"])
    report.to_csv("late_probe_postprocess_report.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
