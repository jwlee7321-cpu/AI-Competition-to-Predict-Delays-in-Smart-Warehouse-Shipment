from __future__ import annotations

import os

import numpy as np
import pandas as pd


T = "avg_delay_minutes_next_30m"


def save(sample: pd.DataFrame, pred: np.ndarray, path: str) -> None:
    out = sample.copy()
    out[T] = np.clip(pred, 0, None).astype(np.float32)
    out.to_csv(path, index=False)
    print(f"{path:40s} mean={out[T].mean():.4f} std={out[T].std():.4f} max={out[T].max():.2f}")


def load_pred(path: str) -> np.ndarray:
    return pd.read_csv(path)[T].to_numpy(dtype=np.float64)


def blend(base: np.ndarray, alt: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * base + w * alt


def mean_rescale(pred: np.ndarray, target_mean: float) -> np.ndarray:
    return np.clip(pred * (target_mean / max(pred.mean(), 1e-9)), 0, None)


def segmented_blend(
    base: np.ndarray,
    alt: np.ndarray,
    seen_mask: np.ndarray,
    w_seen: float,
    w_unseen: float,
    target_mean: float,
) -> np.ndarray:
    pred = base.copy()
    pred[seen_mask] = blend(base[seen_mask], alt[seen_mask], w_seen)
    pred[~seen_mask] = blend(base[~seen_mask], alt[~seen_mask], w_unseen)
    return mean_rescale(pred, target_mean)


def main() -> None:
    sample = pd.read_csv("data/sample_submission.csv")
    train_layout = pd.read_csv("data/train.csv", usecols=["layout_id"])
    test_layout = pd.read_csv("data/test.csv", usecols=["layout_id"])
    seen_mask = test_layout["layout_id"].isin(set(train_layout["layout_id"])).to_numpy()

    base = load_pred("sub_v12_main.csv")
    sources = [
        ("v28", "sub_v28_future_stack_scaled.csv"),
        ("v34pred5", "sub_v34_predbin5_scaled.csv"),
        ("v34time", "sub_v34_timeslot_scaled.csv"),
        ("v34global", "sub_v34_global_scaled.csv"),
    ]

    for tag, path in sources:
        if not os.path.exists(path):
            continue
        alt = load_pred(path)
        for w in (0.70, 0.75, 0.80, 0.85):
            save(sample, blend(base, alt, w), f"sub_port_{tag}_w{int(w*100):02d}.csv")

    if os.path.exists("sub_v28_future_stack_scaled.csv"):
        v28 = load_pred("sub_v28_future_stack_scaled.csv")
        # Higher unseen weight pushes the stronger unseen-row correction without
        # moving seen layouts quite as aggressively.  Mean is rescaled back to v12.
        for ws, wu in [(0.60, 0.80), (0.65, 0.85), (0.70, 0.90), (0.75, 0.90)]:
            pred = segmented_blend(base, v28, seen_mask, ws, wu, base.mean())
            save(sample, pred, f"sub_port_v28_seen{int(ws*100):02d}_unseen{int(wu*100):02d}.csv")

        # Symmetric alternative in case seen layouts are the public bottleneck.
        for ws, wu in [(0.80, 0.60), (0.85, 0.65)]:
            pred = segmented_blend(base, v28, seen_mask, ws, wu, base.mean())
            save(sample, pred, f"sub_port_v28_seen{int(ws*100):02d}_unseen{int(wu*100):02d}.csv")


if __name__ == "__main__":
    main()
