from __future__ import annotations

import numpy as np
import pandas as pd


TARGET = "avg_delay_minutes_next_30m"
BASE_PATH = "artifacts/sub_port_v28_w70.csv"
FAILED_PATH = "artifacts/sub_v41_rank_layout_a035_add_w65.csv"
REFERENCE_PATH = "artifacts/sub_v44_anti_v41layout035_a120.csv"
OUTPUT_PATH = "sub_v44_anti_v41layout035_a120_reproduced.csv"


def main() -> None:
    base = pd.read_csv(BASE_PATH)
    failed = pd.read_csv(FAILED_PATH)
    ref = pd.read_csv(REFERENCE_PATH)

    pred = base[TARGET].to_numpy(dtype=np.float64)
    failed_pred = failed[TARGET].to_numpy(dtype=np.float64)

    reproduced = np.clip(pred - 1.20 * (failed_pred - pred), 0, None)
    reproduced += pred.mean() - reproduced.mean()
    reproduced = np.clip(reproduced, 0, None).astype(np.float32)

    out = base.copy()
    out[TARGET] = reproduced
    out.to_csv(OUTPUT_PATH, index=False)

    ref_pred = ref[TARGET].to_numpy(dtype=np.float64)
    diff = ref_pred - reproduced.astype(np.float64)
    print(f"saved: {OUTPUT_PATH}")
    print(f"max_abs_diff_vs_reference={np.max(np.abs(diff)):.10f}")
    print(f"mean_abs_diff_vs_reference={np.mean(np.abs(diff)):.10f}")
    print(f"mean={out[TARGET].mean():.10f}")
    print(f"std={out[TARGET].std():.10f}")


if __name__ == "__main__":
    main()
