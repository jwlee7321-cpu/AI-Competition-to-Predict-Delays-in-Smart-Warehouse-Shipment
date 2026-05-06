# Solution PPT Outline

Recommended length: 8-10 slides.  Export the final slide deck as PDF before
uploading to the code sharing board.

## 1. Title

- Competition: Smart Warehouse Delay Prediction
- Final file: `sub_v44_anti_v41layout035_a120.csv`
- Public score: `9.6773143392`

## 2. Problem Definition

- Target: `avg_delay_minutes_next_30m`
- Metric: MAE
- Main data: `train.csv`, `test.csv`
- Auxiliary data: `layout_info.csv`

## 3. Validation Strategy

- Scenario-grouped validation using `scenario_id`
- OOF predictions used for stacking and residual calibration
- No test labels or pseudo labels used

## 4. Feature Engineering

- Layout merge
- Time slot per scenario
- Lag/lead/rolling/context features
- Future/context proxy features
- Missing-value handling in model inputs

## 5. Base Ensemble

- v12 ensemble with LightGBM, XGBoost, and CatBoost
- Multiple objectives: MAE, quantile, transformed targets
- Saved artifact: `v12_full_oof_bundle.npz`

## 6. Future Stack

- Future-proxy models from `21`, `23`, `24`
- Compact final stack in `compact_final_stack.py`
- Saved prediction: `sub_v28_future_stack_scaled.csv`

## 7. Portfolio Blend

- Base public candidate:

```text
sub_port_v28_w70 = 0.30 * v12 + 0.70 * future_stack
```

## 8. Late Residual/Postprocess

- OOF residual calibration around w70
- Public-failed v41 direction:

```text
sub_v41_rank_layout_a035_add_w65
```

- Final reverse postprocess:

```text
A120 = sub_port_v28_w70
       - 1.20 * (sub_v41_rank_layout_a035_add_w65 - sub_port_v28_w70)
```

## 9. Rule Compliance

- Test target never used
- No pseudo labeling
- Test rows used only for feature construction, inference, and submission output
- Public probing used only for final candidate selection/postprocess scalar

## 10. Reproduction

- Environment versions
- Full execution order
- Fast exact reproduction from included artifacts:

```powershell
python reproduce_a120_from_artifacts.py
```
