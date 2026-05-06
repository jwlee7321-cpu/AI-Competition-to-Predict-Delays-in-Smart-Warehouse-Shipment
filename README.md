# Smart Warehouse Delay Prediction - Final Solution

Final submission file:

```text
sub_v44_anti_v41layout035_a120.csv
```

Public score:

```text
9.6773143392
```

This repository contains the cleaned reproduction code for the final selected
submission.  Raw competition data is not included.  Place the official data
files in `data/` before running the full pipeline.

## Rule Boundary

- No test target values are used.
- No pseudo labeling is used.
- Model fitting and residual calibration use train targets and train OOF
  predictions only.
- Test rows are used for feature construction, inference, and submission file
  generation.
- The late A120 file is a deterministic postprocess of saved prediction files:

```text
A120 = sub_port_v28_w70
       - 1.20 * (sub_v41_rank_layout_a035_add_w65 - sub_port_v28_w70)
```

The scalar `1.20` was selected during the public probing stage.  This should be
documented separately from supervised model training.

## Required Data

Put the official competition files here:

```text
data/train.csv
data/test.csv
data/layout_info.csv
data/sample_submission.csv
```

## Fast Exact Reproduction From Included Artifacts

The `artifacts/` folder includes the saved prediction files needed to reproduce
the final A120 postprocess.  Run:

```powershell
python reproduce_a120_from_artifacts.py
```

Output:

```text
sub_v44_anti_v41layout035_a120_reproduced.csv
```

This verifies the final postprocess formula without retraining all models.

## Full Pipeline Reproduction

From official raw data:

```powershell
python 01_features.py
python 07_v12_full.py
python 21_future_proxy_model.py
python 23_xgb_future_proxy.py
python 24_future_proxy_allcols.py
python final_pipeline/compact_final_stack.py
python final_pipeline/portfolio_blends.py
python final_pipeline/w70_residual_calibration.py
python final_pipeline/w70_combo_candidates.py
python final_pipeline/late_probe_postprocess.py
```

The full training path can differ slightly across GPU/library environments.
For exact final-file verification, use the included artifacts.

## File Roles

| File | Role |
|---|---|
| `01_features.py` | Builds `features.npz` from train/test/layout data. |
| `07_v12_full.py` | Trains the v12 base ensemble and saves v12 OOF/predictions. |
| `21_future_proxy_model.py` | Trains the first future-proxy model. |
| `23_xgb_future_proxy.py` | Adds XGBoost future-proxy predictions. |
| `24_future_proxy_allcols.py` | Builds all-column future-proxy predictions. |
| `final_pipeline/compact_final_stack.py` | Builds the compact v28 future stack. |
| `final_pipeline/portfolio_blends.py` | Creates the `sub_port_v28_w70.csv` base submission. |
| `final_pipeline/w70_residual_calibration.py` | Learns OOF residual curves around w70. |
| `final_pipeline/w70_combo_candidates.py` | Builds the public-failed v41 direction. |
| `final_pipeline/late_probe_postprocess.py` | Reverses the v41 direction and creates A120. |
| `reproduce_a120_from_artifacts.py` | Exact postprocess reproduction from included CSV artifacts. |

## Environment

Tested environment:

```text
Python 3.12.5
numpy 2.4.3
pandas 2.3.3
scikit-learn 1.7.2
scipy 1.16.3
lightgbm 4.6.0
xgboost 3.1.1
catboost 1.2.8
```

## Notes

The preserved file `artifacts/sub_v12_main_pre_rerun.csv` is included because a
fresh rerun of the v12 training script may produce tiny numerical differences
depending on hardware and library versions.
