# Artifacts

This folder contains the saved prediction files needed for exact verification
of the final A120 postprocess.

Files:

```text
sub_v12_main_pre_rerun.csv
sub_port_v28_w70.csv
sub_v41_rank_layout_a035_add_w65.csv
sub_v44_anti_v41layout035_a120.csv
```

The final selected submission is:

```text
sub_v44_anti_v41layout035_a120.csv
```

Exact formula:

```text
A120 = sub_port_v28_w70
       - 1.20 * (sub_v41_rank_layout_a035_add_w65 - sub_port_v28_w70)
```
