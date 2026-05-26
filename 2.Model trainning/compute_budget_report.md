# Compute Budget Summary

Final reported LM pretraining runs: 9 runs = 3 conditions x seeds 42, 43, and 44.

Hardware assumption for GPU-hour accounting: each final run used 1 x NVIDIA A100 40GB GPU. Therefore GPU-hours equal wall-clock hours.

The `run_summary.json` files record `wall_clock_seconds`, but do not record absolute start or end timestamps. Start/end are therefore marked as `missing` in `compute_budget_runs.csv`.

## Totals

| Condition | Runs | Wall-clock seconds | GPU-hours |
|---|---:|---:|---:|
| Chinese | 3 | 59999.178709 | 16.666439 |
| Pinyin-token | 3 | 60198.135646 | 16.721704 |
| Pinyin-source | 3 | 67143.210608 | 18.650892 |
| **Total final runs** | **9** | **187340.524964** | **52.039035** |

Total computational budget for the 9 final reported pretraining runs: **52.04 A100 GPU-hours**.
