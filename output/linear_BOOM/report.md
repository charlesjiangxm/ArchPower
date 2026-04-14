# Ridge Regression (Linear) -- BOOM Power Prediction Report

## Data Split

| Property | Value |
|---|---|
| Architecture | BOOM |
| Split method | Config-level holdout (val_ratio=0.2) |
| Total configs | 15 |
| Training configs | 12 |
| Testing configs | 3 |
| Samples per config | 8 (workloads) |
| Training samples | 96 |
| Testing samples | 24 |
| Features | 81 (event-only) |
| Targets | 12 (power components) |

## Model

| Property | Value |
|---|---|
| Model type | sklearn Ridge (multi-output) |
| Parameters | 984 |
| alpha | 1.0 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 4.36 | 0.006537 | 0.8560 |
| BP | 3.41 | 0.002491 | 0.5650 |
| ICache | 3.41 | 0.000353 | 0.4675 |
| IFU | 11.04 | 0.003845 | 0.9317 |
| RNU | 35.28 | 0.001982 | 0.5111 |
| LSU | 51.45 | 0.000363 | -1.6652 |
| DCache | 13.11 | 0.001181 | 0.6597 |
| Regfile | 13.20 | 0.000003 | 0.7673 |
| ISU | 5.06 | 0.000018 | 0.9656 |
| ROB | 33.01 | 0.000050 | 0.2908 |
| FU-Pool | 7.83 | 0.000660 | 0.8946 |
| Others | 27.61 | 0.002902 | -1.2041 |
