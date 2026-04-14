# 3-Layer MLP -- BOOM Power Prediction Report

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
| Model type | sklearn MLPRegressor (128, 64, 32) |
| Parameters | 21,228 |
| hidden_layers | (128, 64, 32) |
| activation | relu |
| solver | adam |
| alpha (L2) | 0.001 |
| max_iter | 2000 |
| actual_iter | 260 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 6.29 | 0.010492 | 0.6289 |
| BP | 2.06 | 0.001505 | 0.8413 |
| ICache | 1.74 | 0.000179 | 0.8633 |
| IFU | 18.18 | 0.009619 | 0.5724 |
| RNU | 57.36 | 0.002198 | 0.3991 |
| LSU | 100.57 | 0.000747 | -10.2874 |
| DCache | 6.89 | 0.000606 | 0.9104 |
| Regfile | 19.71 | 0.000003 | 0.6129 |
| ISU | 9.87 | 0.000029 | 0.9084 |
| ROB | 31.29 | 0.000046 | 0.4110 |
| FU-Pool | 7.57 | 0.000563 | 0.9233 |
| Others | 26.83 | 0.003192 | -1.6667 |
