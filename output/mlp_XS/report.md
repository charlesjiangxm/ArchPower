# 3-Layer MLP -- XS Power Prediction Report

## Data Split

| Property | Value |
|---|---|
| Architecture | XS |
| Split method | Config-level holdout (val_ratio=0.2) |
| Total configs | 10 |
| Training configs | 8 |
| Testing configs | 2 |
| Samples per config | 8 (workloads) |
| Training samples | 64 |
| Testing samples | 16 |
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
| actual_iter | 378 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 7.76 | 0.026332 | 0.8578 |
| BP | 18.40 | 0.009258 | 0.4709 |
| ICache | 25.96 | 0.011819 | 0.6838 |
| IFU | 5.59 | 0.002617 | 0.8997 |
| RNU | 24.50 | 0.000480 | -7.0121 |
| LSU | 8.59 | 0.000141 | 0.5557 |
| DCache | 9.15 | 0.001056 | 0.9570 |
| Regfile | 25.95 | 0.000037 | 0.7588 |
| ISU | 17.48 | 0.000297 | 0.7219 |
| ROB | 10.60 | 0.000018 | 0.0898 |
| FU-Pool | 11.39 | 0.022037 | 0.8714 |
| Others | 95.40 | 0.002895 | 0.9472 |
