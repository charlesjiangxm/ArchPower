# XGBoost GBDT -- BOOM Power Prediction Report

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
| Model type | 12 independent XGBRegressor models (one per component) |
| Parameters | N/A (tree ensemble) |
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.3 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 2.36 | 0.004281 | 0.9382 |
| BP | 1.05 | 0.000803 | 0.9549 |
| ICache | 0.85 | 0.000097 | 0.9602 |
| IFU | 7.63 | 0.002040 | 0.9808 |
| RNU | 39.18 | 0.001737 | 0.6245 |
| LSU | 69.37 | 0.000452 | -3.1200 |
| DCache | 6.70 | 0.000689 | 0.8841 |
| Regfile | 13.24 | 0.000003 | 0.6972 |
| ISU | 7.16 | 0.000024 | 0.9339 |
| ROB | 25.75 | 0.000040 | 0.5474 |
| FU-Pool | 6.40 | 0.000535 | 0.9309 |
| Others | 42.94 | 0.005013 | -5.5768 |
