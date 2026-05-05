# XGBoost GBDT (BOOM → BOOM) -- BOOM → BOOM Power Prediction Report

## Data Split

| Property | Value |
|---|---|
| Architecture | BOOM → BOOM |
| Split method | Config-level holdout (val_ratio=1.0) |
| Total configs | 15 |
| Training configs | 0 |
| Testing configs | 15 |
| Samples per config | 8 (workloads) |
| Training samples | 120 |
| Testing samples | 120 |
| Features | 81 (event-only) |
| Targets | 12 (power components) |

## Model

| Property | Value |
|---|---|
| Model type | 12 independent XGBRegressor models, trained on source arch |
| Parameters | N/A (tree ensemble) |
| train_arch | BOOM |
| test_arch | BOOM |
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.3 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 0.01 | 0.000012 | 1.0000 |
| BP | 0.00 | 0.000002 | 1.0000 |
| ICache | 0.00 | 0.000000 | 1.0000 |
| IFU | 0.04 | 0.000009 | 1.0000 |
| RNU | 0.04 | 0.000002 | 1.0000 |
| LSU | 0.03 | 0.000000 | 1.0000 |
| DCache | 0.02 | 0.000001 | 1.0000 |
| Regfile | 0.02 | 0.000000 | 1.0000 |
| ISU | 0.02 | 0.000000 | 1.0000 |
| ROB | 0.03 | 0.000000 | 1.0000 |
| FU-Pool | 0.02 | 0.000001 | 1.0000 |
| Others | 0.04 | 0.000003 | 1.0000 |
