# XGBoost GBDT (XS → BOOM) -- XS → BOOM Power Prediction Report

## Data Split

| Property | Value |
|---|---|
| Architecture | XS → BOOM |
| Split method | Config-level holdout (val_ratio=1.0) |
| Total configs | 15 |
| Training configs | 0 |
| Testing configs | 15 |
| Samples per config | 8 (workloads) |
| Training samples | 80 |
| Testing samples | 120 |
| Features | 81 (event-only) |
| Targets | 12 (power components) |

## Model

| Property | Value |
|---|---|
| Model type | 12 independent XGBRegressor models, trained on source arch |
| Parameters | N/A (tree ensemble) |
| train_arch | XS |
| test_arch | BOOM |
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.3 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 187.55 | 0.191784 | -106.0550 |
| BP | 30.25 | 0.016138 | -17.6404 |
| ICache | 811.88 | 0.054658 | -14604.7626 |
| IFU | 100.89 | 0.016769 | -0.7354 |
| RNU | 65.94 | 0.004014 | -1.7318 |
| LSU | 88.74 | 0.000681 | -0.2440 |
| DCache | 78.79 | 0.006006 | -8.3572 |
| Regfile | 1209.14 | 0.000165 | -1059.0449 |
| ISU | 384.30 | 0.001098 | -190.5623 |
| ROB | 75.02 | 0.000067 | -0.5808 |
| FU-Pool | 1570.52 | 0.126654 | -4121.8488 |
| Others | 113.53 | 0.010795 | -4.6792 |
