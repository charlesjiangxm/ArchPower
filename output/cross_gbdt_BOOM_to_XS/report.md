# XGBoost GBDT (BOOM → XS) -- BOOM → XS Power Prediction Report

## Data Split

| Property | Value |
|---|---|
| Architecture | BOOM → XS |
| Split method | Config-level holdout (val_ratio=1.0) |
| Total configs | 10 |
| Training configs | 0 |
| Testing configs | 10 |
| Samples per config | 8 (workloads) |
| Training samples | 120 |
| Testing samples | 80 |
| Features | 81 (event-only) |
| Targets | 12 (power components) |

## Model

| Property | Value |
|---|---|
| Model type | 12 independent XGBRegressor models, trained on source arch |
| Parameters | N/A (tree ensemble) |
| train_arch | BOOM |
| test_arch | XS |
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.3 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 63.64 | 0.196398 | -10.8890 |
| BP | 50.92 | 0.017665 | -0.1769 |
| ICache | 87.67 | 0.058110 | -7.5768 |
| IFU | 49.21 | 0.017839 | -3.5227 |
| RNU | 307.19 | 0.003867 | -115.7713 |
| LSU | 35.05 | 0.000592 | -8.2713 |
| DCache | 40.10 | 0.006675 | -0.7399 |
| Regfile | 89.42 | 0.000174 | -4.2435 |
| ISU | 74.61 | 0.001091 | -2.5526 |
| ROB | 31.88 | 0.000059 | -3.2496 |
| FU-Pool | 93.63 | 0.127537 | -5.0786 |
| Others | 100.43 | 0.010725 | -0.0822 |
