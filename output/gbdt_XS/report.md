# XGBoost GBDT -- XS Power Prediction Report

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
| Model type | 12 independent XGBRegressor models (one per component) |
| Parameters | N/A (tree ensemble) |
| n_estimators | 100 |
| max_depth | 6 |
| learning_rate | 0.3 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 4.26 | 0.016390 | 0.9449 |
| BP | 10.97 | 0.005121 | 0.8381 |
| ICache | 22.95 | 0.009938 | 0.7764 |
| IFU | 8.92 | 0.004184 | 0.7437 |
| RNU | 18.52 | 0.000312 | -2.3791 |
| LSU | 5.38 | 0.000093 | 0.8060 |
| DCache | 8.73 | 0.001490 | 0.9143 |
| Regfile | 9.42 | 0.000011 | 0.9784 |
| ISU | 12.36 | 0.000267 | 0.7758 |
| ROB | 12.65 | 0.000022 | -0.2929 |
| FU-Pool | 1.81 | 0.002588 | 0.9982 |
| Others | 73.71 | 0.002428 | 0.9629 |
