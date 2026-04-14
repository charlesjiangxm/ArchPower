# Ridge Regression (Linear) -- XS Power Prediction Report

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
| Model type | sklearn Ridge (multi-output) |
| Parameters | 984 |
| alpha | 1.0 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 5.95 | 0.021540 | 0.9048 |
| BP | 16.53 | 0.008414 | 0.5629 |
| ICache | 20.50 | 0.009180 | 0.8092 |
| IFU | 5.87 | 0.002589 | 0.9018 |
| RNU | 24.14 | 0.000260 | -1.3518 |
| LSU | 5.62 | 0.000090 | 0.8194 |
| DCache | 7.79 | 0.001013 | 0.9604 |
| Regfile | 17.08 | 0.000031 | 0.8303 |
| ISU | 16.16 | 0.000216 | 0.8536 |
| ROB | 9.97 | 0.000019 | 0.0237 |
| FU-Pool | 23.11 | 0.026482 | 0.8142 |
| Others | 69.99 | 0.002484 | 0.9612 |
