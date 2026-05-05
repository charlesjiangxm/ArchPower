# Ridge Linear (BOOM → XS) -- BOOM → XS Power Prediction Report

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
| Model type | sklearn Ridge (multi-output), trained on source arch |
| Parameters | 984 |
| train_arch | BOOM |
| test_arch | XS |
| alpha | 1.0 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 63.66 | 0.195981 | -10.8385 |
| BP | 51.59 | 0.017172 | -0.1121 |
| ICache | 87.44 | 0.057998 | -7.5437 |
| IFU | 44.24 | 0.017148 | -3.1792 |
| RNU | 305.69 | 0.003884 | -116.7614 |
| LSU | 27.98 | 0.000553 | -7.0948 |
| DCache | 42.56 | 0.006838 | -0.8259 |
| Regfile | 88.98 | 0.000173 | -4.2159 |
| ISU | 75.11 | 0.001095 | -2.5784 |
| ROB | 37.43 | 0.000067 | -4.5685 |
| FU-Pool | 93.22 | 0.127324 | -5.0583 |
| Others | 105.58 | 0.011711 | -0.2903 |
