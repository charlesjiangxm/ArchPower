# FT-Transformer (Event Attention) -- BOOM Power Prediction Report

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
| Model type | Feature Tokenizer Transformer |
| Parameters | 22,764 |
| d_token | 32 |
| n_blocks | 2 |
| n_heads | 4 |
| d_ffn | 64 |
| dropout | 0.3 |
| attn_dropout | 0.2 |
| pretrain_epochs | 20 |
| supervised_epochs | 50 |
| patience | 10 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 3.61 | 0.004709 | 0.9253 |
| BP | 1.04 | 0.000629 | 0.9723 |
| ICache | 1.50 | 0.000114 | 0.9445 |
| IFU | 14.90 | 0.005110 | 0.8793 |
| RNU | 28.71 | 0.001732 | 0.6268 |
| LSU | 81.44 | 0.000537 | -4.8267 |
| DCache | 13.89 | 0.001304 | 0.5845 |
| Regfile | 14.34 | 0.000003 | 0.7586 |
| ISU | 14.74 | 0.000044 | 0.7822 |
| ROB | 18.25 | 0.000048 | 0.3567 |
| FU-Pool | 12.28 | 0.000873 | 0.8158 |
| Others | 30.51 | 0.002903 | -1.2065 |
