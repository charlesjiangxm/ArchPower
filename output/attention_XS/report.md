# FT-Transformer (Event Attention) -- XS Power Prediction Report

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
| Model type | Feature Tokenizer Transformer |
| Parameters | 40,220 |
| d_token | 48 |
| n_blocks | 2 |
| n_heads | 4 |
| d_ffn | 64 |
| dropout | 0.3 |
| attn_dropout | 0.2 |
| pretrain_epochs | 200 |
| supervised_epochs | 500 |
| patience | 50 |

## Testing Metrics (original scale)

| Component | MAPE (%) | RMSE | R2 |
|---|---|---|---|
| Total | 5.33 | 0.018183 | 0.9322 |
| BP | 15.75 | 0.007706 | 0.6334 |
| ICache | 27.03 | 0.012007 | 0.6737 |
| IFU | 6.56 | 0.002890 | 0.8777 |
| RNU | 19.29 | 0.000240 | -1.0039 |
| LSU | 6.39 | 0.000096 | 0.7950 |
| DCache | 12.31 | 0.001410 | 0.9233 |
| Regfile | 13.57 | 0.000018 | 0.9464 |
| ISU | 17.55 | 0.000244 | 0.8134 |
| ROB | 10.72 | 0.000019 | 0.0324 |
| FU-Pool | 14.63 | 0.019589 | 0.8983 |
| Others | 44.90 | 0.002613 | 0.9570 |
