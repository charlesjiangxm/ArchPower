# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ArchPower started as an open-source dataset/benchmark suite for **architecture-level CPU power modeling** (200 samples from BOOM and XiangShan RISC-V CPUs; features = 101-d mixing hardware + event params; labels = 12 components × 5 power groups). The repo has since been extended with a second, distinct line of work on **cycle-level c906 power modeling** (waveform-presim → `/Pc(openC906)` power on the openC906 RISC-V core, with ~14k signal-state features per sample), explored via RuleFit and FT-Transformer.

The two efforts share a repo but **do not share code or datasets**. Treat them as two parallel pipelines that both happen to live here.

## Repo Layout

```
src/algorithm-archpwr/   Original ArchPower benchmark models (McPAT-Calib*, PANDA, event_*)
src/algorithm-newalg/    c906 RuleFit / FT-Transformer pipeline + shared FeatureSelector
db/archpwr-db/           ArchPower numpy dataset (200×101 features, 200×12×5 labels)
db/c906-db/              c906 paired presim/pwr pickles (5 workload prefixes)
doc/doc-archpwr/         ArchPower feature & event-dataset docs
doc/doc-newalg/          (currently empty — c906 docs would go here)
doc/papers/              Reference papers (APOLLO, DEEP, ArchPower, FT-Transformer)
output/                  Run artifacts: <model>_<arch>/ or <model>_c906_<split>/
script/train_archpwr.ipynb            User-facing notebook for the archpwr models
load_component_features.ipynb         Root-level archpwr feature-inspection notebook
setup.sh                              Downloads db.zip from Google Drive, unpacks to db/
```

`db/` is not in git — run `./setup.sh` once to fetch it (Google Drive download + unzip).

## Running the Models

### c906 family (active line of work) — run from `src/algorithm-newalg/`

```bash
cd src/algorithm-newalg

# RuleFit — surfaces top rules / interactions
python c906_rulefit.py --split loco          --top_k 1000 --max_rules 2000
python c906_rulefit.py --split time_ordered  --top_k 1000 --max_rules 2000

# FT-Transformer — single-target regression with attention extraction
python c906_ft_transformer.py --split time_ordered --fs_method pearson
python c906_ft_transformer.py --split time_ordered --fs_method mcp

# Sweep all 7 feature-selection methods for FT-Transformer (excludes `sequential`)
./run_ft_transformer_sweep.sh                              # defaults: time_ordered, presim
./run_ft_transformer_sweep.sh --split loco --presim presim_large
```

Outputs:
- `output/rulefit_c906_<split>[_<fs_suffix>]/`
- `output/ft_c906_<split>[_<fs_method>][_<presim_subdir>]/`
- Sweep aggregator bundles per-method runs into `output/<SWEEP_DIR>/<method>/` plus a generated `SUMMARY.md`.

### archpwr family — run from `src/algorithm-archpwr/`

```bash
cd src/algorithm-archpwr

python McPAT-Calib.py              # XGBoost on "Others" features → total power
python McPAT-Calib-Component.py    # 11 per-component XGBoost models
python McPAT-Calib-CompGroup.py    # Per-component × per-power-group
python PANDA.py                    # Physics-aware encode/decode via resource fns

python build_event_dataset.py      # Regenerates db/archpwr-db/event_dataset/*.npy
python event_linear.py    --arch XS
python event_gbdt.py      --arch XS --n_estimators 200
python event_mlp.py       --arch XS --hidden_layers 256,128
python event_attention.py --arch XS                          # FT-Transformer, needs PyTorch
```

**⚠ Path drift:** these scripts still hard-code `'../dataset/...'` (see `DataLoader.py:20`, `event_utils.py:28`, `build_event_dataset.py:172`) while the actual data now lives at `../../db/archpwr-db/`. Running them as-is will fail with FileNotFoundError. Either fix the paths in the shared loader modules (`DataLoader.py`, `event_utils.py`, `build_event_dataset.py`) or add a `src/algorithm-archpwr/../dataset` symlink to `db/archpwr-db`. The c906 scripts correctly resolve to `db/c906-db` via `c906_rulefit_utils._default_base_dir()`.

## Architecture

### Two independent pipelines

**c906 pipeline (`src/algorithm-newalg/`).** Operates on the paired presim/pwr pickles under `db/c906-db/{presim,pwr}/`. Each `<prefix>_func.pkl` (signal states) is row-aligned with `<prefix>_pwr.pkl` (`/Pc(openC906)` target, `time_ps` column). Five workload prefixes: `MMU`, `cache`, `csr`, `exception`, `interrupt`. Two split modes:
- `loco` (leave-one-category-out): probes cross-workload generalization.
- `time_ordered`: per-category 80/20 split by ascending `time_ps`, one model per category.

The presim variant is controlled by `--presim_subdir` (`presim` vs `presim_large`); large encodes unknown signal states as NaN (filled with 0 in `load_c906_pair`).

`feature_selectors.py` provides 8 selection methods behind a unified `FeatureSelector` interface: `pearson` (legacy default), `variance`, `univariate` (sklearn SelectKBest), `rfe`, `from_model` (LassoCV / RandomForest), `sequential` (expensive — excluded from sweeps), `mcp` (APOLLO-style MCP-penalised sparse linear), `deep` (DEEP two-step: MCP prune + forward swap). Standardization is float64 throughout because wide-bus signals (e.g. 320-bit data_in) overflow float32.

`ft_transformer_model.py` is a self-contained PyTorch FT-Transformer (Gorishniy 2021) — single-target regression, numerical-only, with AdamW + CosineAnnealingLR + early stopping; `extract_attention` returns `(cls_attn, ff_attn)`.

**archpwr pipeline (`src/algorithm-archpwr/`).** Two sub-families:
1. **XGBoost benchmarks** (`McPAT-Calib*.py`, `PANDA.py`): `from DataLoader import *` eagerly loads `feature.npy` + per-component feature files at import time and exposes `comp`, `feature_of_components`, `encode_table`. No CLI — loop over hardcoded train/test splits (evenly / small / large) for both archs.
2. **Event-dataset models** (`event_*.py`): use `event_utils.py` (loading, normalization, config-level train/val split, metrics, markdown reports) and `argparse`. All share `prepare_data_numpy(arch, val_ratio, seed)` so models are evaluated on the same validation set.

### archpwr dataset structure

- `db/archpwr-db/feature.npy` — (200, 101) global feature matrix
- `db/archpwr-db/label.npy` — flat; reshape to (200, 12, 5)
- `db/archpwr-db/label_28.npy` — 28nm tech node labels (same shape)
- `db/archpwr-db/component_feature/*.npy` — per-component feature subsets via `component_mask.npy`. `Others.npy` is identical to `feature.npy` (full 101-d).
- `db/archpwr-db/event_dataset/` — hardware-normalized event-only dataset, regenerated by `build_event_dataset.py`.
- `db/archpwr-db/statistics/<config>_<workload>/` — raw gem5 statistics per pair.

Label axes: 12 components = `[Total, BP, ICache, IFU, RNU, LSU, DCache, Regfile, ISU, ROB, FU-Pool, Others]`; 5 power groups = `[total_power, combinational_logic, sequential_logic, memory, clock]`.

Sample layout: 0-119 = BOOM (15 configs × 8 workloads), 120-199 = XS (10 × 8). Splits are done at the **config level** then expanded by ×8 — this prevents workload leakage between configs. Always split on configs, not raw samples.

### Model hierarchy (archpwr, increasing complexity)

1. **McPAT-Calib**: single XGBoost predicting total power from "Others" features.
2. **McPAT-Calib-Component**: 11 independent XGBoost models per component; total = sum.
3. **McPAT-Calib-CompGroup**: per-component × per-power-group (44 models); sum across groups.
4. **PANDA**: adds physics-aware `encode_arch_knowledge` / `decode_arch_knowledge` via resource functions. Labels are divided by hardware-dependent scale factors pre-training and multiplied back post-prediction.
5. **Event baselines** (`event_linear`, `event_gbdt`, `event_mlp`): Ridge / XGBoost / MLP from 81 event features → 12 power components.
6. **Event FT-Transformer** (`event_attention.py`): optional masked-feature pre-training + supervised multi-target regression + attention visualization.

### Key domain concepts (archpwr)

- **Resource functions** (PANDA): hardware-dependent scale factors per component. Multiplicative for most, additive for Regfile, lookup-table for ISU, identity for FU-Pool (no `encode_table` entry). In `PANDA.py`, `logic_bias` and `dtlb_bias` are **effectively 0** because `estimate_bias_logic` / `estimate_bias_dtlb` assign to local variables rather than the module-level ones. This is intentionally preserved — do not "fix" it without understanding why.
- **Event-only dataset**: `build_event_dataset.py` strips the 20 hardware-parameter columns (constant within a config) and divides labels by resource functions to remove hardware dependence. Must be regenerated if raw feature/label files change.
- **Metrics**: XGBoost benchmarks report Pearson R + MAPE on scatter plots. `event_utils.compute_metrics` returns MAPE / RMSE / R² (sklearn `r2_score`) — *not* Pearson R; don't confuse the two.

### Shared modules

- `DataLoader.py` and `event_utils.py` are import-only shared modules — edits affect every dependent script. `prepare_data_numpy` is the single source of truth for the event-model normalization and config-level split.
- `c906_rulefit_utils.py` (`load_c906_pair`, `load_all`, `compute_metrics`, `parse_rule_features`, `refit_nonneg_lasso`) is the c906 equivalent.
- `feature_selectors.py` is shared between `c906_rulefit.py` and `c906_ft_transformer.py`.

### `src/` vs notebooks

`src/algorithm-archpwr/event_gbdt.py` and friends are both importable and runnable; notebooks under `script/` and at the repo root import from `src/algorithm-archpwr/` and write outputs to `output/<model>_<arch>/` (notebook outputs go to the same place as the CLI). Notebooks prepend the relevant `src/` path to `sys.path`.

## Dependencies

- numpy, scikit-learn, xgboost, matplotlib, pandas — most models
- torch — `event_attention.py`, `c906_ft_transformer.py`, `ft_transformer_model.py`
- rulefit — `c906_rulefit.py`
- numba (optional) — `feature_selectors.py` (10–50× speed-up for `mcp` / `deep` methods)
- seaborn — attention map plotting

## Documentation

- `doc/doc-archpwr/feature_description.md` — mapping of all 101 features to gem5 statistics + per-component feature files
- `doc/doc-archpwr/event_dataset_generation.md` — derivation of the event-only dataset and resource functions
- `doc/papers/` — reference PDFs (ArchPower, APOLLO, DEEP, FT-Transformer)
