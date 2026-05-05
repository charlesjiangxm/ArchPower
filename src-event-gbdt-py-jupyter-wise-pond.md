# Plan: Convert `src/event_gbdt.py` → Jupyter notebook with analysis cells

## Context
User wants `src/event_gbdt.py` refactored into an interactive Jupyter notebook so the 12-model training pipeline can be explored cell-by-cell. On top of the existing "train + evaluate + write report" flow, four new analysis sections are added:

1. **Predicted-vs-true scatter** per component (train + test on same axes, y=true, x=predict).
2. **Feature importance** — XGBoost `gain`, `cover`, and `weight` (user's "gini index" interpreted as this family; XGBoost regression trees don't use gini impurity directly, but the gain-based importance is the standard regression analog).
3. **Tree visualization** — render a selected tree with feature-named nodes so the reader can see what each split/leaf contains.
4. **SHAP attribution** — TreeExplainer-based global and per-component feature attribution.

Goal: one notebook that preserves every current behavior of `event_gbdt.py` (same metrics, same report) and layers the four new sections in-place, with all artifacts saved to `output/gbdt_<arch>/` as today.

## Target file
- **New**: `src/event_gbdt.ipynb`
- **Existing `src/event_gbdt.py`**: leave in place (still useful as a CLI). Delete only if the user asks.

## Dependencies
| Package | Status | Purpose |
|---|---|---|
| numpy, xgboost, matplotlib | already used | core |
| `shap` | **NOT installed** — user must `pip install shap` | item 5 |
| `graphviz` (Python + system binary) | **NOT installed** — user must `pip install graphviz` + `brew install graphviz` | `xgb.plot_tree` for item 3 |

If graphviz isn't available at runtime, the tree-viz cell falls back to a text dump via `booster.get_dump(dump_format="json")` so the notebook still runs end-to-end.

## Key files to read / reuse (do not duplicate)
- `src/event_utils.py:15` — `COMP_NAMES` (12 component names)
- `src/event_utils.py:34` — `prepare_data_numpy(arch, val_ratio, seed)` — single source of truth for splits/normalization; reuse as-is
- `src/event_utils.py:86` — `denormalize(pred_norm, label_mean, label_std)`
- `src/event_utils.py:91` — `compute_metrics(pred_orig, true_orig)` → dict of rmse/mape/r2
- `src/event_utils.py:120` — `generate_report(...)` — report writer; reuse as-is
- `src/event_attention.py:47-133` — `EVENT_NAMES` (81 event-feature names). Import directly (`from event_attention import EVENT_NAMES`) to avoid copy-paste drift.
- `src/event_gbdt.py` — reference for training hyperparameters and report arguments

## Notebook layout (cells)

1. **[markdown]** Title + one-paragraph description (mirror the docstring of `event_gbdt.py`).

2. **[code] Imports**
   ```python
   import os, numpy as np, xgboost as xgb, matplotlib.pyplot as plt, shap
   from event_utils import COMP_NAMES, prepare_data_numpy, denormalize, compute_metrics, generate_report
   from event_attention import EVENT_NAMES
   ```

3. **[code] Config** — top-level parameters replacing argparse:
   `ARCH="BOOM"`, `VAL_RATIO=0.2`, `SEED=42`, `N_ESTIMATORS=100`, `MAX_DEPTH=6`, `LEARNING_RATE=0.3`, `OUT_DIR=f"../output/gbdt_{ARCH}"`, plus viz-only params `VIZ_COMP="Total"`, `VIZ_TREE=0`. `os.makedirs(OUT_DIR, exist_ok=True)`.

4. **[markdown + code] Load data** — call `prepare_data_numpy(ARCH, VAL_RATIO, SEED)`. Print shapes like the original script.

5. **[markdown + code] Train 12 XGBRegressors** — loop identical to `event_gbdt.py:54-65`, but pass `feature_names=EVENT_NAMES` so downstream tools show readable names. Collect models into a list.

6. **[markdown + code] Evaluate** — original `.py` val-set pathway + additionally compute train-set predictions so the residual/scatter cell has both. Print the per-component metrics table exactly as today.

7. **[markdown + code] Predicted-vs-true scatter (NEW, item 1)** — 4×3 subplot grid, one per component. x=prediction, y=true (denormalized). Overlay train (blue dots) and test (orange dots) with a y=x dashed reference line. Save `OUT_DIR/pred_vs_true.png`.

8. **[markdown + code] Feature importance (NEW, items 2 & 4 + follow-up)** — for each of the 12 models and for each `importance_type ∈ {"gain", "cover", "weight"}`:
   - Extract via `model.get_booster().get_score(importance_type=...)`, dense-fill to length-81 vector.
   - Produce (a) per-component top-15 bar chart (3 subplots side-by-side, one per importance type) → `OUT_DIR/importance_{comp}.png`; (b) a global summary averaging importance across all 12 components → `OUT_DIR/importance_global.png`. Also print the top-10 features per importance type as a markdown-style table in the cell output.

9. **[markdown + code] Tree visualization (NEW, item 3)** — pick `(VIZ_COMP, VIZ_TREE)`. Call `xgb.plot_tree(models[comp_idx], num_trees=VIZ_TREE, rankdir="LR", ax=...)` with a large figure (`figsize=(32, 14)`). Each internal node shows `feature_name < threshold` (feature names come from the `feature_names` set in step 5); leaves show leaf value. Save `OUT_DIR/tree_{VIZ_COMP}_t{VIZ_TREE}.png`. Wrap in try/except for `graphviz` ImportError → fallback prints `booster.get_dump(dump_format="json")[VIZ_TREE]` pretty-printed so the tree structure is still inspectable without graphviz.

10. **[markdown + code] SHAP attribution (NEW, item 5)** — for each of the 12 models:
    - `explainer = shap.TreeExplainer(model)`
    - `sv = explainer.shap_values(X_val)` (normalized features — must match training space)
    - `shap.summary_plot(sv, X_val_raw, feature_names=EVENT_NAMES, show=False)` then save beeswarm → `OUT_DIR/shap_beeswarm_{comp}.png`
    - `shap.summary_plot(sv, X_val_raw, feature_names=EVENT_NAMES, plot_type="bar", show=False)` → `OUT_DIR/shap_bar_{comp}.png`
    Use `X_val_raw` for *display* values (so axes show raw event counts) while `sv` is computed on the normalized inputs the model was trained with — SHAP supports this separation.
    Finally: one global plot aggregating `mean(|sv|)` across the 12 components → `OUT_DIR/shap_global.png`.

11. **[markdown + code] Generate report** — identical `generate_report(...)` call as `event_gbdt.py:82-97`, writing `OUT_DIR/report.md`. No changes.

## Design decisions
- **Notebook at `src/event_gbdt.ipynb`**, not repo root, so `from event_utils import ...` and `from event_attention import EVENT_NAMES` work without `sys.path` hacks; consistent with the CLAUDE.md rule that event scripts run from `src/`.
- **`EVENT_NAMES` imported** from `event_attention.py` rather than copied — single source of truth; matches how `event_utils` is reused across the `.py` scripts.
- **Plots to `OUT_DIR`** (not inline-only) so artifacts are reproducible outside the notebook.
- **SHAP on val set** (not train) — more informative for generalization; consistent with where metrics are reported.
- **Tree viz default = `("Total", 0)`**; exposed as config vars at the top so the reader can change without scrolling.
- **Interpretation of "gini index"**: XGBoost's `gain`/`cover`/`weight` feature importances (all three, since follow-up message asked for them explicitly). This is the standard gradient-boosting analog to sklearn's gini-based feature importance.
- **Keep `.py` script**: untouched; the notebook is additive. User can delete afterward if desired.

## Verification
Run from `src/`:
1. `pip install shap graphviz` and `brew install graphviz` (if on macOS) before first run.
2. Open `src/event_gbdt.ipynb`, **Run All**.
3. Check:
   - Per-component metrics printout matches `python event_gbdt.py --arch BOOM` within RNG tolerance.
   - `output/gbdt_BOOM/report.md` identical to what the `.py` produces.
   - New files exist: `pred_vs_true.png`, `importance_global.png`, `importance_{comp}.png` (×12), `tree_Total_t0.png`, `shap_beeswarm_{comp}.png` (×12), `shap_bar_{comp}.png` (×12), `shap_global.png`.
   - Tree-viz cell renders successfully (or the JSON fallback prints a readable tree).
4. Re-run with `ARCH="XS"` to smoke-test the XiangShan path.
