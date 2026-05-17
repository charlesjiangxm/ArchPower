#!/usr/bin/env bash
# Run the RuleFit x FeatureSelector sweep on c906-db.
#
# Iterates over 7 selectors (sequential is excluded), trains RuleFit on the
# selected columns for each, captures wall time and a stdout log per method,
# then aggregates the per-method output directories into a single sweep folder
# with a generated SUMMARY.md.
#
# Usage (from anywhere):
#   ./run_rulefit_sweep.sh [--split SPLIT] [--presim PRESIM_SUBDIR]
#                          [--sweep-dir SWEEP_DIR]
#                          [--skip-run] [--skip-aggregate]
#                          [--extra "--top_k 1000 --max_rules 2000"]
#
# Defaults:
#   --split        time_ordered
#   --presim       presim
#   --sweep-dir    rulefit_c906_sweep_feature_selection_presim
#
# --skip-run        only aggregates outputs already on disk
# --skip-aggregate  runs the 7 trainings but doesn't bundle them into SWEEP_DIR
# --extra           appended verbatim to every c906_rulefit.py invocation
#                   (e.g. --top_k 1000, --max_rules 2000, --lasso_mode nonneg).
#                   If --lasso_mode nonneg is present, the script tracks the
#                   matching `_nonneg` suffix that c906_rulefit.py writes.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults and CLI parsing
# ---------------------------------------------------------------------------

SPLIT="time_ordered"
PRESIM="presim"
SWEEP_DIR="rulefit_c906_sweep_feature_selection_presim"
SKIP_RUN=0
SKIP_AGG=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)         SPLIT="$2"; shift 2 ;;
    --presim)        PRESIM="$2"; shift 2 ;;
    --sweep-dir)     SWEEP_DIR="$2"; shift 2 ;;
    --skip-run)      SKIP_RUN=1; shift ;;
    --skip-aggregate)SKIP_AGG=1; shift ;;
    --extra)         EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help)
      # Print the header doc block until the first non-comment line.
      awk 'NR>1 && /^[^#]/ {exit} NR>1 {print}' "$0"
      exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

# Methods to sweep (sequential intentionally excluded).
METHODS=(pearson variance univariate rfe from_model mcp deep)

# c906_rulefit.py adds a "_nonneg" suffix when --lasso_mode nonneg is set.
# Peek at EXTRA_ARGS so out_dir_for() points at the right paths.
LASSO_MODE="normal"
if [[ "$EXTRA_ARGS" == *"--lasso_mode nonneg"* ]]; then
  LASSO_MODE="nonneg"
fi

# ---------------------------------------------------------------------------
# Resolve paths relative to this script.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/c906_rulefit.py"
OUTPUT_DIR="$REPO_ROOT/output"
LOG_DIR="$OUTPUT_DIR/_rulefit_sweep_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

echo "============================================================"
echo "  RuleFit x FeatureSelector sweep"
echo "  split=$SPLIT  presim=$PRESIM  lasso_mode=$LASSO_MODE"
echo "  sweep-dir=$SWEEP_DIR"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra=$EXTRA_ARGS"
echo "  runner=$RUNNER"
echo "============================================================"

# Returns the output directory c906_rulefit.py uses for a given method.
# Logic must mirror c906_rulefit.py:main().
out_dir_for() {
  local method="$1"
  local suffix=""
  [[ "$method"     != "pearson" ]] && suffix+="_$method"
  [[ "$LASSO_MODE" != "normal"  ]] && suffix+="_$LASSO_MODE"
  [[ "$PRESIM"     != "presim"  ]] && suffix+="_$PRESIM"
  echo "$OUTPUT_DIR/rulefit_c906_${SPLIT}${suffix}"
}

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

WALLTIMES_FILE="$LOG_DIR/walltimes.tsv"
: > "$WALLTIMES_FILE"

if [[ $SKIP_RUN -eq 0 ]]; then
  for method in "${METHODS[@]}"; do
    log_file="$LOG_DIR/${method}.log"
    echo
    echo "------------------------------------------------------------"
    echo "  Running method=$method  (log: $log_file)"
    echo "------------------------------------------------------------"
    t0=$(date +%s)
    # shellcheck disable=SC2086  # we want word-splitting on $EXTRA_ARGS
    python "$RUNNER" \
        --split "$SPLIT" \
        --fs_method "$method" \
        --presim_subdir "$PRESIM" \
        $EXTRA_ARGS \
        > "$log_file" 2>&1
    rc=$?
    t1=$(date +%s)
    secs=$((t1 - t0))
    if [[ $rc -ne 0 ]]; then
      echo "  FAILED (exit $rc). Tail of log:"
      tail -30 "$log_file"
      exit $rc
    fi
    printf "%s\t%ds\n" "$method" "$secs" | tee -a "$WALLTIMES_FILE"
  done
else
  echo "(--skip-run set: not running training)"
fi

# ---------------------------------------------------------------------------
# Aggregate into sweep dir with SUMMARY.md
# ---------------------------------------------------------------------------

if [[ $SKIP_AGG -eq 0 ]]; then
  SWEEP_PATH="$OUTPUT_DIR/$SWEEP_DIR"
  echo
  echo "------------------------------------------------------------"
  echo "  Aggregating into $SWEEP_PATH/"
  echo "------------------------------------------------------------"
  mkdir -p "$SWEEP_PATH"
  for method in "${METHODS[@]}"; do
    src="$(out_dir_for "$method")"
    dst="$SWEEP_PATH/$method"
    if [[ ! -d "$src" ]]; then
      echo "  SKIP $method: source dir $src not found"
      continue
    fi
    if [[ -d "$dst" ]]; then
      echo "  RM   existing $dst"
      rm -rf "$dst"
    fi
    mv "$src" "$dst"
    echo "  MV   $src -> $dst"
  done

  # SUMMARY.md generation via inline Python.
  python - "$SWEEP_PATH" "$WALLTIMES_FILE" "$SPLIT" "$PRESIM" "$LASSO_MODE" <<'PY'
import os, sys
sweep, walls_file, split, presim, lasso_mode = sys.argv[1:6]
methods = ["pearson","variance","univariate","rfe","from_model","mcp","deep"]
categories_order = ["MMU","cache","csr","exception","interrupt"]

# Wall times.
walls = {}
if os.path.exists(walls_file):
    for ln in open(walls_file):
        ln = ln.strip()
        if not ln: continue
        m, s = ln.split("\t")
        walls[m] = int(s.rstrip("s"))

# Per-method, parse the report.md table for test R^2, feats and rules per
# category. rulefit's table is 11 columns (no best_epoch).
test_r2 = {}      # method -> {category -> r2}
feats   = {}      # method -> {category -> feats}
rules   = {}      # method -> {category -> "nz/total"}
for m in methods:
    rep = os.path.join(sweep, m, "report.md")
    if not os.path.exists(rep):
        continue
    test_r2[m] = {}; feats[m] = {}; rules[m] = {}
    in_table = False
    for ln in open(rep):
        if ln.startswith("## Per-fold"):
            in_table = True; continue
        if in_table and ln.startswith("##"):
            in_table = False
        if not in_table: continue
        # Match a data row -- ignore header / separator rows.
        if not ln.startswith("|") or ":" in ln or "label" in ln:
            continue
        parts = [p.strip() for p in ln.strip().strip("|").split("|")]
        if len(parts) < 11: continue
        try:
            label   = parts[0]
            fkept   = int(parts[3])
            nz_tot  = parts[4]
            r2_te   = float(parts[10])
        except ValueError:
            continue
        test_r2[m][label] = r2_te
        feats[m][label]   = fkept
        rules[m][label]   = nz_tot

# Compose SUMMARY.md.
out = []
out.append(f"# RuleFit on c906 -- feature-selection sweep")
out.append("")
out.append(f"Sweep settings: `--split {split}` `--presim_subdir {presim}` "
           f"`--lasso_mode {lasso_mode}`, RuleFit defaults from c906_rulefit.py. "
           f"`sequential` is intentionally excluded.")
out.append("")
out.append("## Wall-time per method\n")
out.append("| Method | Wall time | Subfolder |")
out.append("|---|---:|---|")
for m in methods:
    w = walls.get(m, "?")
    w_str = "-"
    if isinstance(w, int):
        mins = w/60.0
        w_str = f"**{w} s ({mins:.1f} min)**"
    out.append(f"| {m} | {w_str} | `{m}/` |")
out.append("")
total = sum(v for v in walls.values() if isinstance(v, int))
if total:
    out.append(f"**Total ~ {total/60.0:.1f} min**")
    out.append("")

# Categories actually present in at least one method's table.
cats = [c for c in categories_order if any(c in test_r2.get(m, {}) for m in methods)]

# Test-set R^2 table.
if cats:
    out.append("## Test-set R² per category\n")
    out.append("| Method | feats (avg) | " + " | ".join(cats) + " |")
    out.append("|---|---:|" + "|".join(["---:"] * len(cats)) + "|")
    for m in methods:
        if m not in test_r2: continue
        f_avg = sum(feats[m].values()) / max(len(feats[m]), 1) if feats.get(m) else 0
        row = [m, f"{f_avg:.0f}"]
        for c in cats:
            v = test_r2[m].get(c)
            row.append(f"{v:.3f}" if v is not None else "-")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("> Per-fold R² is read from each method's `report.md`. Full "
               "metric tables (train/test RMSE, MAPE, R²) and rule listings "
               "live in the per-method subfolders.")
    out.append("")

# Rules (nonzero / total) table -- the headline diagnostic for RuleFit.
if cats:
    out.append("## Rules (nonzero / total) per category\n")
    out.append("| Method | " + " | ".join(cats) + " |")
    out.append("|---|" + "|".join(["---:"] * len(cats)) + "|")
    for m in methods:
        if m not in rules: continue
        row = [m]
        for c in cats:
            row.append(rules[m].get(c, "-"))
        out.append("| " + " | ".join(row) + " |")
    out.append("")

out.append("## Per-method outputs\n")
out.append("Each subfolder contains: `report.md`, per-category subdirs with "
           "`rules.csv`, `top_rules.png`, `top_features.png`, "
           "`interaction_heatmap.png`, `pred_vs_true.png`, "
           "`selected_features.pkl`, plus a top-level `global/` with "
           "`top_features.png` and `interaction_heatmap.png` aggregated across "
           "folds/categories.")
out.append("")
out.append("Reproduce a single method from the repo root:")
out.append("```")
out.append("python src/algorithm-newalg/c906_rulefit.py \\")
out.append(f"  --split {split} --fs_method <method> --presim_subdir {presim}"
           + (f" --lasso_mode {lasso_mode}" if lasso_mode != "normal" else ""))
out.append("```")

with open(os.path.join(sweep, "SUMMARY.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print(f"  Wrote {os.path.join(sweep, 'SUMMARY.md')}")
PY

  echo
  echo "  Sweep aggregated. Top-level layout:"
  ls -1 "$SWEEP_PATH"
else
  echo "(--skip-aggregate set: outputs left in their per-method dirs)"
fi

echo
echo "Wall-time summary:"
column -t -s "$(printf '\t')" "$WALLTIMES_FILE" 2>/dev/null || cat "$WALLTIMES_FILE"
echo "Done."
