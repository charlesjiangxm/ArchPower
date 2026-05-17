#!/usr/bin/env bash
# Run the FT-Transformer × FeatureSelector sweep on c906-db.
#
# Iterates over 7 selectors (sequential is excluded), trains FT-Transformer on
# the selected columns for each, captures wall time and a stdout log per
# method, then aggregates the per-method output directories into a single
# sweep folder with a generated SUMMARY.md.
#
# Usage (from anywhere):
#   ./run_ft_transformer_sweep.sh [--split SPLIT] [--presim PRESIM_SUBDIR]
#                                  [--sweep-dir SWEEP_DIR] [--device DEVICE]
#                                  [--skip-run] [--skip-aggregate]
#                                  [--extra "--ft_max_epochs 200"]
#
# Defaults:
#   --split        time_ordered
#   --presim       presim
#   --sweep-dir    ft_transformer_c906_sweep_feature_selection_presim
#   --device       auto    (FT-Transformer auto-selects mps/cuda/cpu)
#
# --skip-run        only aggregates outputs already on disk
# --skip-aggregate  runs the 7 trainings but doesn't bundle them into SWEEP_DIR
# --extra           appended verbatim to every c906_ft_transformer.py invocation
#                   (useful for e.g. --ft_max_epochs 200, --ft_n_blocks 2 etc.)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults and CLI parsing
# ---------------------------------------------------------------------------

SPLIT="time_ordered"
PRESIM="presim"
SWEEP_DIR="ft_transformer_c906_sweep_feature_selection_presim"
DEVICE="auto"
SKIP_RUN=0
SKIP_AGG=0
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --split)         SPLIT="$2"; shift 2 ;;
    --presim)        PRESIM="$2"; shift 2 ;;
    --sweep-dir)     SWEEP_DIR="$2"; shift 2 ;;
    --device)        DEVICE="$2"; shift 2 ;;
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

# ---------------------------------------------------------------------------
# Resolve paths relative to this script.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/c906_ft_transformer.py"
OUTPUT_DIR="$REPO_ROOT/output"
LOG_DIR="$OUTPUT_DIR/_ft_sweep_logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

echo "============================================================"
echo "  FT-Transformer × FeatureSelector sweep"
echo "  split=$SPLIT  presim=$PRESIM  device=$DEVICE"
echo "  sweep-dir=$SWEEP_DIR"
[[ -n "$EXTRA_ARGS" ]] && echo "  extra=$EXTRA_ARGS"
echo "  runner=$RUNNER"
echo "============================================================"

# Returns the output directory c906_ft_transformer.py uses for a given method.
# Logic must mirror c906_ft_transformer.py:main().
out_dir_for() {
  local method="$1"
  local suffix=""
  [[ "$method" != "pearson" ]] && suffix+="_$method"
  [[ "$PRESIM" != "presim" ]]  && suffix+="_$PRESIM"
  echo "$OUTPUT_DIR/ft_c906_${SPLIT}${suffix}"
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
        --ft_device "$DEVICE" \
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
  python - "$SWEEP_PATH" "$WALLTIMES_FILE" "$SPLIT" "$PRESIM" <<'PY'
import os, re, sys
sweep, walls_file, split, presim = sys.argv[1:5]
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

# Per-method, parse the report.md table for test R^2 per category.
metric_re = re.compile(r"^\|\s*(\S+)\s*\|.+\|\s*([-\d.]+)\s*\|\s*[-\d.]+\s*\|\s*([-\d.]+)\s*\|\s*[\d.]+\s*\|\s*[-\d.]+s?\s*\|\s*$")
test_r2 = {}      # method -> {category -> r2}
feats   = {}      # method -> {category -> feats}
best_ep = {}      # method -> {category -> best_epoch}
for m in methods:
    rep = os.path.join(sweep, m, "report.md")
    if not os.path.exists(rep):
        continue
    test_r2[m] = {}; feats[m] = {}; best_ep[m] = {}
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
        if len(parts) < 12: continue
        try:
            label, n_tr, n_te, fkept = parts[0], parts[1], parts[2], int(parts[3])
            r2_te = float(parts[9])
            be    = int(parts[10])
        except ValueError:
            continue
        test_r2[m][label] = r2_te
        feats[m][label]   = fkept
        best_ep[m][label] = be

# Compose SUMMARY.md.
out = []
out.append(f"# FT-Transformer on c906 — feature-selection sweep")
out.append("")
out.append(f"Sweep settings: `--split {split}` `--presim_subdir {presim}`, "
           f"FT-Transformer defaults from c906_ft_transformer.py. "
           f"`sequential` is intentionally excluded.")
out.append("")
out.append("## Wall-time per method\n")
out.append("| Method | Wall time | Subfolder |")
out.append("|---|---:|---|")
for m in methods:
    w = walls.get(m, "?")
    w_str = f"{w} s" if isinstance(w, int) else "—"
    if isinstance(w, int):
        mins = w/60.0
        w_str = f"**{w} s ({mins:.1f} min)**"
    out.append(f"| {m} | {w_str} | `{m}/` |")
out.append("")
total = sum(v for v in walls.values() if isinstance(v, int))
if total:
    out.append(f"**Total ≈ {total/60.0:.1f} min**")
    out.append("")

# Test-set R² table.
cats = [c for c in categories_order if any(c in test_r2.get(m, {}) for m in methods)]
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
            row.append(f"{v:.3f}" if v is not None else "—")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("> Per-fold R² is read from each method's `report.md`. Best-epoch "
               "and full metric tables live in the per-method subfolders.")
    out.append("")

out.append("## Per-method outputs\n")
out.append("Each subfolder contains: `report.md`, per-category subdirs with "
           "`training_curve.png`, `pred_vs_true.png`, `cls_attention_top30.png`, "
           "`interaction_heatmap.png`, `cls_attention.csv`, "
           "`interaction_matrix_top30.csv`, `test_predictions.csv`, `model.pt`, "
           "plus a top-level `global/` aggregating across folds.")
out.append("")
out.append("Reproduce a single method from the repo root:")
out.append("```")
out.append("python src/algorithm-newalg/c906_ft_transformer.py \\")
out.append(f"  --split {split} --fs_method <method> --presim_subdir {presim}")
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
