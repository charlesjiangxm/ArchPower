"""
RuleFit regression on the c906-db waveform -> /Pc(openC906) power dataset.

Trains a RuleFit model that surfaces the most important features AND their
interactions as human-readable rules.  Two split modes are supported:

* loco          -- leave-one-category-out (5 folds over MMU/cache/csr/
                   exception/interrupt). Probes cross-workload generalization.
* time_ordered  -- per-category 80/20 split by ascending time_ps. Trains one
                   model per category; reports them side-by-side.

Outputs (per split mode) go to ../../output/rulefit_c906_<split>/:
  report.md
  <fold>/
    rules.csv
    top_rules.png
    top_features.png
    interaction_heatmap.png
    pred_vs_true.png
  global/
    top_features.png
    interaction_heatmap.png

Usage (from src/algorithm-newalg/):
  python c906_rulefit.py --split time_ordered --fs_method none
  python c906_rulefit.py --split loco --top_k 1000 --max_rules 2000
  python c906_rulefit.py --split time_ordered --top_k 1000 --max_rules 2000
"""

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rulefit import RuleFit

from c906_rulefit_utils import (
    PREFIXES, load_c906_pair, load_all,
    compute_metrics, parse_rule_features,
    refit_nonneg_lasso, validate_presim_subdir,
)
from feature_selectors import FeatureSelector
from ft_transformer_model import Standardizer


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _shorten(name, max_len=70):
    if len(name) <= max_len:
        return name
    return name[:8] + "..." + name[-(max_len - 11):]


def plot_top_rules(rules_df, out_path, top_n=20):
    """Bar chart of the top-N rules by importance, signed by coef sign."""
    df = rules_df[rules_df["coef"] != 0].copy()
    if df.empty:
        return
    df = df.sort_values("importance", ascending=False).head(top_n)
    labels = [
        f"[{r['type']}] {_shorten(r['rule'], 90)}" for _, r in df.iterrows()
    ]
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in df["coef"]]
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(df) + 1))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df["importance"], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(df)} RuleFit rules (green = positive coef, red = negative)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def aggregate_feature_importance(rules_df, feature_names):
    """Aggregate rule importance into per-feature scores.

    Each rule contributes `importance / n_features_in_rule` to every feature
    it references (linear or rule). Returns a pd.Series indexed by feature
    name, descending.
    """
    scores = {}
    for _, r in rules_df.iterrows():
        if r["coef"] == 0:
            continue
        feats = parse_rule_features(r["rule"], r["type"], feature_names)
        if not feats:
            continue
        share = float(r["importance"]) / len(feats)
        for f in feats:
            scores[f] = scores.get(f, 0.0) + share
    return pd.Series(scores).sort_values(ascending=False)


def plot_top_features(feat_scores, out_path, top_n=30, title="Top features"):
    df = feat_scores.head(top_n)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(df) + 1))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df.values, color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten(n, 80) for n in df.index], fontsize=7)
    ax.set_xlabel("Aggregated importance (sum over rules)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_interaction_matrix(rules_df, top_features):
    """Co-occurrence matrix of `top_features` inside multi-feature rules,
    weighted by rule importance."""
    idx = {f: i for i, f in enumerate(top_features)}
    n = len(top_features)
    M = np.zeros((n, n), dtype=np.float64)
    for _, r in rules_df.iterrows():
        if r["coef"] == 0 or r["type"] != "rule":
            continue
        feats = parse_rule_features(r["rule"], r["type"], top_features)
        # restrict to features in our top set
        feats = [f for f in feats if f in idx]
        if len(feats) < 2:
            continue
        w = float(r["importance"])
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                a, b = idx[feats[i]], idx[feats[j]]
                M[a, b] += w
                M[b, a] += w
    return M


def plot_interaction_heatmap(M, top_features, out_path, title="Feature interactions"):
    if M.size == 0 or M.max() == 0:
        # produce an empty notice plot
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No multi-feature rules to display.",
                ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return
    short = [_shorten(n, 50) for n in top_features]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(M, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="co-occurrence weight (sum importance)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pred_vs_true(y_true, y_pred, out_path, title="Predicted vs True"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True /Pc(openC906)")
    ax.set_ylabel("Predicted")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"{title}\nR² = {r2:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop (one fold or one category)
# ---------------------------------------------------------------------------

def _select_columns(X_train, y_train, X_test, args):
    """Return the columns to train on and elapsed selector time.

    ``--fs_method none`` bypasses FeatureSelector and keeps the raw input
    columns, including constant columns. ``top_k`` is ignored.  RuleFit's
    tree ensemble internally casts to float32, so this bypass mode still drops
    columns whose train/test values cannot be represented in float32; that is
    a model-compatibility guard rather than a scoring/ranking selector.
    """
    t_fs = time.time()
    if args.fs_method == "none":
        cols = list(X_train.columns)

        # Compatibility guard for RuleFit / sklearn GBRT internals.
        f32_max = float(np.finfo(np.float32).max)
        Xtr_raw = X_train[cols].to_numpy(dtype=np.float64, copy=False)
        Xte_raw = X_test[cols].to_numpy(dtype=np.float64, copy=False)
        train_abs_max = np.nanmax(np.abs(Xtr_raw), axis=0)
        test_abs_max = np.nanmax(np.abs(Xte_raw), axis=0)
        abs_max = np.maximum(train_abs_max, test_abs_max)
        safe = np.isfinite(abs_max) & (abs_max < f32_max)
        if not safe.all():
            n_drop = int((~safe).sum())
            cols = [c for c, keep in zip(cols, safe) if keep]
            print(f"  feature selection bypassed: dropped {n_drop} columns "
                  f"that would overflow RuleFit's float32 tree backend")
        if not cols:
            raise RuntimeError(
                "No feature columns remain after RuleFit float32 compatibility "
                "filter in --fs_method none mode."
            )

        fs_seconds = time.time() - t_fs
        print(f"  feature selection bypassed: using {len(cols)} / "
              f"{X_train.shape[1]} features (top_k ignored, {fs_seconds:.1f}s)")
        return cols, fs_seconds

    selector = FeatureSelector.from_args(args)
    cols = selector.fit_select(X_train, y_train)
    fs_seconds = time.time() - t_fs
    print(f"  features kept: {len(cols)} via fs_method={args.fs_method} "
          f"({fs_seconds:.1f}s)")
    return cols, fs_seconds


def run_one(label, X_train, y_train, X_test, y_test, args, out_dir):
    """Train + evaluate + dump artifacts for one fold/category."""
    t0 = time.time()
    print(f"\n=== {label} ===")
    print(f"  train rows: {len(X_train):,}   test rows: {len(X_test):,}")

    cols, _ = _select_columns(X_train, y_train, X_test, args)

    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # Keep raw arrays in float64, then z-score both X and y before giving them
    # to RuleFit.  This is essential for c906: selected bus-value features can
    # be as large as ~1e38.  RuleFit's internal LassoCV builds its alpha grid
    # from the model matrix; with those huge uncentered/raw values the alpha
    # grid becomes enormous and the final sparse linear stage selects all-zero
    # coefficients, i.e. an intercept-only model with train R² = 0.
    Xtr = X_train[cols].to_numpy(dtype=np.float64, copy=False)
    Xte = X_test[cols].to_numpy(dtype=np.float64, copy=False)
    ytr = y_train.to_numpy(dtype=np.float64, copy=False)
    yte = y_test.to_numpy(dtype=np.float64, copy=False)
    std = Standardizer.fit(Xtr, ytr)
    Xtr_z = std.transform_X(Xtr)
    Xte_z = std.transform_X(Xte)
    ytr_z = std.transform_y(ytr)
    print("  preprocessing: z-scored selected X and target y before RuleFit")

    selected_path = os.path.join(label_dir, "selected_features.pkl")
    pd.to_pickle(
        {
            "cols": list(cols),
            "X_train": X_train[cols],
            "y_train": y_train,
            "X_test": X_test[cols],
            "y_test": y_test,
            "fs_method": args.fs_method,
            "preprocessing": "zscore_selected_X_and_y_before_rulefit",
            "rule_units": "standardized_X_thresholds_and_standardized_y_coefficients",
            "x_mean": std.x_mean,
            "x_std": std.x_std,
            "y_mean": std.y_mean,
            "y_std": std.y_std,
        },
        selected_path,
    )
    print(f"  saved selected features -> {selected_path}")

    rf = RuleFit(
        tree_size=args.tree_size,
        max_rules=args.max_rules,
        memory_par=args.memory_par,
        rfmode="regress",
        random_state=args.seed,
    )
    print(f"  fitting RuleFit (tree_size={args.tree_size}, max_rules={args.max_rules}) ...")
    rf.fit(Xtr_z, ytr_z, feature_names=cols)

    if args.lasso_mode == "nonneg":
        print("  refitting Lasso with positive=True (non-negative coefficients) ...")
        refit_nonneg_lasso(rf, Xtr_z, ytr_z, random_state=args.seed)

    yp_tr = std.inverse_y(rf.predict(Xtr_z))
    yp_te = std.inverse_y(rf.predict(Xte_z))
    m_tr = compute_metrics(ytr, yp_tr)
    m_te = compute_metrics(yte, yp_te)
    print(f"  train: RMSE={m_tr['rmse']:.5f}  MAPE={m_tr['mape']:.2f}%  R²={m_tr['r2']:.4f}")
    print(f"  test : RMSE={m_te['rmse']:.5f}  MAPE={m_te['mape']:.2f}%  R²={m_te['r2']:.4f}")

    rules = rf.get_rules()
    rules = rules.sort_values("importance", ascending=False).reset_index(drop=True)
    nonzero = rules[rules["coef"] != 0]
    print(f"  rules: total={len(rules)}  nonzero coef={len(nonzero)}  "
          f"linear nonzero={(nonzero['type']=='linear').sum()}  "
          f"rule nonzero={(nonzero['type']=='rule').sum()}")

    rules.to_csv(os.path.join(label_dir, "rules.csv"), index=False)
    plot_top_rules(rules, os.path.join(label_dir, "top_rules.png"))

    feat_scores = aggregate_feature_importance(rules, cols)
    plot_top_features(
        feat_scores, os.path.join(label_dir, "top_features.png"),
        title=f"Top features ({label}) -- aggregated rule importance",
    )

    top30 = feat_scores.head(30).index.tolist()
    M = build_interaction_matrix(rules, top30)
    plot_interaction_heatmap(
        M, top30, os.path.join(label_dir, "interaction_heatmap.png"),
        title=f"Feature interactions ({label}) -- top-30 co-occurrence",
    )

    plot_pred_vs_true(
        yte, yp_te, os.path.join(label_dir, "pred_vs_true.png"),
        title=f"Predicted vs True ({label}, test)",
    )

    print(f"  done in {time.time()-t0:.1f}s")
    return {
        "label": label,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_kept": len(cols),
        "n_rules_total": int(len(rules)),
        "n_rules_nonzero": int(len(nonzero)),
        "metrics_train": m_tr,
        "metrics_test": m_te,
        "feat_scores": feat_scores,
        "interaction_top30": top30,
        "interaction_matrix": M,
    }


# ---------------------------------------------------------------------------
# Split drivers
# ---------------------------------------------------------------------------

def driver_loco(args, out_dir):
    """Load 5 prefixes once into a list, then per-fold concat the 4 train
    prefixes on demand. Keeps peak memory lower than concatenating everything
    upfront and then re-slicing per fold."""
    print(f"Loading all 5 prefix pairs from presim_subdir={args.presim_subdir!r} ...")
    pairs = {}
    for p in PREFIXES:
        X, y, _ = load_c906_pair(p, presim_subdir=args.presim_subdir)
        pairs[p] = (X, y)
        print(f"  {p}: X={X.shape}")
    results = []
    for held in PREFIXES:
        train_prefixes = [p for p in PREFIXES if p != held]
        Xtr = pd.concat([pairs[p][0] for p in train_prefixes], ignore_index=True)
        ytr = pd.concat([pairs[p][1] for p in train_prefixes], ignore_index=True)
        Xte, yte = pairs[held]
        results.append(run_one(f"held_{held}", Xtr, ytr, Xte, yte, args, out_dir))
        del Xtr, ytr
    return results


def driver_time_ordered(args, out_dir):
    print(f"Loading prefixes from presim_subdir={args.presim_subdir!r} ...")
    results = []
    for pref in PREFIXES:
        X, y, t = load_c906_pair(pref, presim_subdir=args.presim_subdir)
        order = np.argsort(t.to_numpy())
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        n = len(X)
        n_test = max(1, int(round(n * args.test_ratio)))
        n_train = n - n_test
        Xtr, ytr = X.iloc[:n_train], y.iloc[:n_train]
        Xte, yte = X.iloc[n_train:], y.iloc[n_train:]
        results.append(run_one(pref, Xtr, ytr, Xte, yte, args, out_dir))
    return results


# ---------------------------------------------------------------------------
# Aggregation & report
# ---------------------------------------------------------------------------

def aggregate_global(results, out_dir):
    """Sum per-fold feature scores and interaction matrices to a global view."""
    global_dir = os.path.join(out_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    global_scores = {}
    for r in results:
        for f, v in r["feat_scores"].items():
            global_scores[f] = global_scores.get(f, 0.0) + float(v)
    gs = pd.Series(global_scores).sort_values(ascending=False)
    plot_top_features(
        gs, os.path.join(global_dir, "top_features.png"),
        title="Global top features (summed across folds/categories)",
    )

    top30 = gs.head(30).index.tolist()
    M = np.zeros((30, 30), dtype=np.float64)
    idx = {f: i for i, f in enumerate(top30)}
    for r in results:
        for i_local, f_i in enumerate(r["interaction_top30"]):
            if f_i not in idx:
                continue
            for j_local, f_j in enumerate(r["interaction_top30"]):
                if f_j not in idx or i_local == j_local:
                    continue
                M[idx[f_i], idx[f_j]] += r["interaction_matrix"][i_local, j_local]
    plot_interaction_heatmap(
        M, top30, os.path.join(global_dir, "interaction_heatmap.png"),
        title="Global feature interactions -- top 30 (summed across folds)",
    )
    return gs


def write_report(args, results, global_scores, out_dir):
    path = os.path.join(out_dir, "report.md")
    lines = []
    lines.append(f"# RuleFit on c906-db -- split = `{args.split}`\n")
    lines.append("## Hyperparameters\n")
    lines.append(f"- top_k (features): {args.top_k}"
                 f"{' (ignored; feature selection bypassed)' if args.fs_method == 'none' else ''}")
    lines.append(f"- tree_size: {args.tree_size}")
    lines.append(f"- max_rules: {args.max_rules}")
    lines.append(f"- memory_par: {args.memory_par}")
    lines.append(f"- seed: {args.seed}")
    lines.append("- preprocessing: selected X columns and target y are z-scored "
                 "on the training split before RuleFit; predictions are "
                 "inverse-transformed before metrics/plots")
    lines.append("- rule units: `rules.csv` thresholds and coefficients are in "
                 "the standardized RuleFit model space")
    lines.append(f"- lasso_mode: {args.lasso_mode}"
                 f"{' (LassoCV positive=True; all coefficients >= 0)' if args.lasso_mode == 'nonneg' else ''}")
    lines.append(f"- fs_method: {args.fs_method}"
                 f"{' (feature selection bypassed)' if args.fs_method == 'none' else ''}")
    if args.fs_method == "univariate":
        lines.append(f"- fs_score_func: {args.fs_score_func}")
    if args.fs_method == "variance":
        lines.append(f"- fs_variance_threshold: {args.fs_variance_threshold}")
    if args.fs_method == "rfe":
        lines.append(f"- fs_rfe_step: {args.fs_rfe_step}")
    if args.fs_method == "from_model":
        lines.append(f"- fs_from_model_estimator: {args.fs_from_model_estimator}")
    if args.fs_method == "sequential":
        lines.append(f"- fs_sfs_tol: {args.fs_sfs_tol}")
    if args.fs_method in ("mcp", "deep"):
        lines.append(f"- fs_mcp_gamma: {args.fs_mcp_gamma}")
        lines.append(f"- fs_mcp_positive: {args.fs_mcp_positive}")
        lines.append(f"- fs_mcp_lambda_path_len: {args.fs_mcp_lambda_path_len}")
    if args.fs_method == "deep":
        lines.append(f"- fs_deep_v_i_multiplier: {args.fs_deep_v_i_multiplier}")
        lines.append(f"- fs_deep_max_swap_iters: {args.fs_deep_max_swap_iters}")
    if args.split == "time_ordered":
        lines.append(f"- test_ratio: {args.test_ratio}")
    lines.append("")
    lines.append("## Per-fold/category metrics\n")
    lines.append(
        "| label | n_train | n_test | feats | rules (nz/total) | "
        "train RMSE | train MAPE% | train R² | "
        "test RMSE | test MAPE% | test R² |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        mtr = r["metrics_train"]
        mte = r["metrics_test"]
        lines.append(
            f"| {r['label']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['n_features_kept']} | {r['n_rules_nonzero']}/{r['n_rules_total']} | "
            f"{mtr['rmse']:.5f} | {mtr['mape']:.2f} | {mtr['r2']:.4f} | "
            f"{mte['rmse']:.5f} | {mte['mape']:.2f} | {mte['r2']:.4f} |"
        )
    lines.append("")
    lines.append("## Top 20 features globally\n")
    lines.append("| rank | feature | summed importance |")
    lines.append("|---:|---|---:|")
    for i, (name, val) in enumerate(global_scores.head(20).items(), 1):
        lines.append(f"| {i} | `{name}` | {val:.4f} |")
    lines.append("")
    lines.append("## Files\n")
    lines.append("Each fold/category lives in its own subfolder; aggregated views in `global/`:\n")
    lines.append("- `<label>/rules.csv` -- full RuleFit rule table")
    lines.append("- `<label>/top_rules.png` -- top 20 rules by importance (signed)")
    lines.append("- `<label>/top_features.png` -- per-feature aggregated importance")
    lines.append("- `<label>/interaction_heatmap.png` -- top-30 feature co-occurrence")
    lines.append("- `<label>/pred_vs_true.png` -- scatter of test predictions")
    lines.append("- `global/top_features.png`, `global/interaction_heatmap.png` -- aggregated across folds/categories")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RuleFit on c906-db power data")
    parser.add_argument("--split", choices=["loco", "time_ordered"], required=True)
    parser.add_argument("--top_k", type=int, default=1000,
                        help="Number of features to keep by |corr| after "
                             "zero-var drop; ignored when --fs_method none")
    parser.add_argument("--tree_size", type=int, default=4)
    parser.add_argument("--max_rules", type=int, default=2000)
    parser.add_argument("--memory_par", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Used only when --split time_ordered")
    parser.add_argument("--lasso_mode", choices=["normal", "nonneg"], default="normal",
                        help="Lasso for the final RuleFit linear stage. "
                             "'normal' = standard LassoCV (signed coefs); "
                             "'nonneg' = LassoCV with positive=True (coefs >= 0).")
    fs_choices = ("none",) + tuple(FeatureSelector.METHODS)
    parser.add_argument(
        "--fs_method", choices=fs_choices, default="pearson",
        help="Feature-selection method run before RuleFit. Default 'pearson' "
             "matches the legacy filter_features behavior. All non-'none' "
             "methods standardize features in float64 first (fixing the "
             "float32 overflow on wide-bus signals). Use 'none' to bypass "
             "FeatureSelector and train on all input columns; top_k is ignored "
             "and only RuleFit float32-incompatible columns are dropped.",
    )
    parser.add_argument(
        "--fs_score_func", choices=["f_regression", "mutual_info_regression"],
        default="f_regression",
        help="Score function for --fs_method univariate. mutual_info_regression "
             "is much slower at large p.",
    )
    parser.add_argument(
        "--fs_variance_threshold", type=float, default=0.0,
        help="For --fs_method variance: drop columns with raw variance <= this.",
    )
    parser.add_argument(
        "--fs_rfe_step", type=float, default=0.1,
        help="For --fs_method rfe: fraction (or int) of features eliminated per round.",
    )
    parser.add_argument(
        "--fs_from_model_estimator", choices=["lasso", "rf"], default="lasso",
        help="Estimator backing --fs_method from_model.",
    )
    parser.add_argument(
        "--fs_from_model_max_iter", type=int, default=5000,
        help="LassoCV max_iter for --fs_method from_model with lasso.",
    )
    parser.add_argument(
        "--fs_sfs_tol", type=float, default=1e-4,
        help="Tolerance for --fs_method sequential. WARNING: sequential is "
             "impractical at --top_k=1000 (hours of compute). Pair with small top_k.",
    )
    parser.add_argument(
        "--fs_mcp_gamma", type=float, default=5.0,
        help="MCP penalty gamma (controls the |w|>gamma*lambda flat region).",
    )
    parser.add_argument(
        "--fs_mcp_positive", action="store_true",
        help="Constrain MCP weights >= 0 (independent of --lasso_mode).",
    )
    parser.add_argument(
        "--fs_mcp_max_iter", type=int, default=200,
        help="Max coordinate-descent sweeps per lambda value.",
    )
    parser.add_argument(
        "--fs_mcp_tol", type=float, default=1e-5,
        help="Convergence tolerance for MCP coordinate descent.",
    )
    parser.add_argument(
        "--fs_mcp_lambda_path_len", type=int, default=50,
        help="Number of lambdas in the warm-start regularization path.",
    )
    parser.add_argument(
        "--fs_deep_v_i_multiplier", type=int, default=3,
        help="For --fs_method deep: |V_I| target = multiplier * top_k.",
    )
    parser.add_argument(
        "--fs_deep_max_swap_iters", type=int, default=20,
        help="Max outer swap-refinement passes for --fs_method deep. Set to "
             "0 to skip step 2 (use OMP-only forward selection).",
    )
    parser.add_argument(
        "--fs_n_jobs", type=int, default=-1,
        help="Parallelism for sklearn estimators (LassoCV, RF, SFS) and "
             "joblib-parallel mutual_info_regression. -1 = all cores. "
             "MCP/DEEP coord descent is algorithmically sequential; numba "
             "is auto-used if installed.",
    )
    parser.add_argument(
        "--presim_subdir", "--presim", dest="presim_subdir",
        type=str, default="presim",
        help="Folder name under db/c906-db holding *_func.pkl presim files "
             "(e.g. presim, presim_large, presim_no_addr_data).",
    )
    args = parser.parse_args()
    try:
        args.presim_subdir = validate_presim_subdir(args.presim_subdir)
    except ValueError as exc:
        parser.error(str(exc))

    warnings.filterwarnings("ignore", category=FutureWarning)

    out_suffix = ""
    if args.fs_method != "pearson":
        out_suffix += f"_{args.fs_method}"
    if args.lasso_mode != "normal":
        out_suffix += f"_{args.lasso_mode}"
    if args.presim_subdir != "presim":
        out_suffix += f"_{args.presim_subdir}"
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "output",
        f"rulefit_c906_{args.split}{out_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {os.path.abspath(out_dir)}")

    if args.split == "loco":
        results = driver_loco(args, out_dir)
    else:
        results = driver_time_ordered(args, out_dir)

    global_scores = aggregate_global(results, out_dir)
    write_report(args, results, global_scores, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
