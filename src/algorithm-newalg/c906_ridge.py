"""
Ridge linear regression on the c906-db dataset.

Mirrors the shape of c906_ft_transformer.py but swaps the downstream model
from FT-Transformer to scikit-learn RidgeCV.  Feature selection is delegated
to the same `FeatureSelector` class used by the RuleFit / FT-Transformer
pipelines.

Two split modes:

* loco          -- leave-one-category-out (5 folds over MMU/cache/csr/
                   exception/interrupt).
* time_ordered  -- per-category 80/20 split by ascending time_ps. One
                   model per category.

Outputs go to ../../output/ridge_c906_<split>[_<fs_method>][_<presim_subdir>]/.

Usage (from src/algorithm-newalg/):
  python c906_ridge.py --split time_ordered --fs_method none
  python c906_ridge.py --split time_ordered --fs_method pearson
  python c906_ridge.py --split time_ordered --fs_method mcp
"""

import argparse
import os
import pickle
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from c906_rulefit_utils import (
    PREFIXES, load_c906_pair, compute_metrics, validate_presim_subdir,
)
from feature_selectors import FeatureSelector
from ft_transformer_model import Standardizer


# ---------------------------------------------------------------------------
# Plotting helpers (copied from c906_ft_transformer.py so this script stays
# independent).
# ---------------------------------------------------------------------------

def _shorten(name, max_len=70):
    if len(name) <= max_len:
        return name
    return name[:8] + "..." + name[-(max_len - 11):]


def plot_pred_vs_true(y_true, y_pred, out_path, title="Predicted vs True"):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True /Pc(openC906)")
    ax.set_ylabel("Predicted")
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"{title}\nR² = {r2:.4f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_top_features(feat_scores, out_path, top_n=30, title="Top features",
                      xlabel="|coef| (standardized X)"):
    df = feat_scores.head(top_n)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(df) + 1))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df.values, color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten(n, 80) for n in df.index], fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature-selection wrapper (copied from c906_ft_transformer.py)
# ---------------------------------------------------------------------------

def _select_columns(X_train, y_train, args):
    t_fs = time.time()
    if args.fs_method == "none":
        cols = list(X_train.columns)
        fs_seconds = time.time() - t_fs
        print(f"  feature selection bypassed: using all {len(cols)} features "
              f"(top_k ignored, {fs_seconds:.1f}s)")
        return cols, fs_seconds

    selector = FeatureSelector.from_args(args)
    cols = selector.fit_select(X_train, y_train)
    fs_seconds = time.time() - t_fs
    print(f"  features kept: {len(cols)} via fs_method={args.fs_method} "
          f"({fs_seconds:.1f}s)")
    return cols, fs_seconds


# ---------------------------------------------------------------------------
# Training one fold/category
# ---------------------------------------------------------------------------

def _parse_alphas(s):
    return [float(x) for x in s.split(",") if x.strip()]


def run_one(label, X_train, y_train, X_test, y_test, args, out_dir):
    t0 = time.time()
    print(f"\n=== {label} ===")
    print(f"  train rows: {len(X_train):,}   test rows: {len(X_test):,}")

    cols, fs_seconds = _select_columns(X_train, y_train, args)

    Xtr = X_train[cols].to_numpy(dtype=np.float64, copy=False)
    Xte = X_test[cols].to_numpy(dtype=np.float64, copy=False)
    ytr = y_train.to_numpy(dtype=np.float64, copy=False)
    yte = y_test.to_numpy(dtype=np.float64, copy=False)

    std = Standardizer.fit(Xtr, ytr)
    Xtr_z = std.transform_X(Xtr)
    Xte_z = std.transform_X(Xte)
    ytr_z = std.transform_y(ytr)

    alphas = _parse_alphas(args.ridge_alphas)
    print("  preprocessing: z-scored selected X and target y before RidgeCV")
    print(f"  fitting RidgeCV over alphas={alphas} on z-scored data ...")
    t_train = time.time()
    model = RidgeCV(
        alphas=alphas,
        fit_intercept=args.ridge_fit_intercept,
        scoring=None,
    )
    model.fit(Xtr_z, ytr_z)
    train_seconds = time.time() - t_train
    best_alpha = float(model.alpha_)
    print(f"  best alpha = {best_alpha}")

    yp_tr = std.inverse_y(model.predict(Xtr_z))
    yp_te = std.inverse_y(model.predict(Xte_z))
    m_tr = compute_metrics(ytr, yp_tr)
    m_te = compute_metrics(yte, yp_te)
    print(f"  train: RMSE={m_tr['rmse']:.5f}  MAPE={m_tr['mape']:.2f}%  "
          f"R²={m_tr['r2']:.4f}")
    print(f"  test : RMSE={m_te['rmse']:.5f}  MAPE={m_te['mape']:.2f}%  "
          f"R²={m_te['r2']:.4f}")

    # |coef| as feature importance. Coefficients are on z-scored X so they
    # are directly comparable across features in raw units.
    feat_scores = pd.Series(
        np.abs(model.coef_), index=cols
    ).sort_values(ascending=False)

    # Save artifacts.
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    plot_pred_vs_true(yte, yp_te, os.path.join(label_dir, "pred_vs_true.png"),
                      title=f"Predicted vs True ({label}, test)")
    plot_top_features(
        feat_scores, os.path.join(label_dir, "feature_importance_top30.png"),
        title=f"Top features ({label}) -- Ridge |coef|",
    )
    feat_scores.to_csv(os.path.join(label_dir, "feature_importance.csv"))
    pd.DataFrame({"y_true": yte, "y_pred": yp_te}).to_csv(
        os.path.join(label_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(label_dir, "model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "selected_cols": list(cols),
            "preprocessing": "zscore_selected_X_and_y_before_ridge",
            "standardizer": {
                "x_mean": std.x_mean, "x_std": std.x_std,
                "y_mean": std.y_mean, "y_std": std.y_std,
            },
            "best_alpha": best_alpha,
            "alphas_tried": alphas,
        }, f)

    print(f"  done in {time.time() - t0:.1f}s")
    return {
        "label": label,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_kept": len(cols),
        "metrics_train": m_tr,
        "metrics_test": m_te,
        "feat_scores": feat_scores,
        "best_alpha": best_alpha,
        "train_seconds": train_seconds,
        "fs_seconds": fs_seconds,
    }


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------

def driver_loco(args, out_dir):
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
        results.append(run_one(
            f"held_{held}", Xtr, ytr, Xte, yte, args, out_dir,
        ))
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
        n_test = max(1, int(round(len(X) * args.test_ratio)))
        n_train = len(X) - n_test
        results.append(run_one(
            pref, X.iloc[:n_train], y.iloc[:n_train],
            X.iloc[n_train:], y.iloc[n_train:],
            args, out_dir,
        ))
    return results


# ---------------------------------------------------------------------------
# Aggregation and report
# ---------------------------------------------------------------------------

def aggregate_global(results, out_dir):
    global_dir = os.path.join(out_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    summed = {}
    for r in results:
        for f, v in r["feat_scores"].items():
            summed[f] = summed.get(f, 0.0) + float(v)
    gs = pd.Series(summed).sort_values(ascending=False)
    gs.to_csv(os.path.join(global_dir, "feature_importance_summed.csv"))
    plot_top_features(
        gs, os.path.join(global_dir, "top_features.png"), top_n=30,
        title="Global top features (summed Ridge |coef| across folds)",
    )
    return gs


def write_report(args, results, global_scores, out_dir):
    path = os.path.join(out_dir, "report.md")
    lines = []
    lines.append(f"# Ridge on c906-db -- split = `{args.split}`\n")
    lines.append("## Hyperparameters\n")
    lines.append(f"- top_k (features): {args.top_k}"
                 f"{' (ignored; feature selection bypassed)' if args.fs_method == 'none' else ''}")
    lines.append(f"- presim_subdir: {args.presim_subdir}")
    lines.append(f"- seed: {args.seed}")
    if args.split == "time_ordered":
        lines.append(f"- test_ratio: {args.test_ratio}")
    lines.append(f"- fs_method: {args.fs_method}"
                 f"{' (feature selection bypassed)' if args.fs_method == 'none' else ''}")
    lines.append("- preprocessing: selected X columns and target y are z-scored "
                 "on the training split before RidgeCV; predictions are "
                 "inverse-transformed before metrics/plots")
    lines.append(f"- ridge_alphas: {args.ridge_alphas}")
    lines.append(f"- ridge_fit_intercept: {args.ridge_fit_intercept}")
    lines.append("")
    lines.append("## Per-fold/category metrics\n")
    lines.append(
        "| label | n_train | n_test | feats | "
        "train RMSE | train MAPE% | train R² | "
        "test RMSE | test MAPE% | test R² | best_epoch | train_s |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        mtr = r["metrics_train"]
        mte = r["metrics_test"]
        # best_epoch column is reused for the index of the chosen alpha in
        # the user-provided alphas list (RidgeCV has no epoch concept).
        # The sweep parser only requires this be an int.
        try:
            alphas = _parse_alphas(args.ridge_alphas)
            alpha_idx = alphas.index(r["best_alpha"])
        except ValueError:
            alpha_idx = 0
        lines.append(
            f"| {r['label']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['n_features_kept']} | "
            f"{mtr['rmse']:.5f} | {mtr['mape']:.2f} | {mtr['r2']:.4f} | "
            f"{mte['rmse']:.5f} | {mte['mape']:.2f} | {mte['r2']:.4f} | "
            f"{alpha_idx} | {r['train_seconds']:.1f} |"
        )
    lines.append("")
    lines.append("## Best alpha per fold/category\n")
    lines.append("| label | best_alpha |")
    lines.append("|---|---:|")
    for r in results:
        lines.append(f"| {r['label']} | {r['best_alpha']:g} |")
    lines.append("")
    lines.append("## Top 20 features globally\n")
    lines.append("| rank | feature | summed |coef| |")
    lines.append("|---:|---|---:|")
    for i, (name, val) in enumerate(global_scores.head(20).items(), 1):
        lines.append(f"| {i} | `{name}` | {val:.5f} |")
    lines.append("")
    lines.append("## Files\n")
    lines.append("Each fold/category subfolder contains:\n")
    lines.append("- `pred_vs_true.png` -- scatter on the test split")
    lines.append("- `feature_importance_top30.png` -- top-30 features by |coef|")
    lines.append("- `feature_importance.csv`, `test_predictions.csv`")
    lines.append("- `model.pkl` -- RidgeCV model + selected cols + standardizer + best alpha")
    lines.append("")
    lines.append("`global/` contains the same artifacts aggregated across all folds.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ridge regression on c906-db power data")
    parser.add_argument("--split", choices=["loco", "time_ordered"], required=True)
    parser.add_argument("--top_k", type=int, default=1000,
                        help="Number of features to keep; ignored when "
                             "--fs_method none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Used only when --split time_ordered")

    fs_choices = ("none",) + tuple(FeatureSelector.METHODS)
    parser.add_argument(
        "--fs_method", choices=fs_choices, default="pearson",
        help="Feature-selection method. Use 'none' to bypass FeatureSelector "
             "and train on all input columns (top_k is ignored).",
    )
    parser.add_argument("--fs_score_func",
                        choices=["f_regression", "mutual_info_regression"],
                        default="f_regression")
    parser.add_argument("--fs_variance_threshold", type=float, default=0.0)
    parser.add_argument("--fs_rfe_step", type=float, default=0.1)
    parser.add_argument("--fs_from_model_estimator", choices=["lasso", "rf"], default="lasso")
    parser.add_argument("--fs_from_model_max_iter", type=int, default=5000)
    parser.add_argument("--fs_sfs_tol", type=float, default=1e-4)
    parser.add_argument("--fs_mcp_gamma", type=float, default=5.0)
    parser.add_argument("--fs_mcp_positive", action="store_true")
    parser.add_argument("--fs_mcp_max_iter", type=int, default=200)
    parser.add_argument("--fs_mcp_tol", type=float, default=1e-5)
    parser.add_argument("--fs_mcp_lambda_path_len", type=int, default=50)
    parser.add_argument("--fs_deep_v_i_multiplier", type=int, default=3)
    parser.add_argument("--fs_deep_max_swap_iters", type=int, default=20)
    parser.add_argument("--fs_n_jobs", type=int, default=-1,
                        help="Parallelism for sklearn estimators (LassoCV, RF, "
                             "SFS) and joblib-parallel mutual_info_regression. "
                             "-1 = all cores.")
    parser.add_argument(
        "--presim_subdir", "--presim", dest="presim_subdir",
        type=str, default="presim",
        help="Folder name under db/c906-db holding *_func.pkl presim files "
             "(e.g. presim, presim_large, presim_no_addr_data).",
    )

    # Ridge-specific
    parser.add_argument("--ridge_alphas", type=str, default="0.01,0.1,1,10,100",
                        help="Comma-separated alpha grid for RidgeCV.")
    parser.add_argument("--ridge_fit_intercept", action="store_true", default=True)
    parser.add_argument("--no_ridge_fit_intercept", dest="ridge_fit_intercept",
                        action="store_false")

    args = parser.parse_args()
    try:
        args.presim_subdir = validate_presim_subdir(args.presim_subdir)
    except ValueError as exc:
        parser.error(str(exc))
    warnings.filterwarnings("ignore", category=FutureWarning)

    np.random.seed(args.seed)

    out_suffix = ""
    if args.fs_method != "pearson":
        out_suffix += f"_{args.fs_method}"
    if args.presim_subdir != "presim":
        out_suffix += f"_{args.presim_subdir}"
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "output",
        f"ridge_c906_{args.split}{out_suffix}",
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
