"""
XGBoost GBDT regression on the c906-db dataset.

Mirrors the shape of c906_ft_transformer.py but swaps the downstream model
for an XGBRegressor with early stopping on an internal validation split.
Feature selection is delegated to the same `FeatureSelector` class used by
the RuleFit / FT-Transformer pipelines.

Two split modes:

* loco          -- leave-one-category-out (5 folds over MMU/cache/csr/
                   exception/interrupt).
* time_ordered  -- per-category 80/20 split by ascending time_ps. One
                   model per category.

Outputs go to ../../output/gbdt_c906_<split>[_<fs_method>][_<presim_subdir>]/.

Usage (from src/algorithm-newalg/):
  python c906_gbdt.py --split time_ordered --fs_method none
  python c906_gbdt.py --split time_ordered --fs_method pearson
  python c906_gbdt.py --split time_ordered --fs_method mcp
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
import xgboost as xgb

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


def plot_training_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train (RMSE on z-y)", color="#1f77b4")
    ax.plot(history["val_loss"], label="val (RMSE on z-y)", color="#d62728")
    ax.axvline(history["best_epoch"], color="gray", linestyle="--", lw=1,
               label=f"best iter={history['best_epoch']}")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE (standardized y)")
    ax.set_title("Training and validation RMSE")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
                      xlabel="XGBoost feature importance (gain)"):
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
# Split helpers (copied from c906_ft_transformer.py)
# ---------------------------------------------------------------------------

def _split_tail(X, y, val_ratio):
    n_val = max(1, int(round(len(X) * val_ratio)))
    if n_val >= len(X):
        raise ValueError("val_ratio too large; would leave zero training rows.")
    return (
        X.iloc[:-n_val], y.iloc[:-n_val],
        X.iloc[-n_val:], y.iloc[-n_val:],
    )


def _split_per_category(X, y, category, val_ratio, seed):
    rng = np.random.RandomState(seed)
    val_idx = []
    cats = pd.Series(category).reset_index(drop=True)
    for cat in cats.unique():
        idx = np.flatnonzero(cats.values == cat)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.append(idx[:n_val])
    val_idx = np.concatenate(val_idx)
    mask = np.zeros(len(X), dtype=bool)
    mask[val_idx] = True
    return (
        X.iloc[~mask].reset_index(drop=True), y.iloc[~mask].reset_index(drop=True),
        X.iloc[mask].reset_index(drop=True),  y.iloc[mask].reset_index(drop=True),
    )


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

def run_one(label, X_train, y_train, X_test, y_test, args, out_dir,
            *, category_train=None, split_mode):
    t0 = time.time()
    print(f"\n=== {label} ===")
    print(f"  train rows: {len(X_train):,}   test rows: {len(X_test):,}")

    cols, fs_seconds = _select_columns(X_train, y_train, args)

    if split_mode == "time_ordered":
        X_tt, y_tt, X_tv, y_tv = _split_tail(X_train, y_train, args.gbdt_val_ratio)
    else:
        if category_train is None:
            raise ValueError("category_train is required for loco split mode.")
        X_tt, y_tt, X_tv, y_tv = _split_per_category(
            X_train, y_train, category_train, args.gbdt_val_ratio, args.seed,
        )
    print(f"  train-train: {len(X_tt):,}   train-val: {len(X_tv):,}")

    Xtt = X_tt[cols].to_numpy(dtype=np.float64, copy=False)
    Xtv = X_tv[cols].to_numpy(dtype=np.float64, copy=False)
    Xte = X_test[cols].to_numpy(dtype=np.float64, copy=False)
    Xtr_all = X_train[cols].to_numpy(dtype=np.float64, copy=False)
    ytt = y_tt.to_numpy(dtype=np.float64, copy=False)
    ytv = y_tv.to_numpy(dtype=np.float64, copy=False)
    yte = y_test.to_numpy(dtype=np.float64, copy=False)
    ytr_all = y_train.to_numpy(dtype=np.float64, copy=False)

    # XGBoost trees are less scale-sensitive than linear/NN models, but the
    # c906 presim matrix contains very wide bus-value features (~1e38).  Use
    # the same train-split z-score preprocessing as the other c906 pipelines
    # so the booster never sees huge raw magnitudes, and train on standardized
    # y so early-stopping RMSE is comparable across categories.
    std = Standardizer.fit(Xtt, ytt)
    Xtt_z = std.transform_X(Xtt)
    Xtv_z = std.transform_X(Xtv)
    Xte_z = std.transform_X(Xte)
    Xtr_all_z = std.transform_X(Xtr_all)
    ytt_z = std.transform_y(ytt)
    ytv_z = std.transform_y(ytv)
    print("  preprocessing: z-scored selected X and target y before XGBoost")

    print(f"  fitting XGBRegressor ("
          f"n_estimators={args.gbdt_n_estimators}, "
          f"max_depth={args.gbdt_max_depth}, "
          f"lr={args.gbdt_learning_rate}, "
          f"tree_method={args.gbdt_tree_method}) ...")
    t_train = time.time()
    model = xgb.XGBRegressor(
        n_estimators=args.gbdt_n_estimators,
        max_depth=args.gbdt_max_depth,
        learning_rate=args.gbdt_learning_rate,
        subsample=args.gbdt_subsample,
        colsample_bytree=args.gbdt_colsample_bytree,
        tree_method=args.gbdt_tree_method,
        n_jobs=args.gbdt_n_jobs,
        random_state=args.seed,
        early_stopping_rounds=args.gbdt_early_stopping_rounds,
        eval_metric="rmse",
    )
    model.fit(
        Xtt_z, ytt_z,
        eval_set=[(Xtt_z, ytt_z), (Xtv_z, ytv_z)],
        verbose=False,
    )
    train_seconds = time.time() - t_train
    best_iter = int(model.best_iteration) if hasattr(model, "best_iteration") else int(args.gbdt_n_estimators - 1)
    print(f"  best_iteration = {best_iter}")

    evals_result = model.evals_result()
    # XGBoost names the two eval sets validation_0 (train) and validation_1 (val).
    train_curve = evals_result.get("validation_0", {}).get("rmse", [])
    val_curve = evals_result.get("validation_1", {}).get("rmse", [])
    history = {
        "train_loss": list(train_curve),
        "val_loss": list(val_curve),
        "best_epoch": best_iter,
    }

    yp_tr = std.inverse_y(model.predict(Xtr_all_z))
    yp_te = std.inverse_y(model.predict(Xte_z))
    m_tr = compute_metrics(ytr_all, yp_tr)
    m_te = compute_metrics(yte, yp_te)
    print(f"  train: RMSE={m_tr['rmse']:.5f}  MAPE={m_tr['mape']:.2f}%  "
          f"R²={m_tr['r2']:.4f}")
    print(f"  test : RMSE={m_te['rmse']:.5f}  MAPE={m_te['mape']:.2f}%  "
          f"R²={m_te['r2']:.4f}")

    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    feat_scores = pd.Series(importance, index=cols).sort_values(ascending=False)

    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    if history["train_loss"] and history["val_loss"]:
        plot_training_curve(history, os.path.join(label_dir, "training_curve.png"))
    plot_pred_vs_true(yte, yp_te, os.path.join(label_dir, "pred_vs_true.png"),
                      title=f"Predicted vs True ({label}, test)")
    plot_top_features(
        feat_scores, os.path.join(label_dir, "feature_importance_top30.png"),
        title=f"Top features ({label}) -- XGBoost gain",
    )
    feat_scores.to_csv(os.path.join(label_dir, "feature_importance.csv"))
    pd.DataFrame({"y_true": yte, "y_pred": yp_te}).to_csv(
        os.path.join(label_dir, "test_predictions.csv"), index=False)
    with open(os.path.join(label_dir, "model.pkl"), "wb") as f:
        pickle.dump({
            "model": model,
            "selected_cols": list(cols),
            "preprocessing": "zscore_selected_X_and_y_before_xgboost",
            "standardizer": {
                "x_mean": std.x_mean, "x_std": std.x_std,
                "y_mean": std.y_mean, "y_std": std.y_std,
            },
            "best_iteration": best_iter,
            "config": {
                "n_estimators": args.gbdt_n_estimators,
                "max_depth": args.gbdt_max_depth,
                "learning_rate": args.gbdt_learning_rate,
                "subsample": args.gbdt_subsample,
                "colsample_bytree": args.gbdt_colsample_bytree,
                "tree_method": args.gbdt_tree_method,
                "early_stopping_rounds": args.gbdt_early_stopping_rounds,
            },
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
        "best_epoch": best_iter,
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
        cat_tr = pd.concat(
            [pd.Series([p] * len(pairs[p][0])) for p in train_prefixes],
            ignore_index=True,
        )
        Xte, yte = pairs[held]
        results.append(run_one(
            f"held_{held}", Xtr, ytr, Xte, yte, args, out_dir,
            category_train=cat_tr, split_mode="loco",
        ))
        del Xtr, ytr, cat_tr
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
            args, out_dir, split_mode="time_ordered",
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
        title="Global top features (summed XGBoost gain across folds)",
    )
    return gs


def write_report(args, results, global_scores, out_dir):
    path = os.path.join(out_dir, "report.md")
    lines = []
    lines.append(f"# XGBoost GBDT on c906-db -- split = `{args.split}`\n")
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
                 "on the internal train-train split before XGBoost; "
                 "predictions are inverse-transformed before metrics/plots")
    lines.append(f"- gbdt_n_estimators: {args.gbdt_n_estimators}")
    lines.append(f"- gbdt_max_depth: {args.gbdt_max_depth}")
    lines.append(f"- gbdt_learning_rate: {args.gbdt_learning_rate}")
    lines.append(f"- gbdt_subsample: {args.gbdt_subsample}")
    lines.append(f"- gbdt_colsample_bytree: {args.gbdt_colsample_bytree}")
    lines.append(f"- gbdt_early_stopping_rounds: {args.gbdt_early_stopping_rounds}")
    lines.append(f"- gbdt_val_ratio: {args.gbdt_val_ratio}")
    lines.append(f"- gbdt_tree_method: {args.gbdt_tree_method}")
    lines.append(f"- gbdt_n_jobs: {args.gbdt_n_jobs}")
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
        lines.append(
            f"| {r['label']} | {r['n_train']:,} | {r['n_test']:,} | "
            f"{r['n_features_kept']} | "
            f"{mtr['rmse']:.5f} | {mtr['mape']:.2f} | {mtr['r2']:.4f} | "
            f"{mte['rmse']:.5f} | {mte['mape']:.2f} | {mte['r2']:.4f} | "
            f"{r['best_epoch']} | {r['train_seconds']:.1f} |"
        )
    lines.append("")
    lines.append("## Top 20 features globally\n")
    lines.append("| rank | feature | summed gain |")
    lines.append("|---:|---|---:|")
    for i, (name, val) in enumerate(global_scores.head(20).items(), 1):
        lines.append(f"| {i} | `{name}` | {val:.5f} |")
    lines.append("")
    lines.append("## Files\n")
    lines.append("Each fold/category subfolder contains:\n")
    lines.append("- `training_curve.png` -- train/val RMSE per boosting round on standardized y")
    lines.append("- `pred_vs_true.png` -- scatter on the test split")
    lines.append("- `feature_importance_top30.png` -- top-30 features by gain")
    lines.append("- `feature_importance.csv`, `test_predictions.csv`")
    lines.append("- `model.pkl` -- XGBRegressor + selected cols + standardizer + best_iteration + config")
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
        description="XGBoost GBDT regression on c906-db power data")
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

    # GBDT-specific
    parser.add_argument("--gbdt_n_estimators", type=int, default=500)
    parser.add_argument("--gbdt_max_depth", type=int, default=6)
    parser.add_argument("--gbdt_learning_rate", type=float, default=0.05)
    parser.add_argument("--gbdt_subsample", type=float, default=0.8)
    parser.add_argument("--gbdt_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--gbdt_early_stopping_rounds", type=int, default=30)
    parser.add_argument("--gbdt_val_ratio", type=float, default=0.2,
                        help="Fraction of training rows held out for early stopping.")
    parser.add_argument("--gbdt_n_jobs", type=int, default=-1)
    parser.add_argument("--gbdt_tree_method", type=str, default="hist")

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
        f"gbdt_c906_{args.split}{out_suffix}",
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
