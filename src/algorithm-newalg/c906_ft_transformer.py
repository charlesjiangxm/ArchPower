"""
FT-Transformer regression on the c906-db dataset.

Mirrors the shape of c906_rulefit.py but swaps the downstream model from
RuleFit to FT-Transformer (Gorishniy et al., NeurIPS 2021).  Feature
selection is delegated to the same `FeatureSelector` class used by the
RuleFit pipeline.

Two split modes:

* loco          -- leave-one-category-out (5 folds over MMU/cache/csr/
                   exception/interrupt).
* time_ordered  -- per-category 80/20 split by ascending time_ps. One
                   model per category.

Outputs go to ../../output/ft_c906_<split>[_<fs_method>][_<presim_subdir>]/.

Usage (from src/algorithm-newalg/):
  python c906_ft_transformer.py --split time_ordered --fs_method none
  python c906_ft_transformer.py --split time_ordered --fs_method pearson
  python c906_ft_transformer.py --split time_ordered --fs_method mcp
"""

import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from c906_rulefit_utils import PREFIXES, load_c906_pair, compute_metrics
from feature_selectors import FeatureSelector
from ft_transformer_model import (
    FTTransformer, Standardizer, default_device, describe_device,
    train_ft_transformer, predict as ft_predict, extract_attention,
)


# ---------------------------------------------------------------------------
# Plotting helpers (copied/adapted from c906_rulefit.py so the two scripts
# stay independent).
# ---------------------------------------------------------------------------

def _shorten(name, max_len=70):
    if len(name) <= max_len:
        return name
    return name[:8] + "..." + name[-(max_len - 11):]


def plot_training_curve(history, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train (MSE on z-y)", color="#1f77b4")
    ax.plot(history["val_loss"], label="val (MSE on z-y)", color="#d62728")
    ax.axvline(history["best_epoch"], color="gray", linestyle="--", lw=1,
               label=f"best epoch={history['best_epoch']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardized y)")
    ax.set_title("Training and validation loss")
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


def plot_top_features(feat_scores, out_path, top_n=30, title="Top features"):
    df = feat_scores.head(top_n)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 0.4 * len(df) + 1))
    y_pos = np.arange(len(df))[::-1]
    ax.barh(y_pos, df.values, color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_shorten(n, 80) for n in df.index], fontsize=7)
    ax.set_xlabel("CLS attention (mean over heads, layers, test examples)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_interaction_heatmap(M, top_features, out_path, title="Feature interactions"):
    if M.size == 0 or M.max() == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No feature-feature attention to display.",
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
    fig.colorbar(im, ax=ax, label="attention weight (mean over heads/layers/examples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation split helpers
# ---------------------------------------------------------------------------

def _split_tail(X, y, val_ratio):
    """Take the last `val_ratio` rows as validation. Preserves time order."""
    n_val = max(1, int(round(len(X) * val_ratio)))
    if n_val >= len(X):
        raise ValueError("val_ratio too large; would leave zero training rows.")
    return (
        X.iloc[:-n_val], y.iloc[:-n_val],
        X.iloc[-n_val:], y.iloc[-n_val:],
    )


def _split_per_category(X, y, category, val_ratio, seed):
    """Per-prefix random validation split. Concats per-prefix slices."""
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


# ---------------------------------------------------------------------------
# Training one fold/category
# ---------------------------------------------------------------------------

def _select_columns(X_train, y_train, args):
    """Return the columns to train on and elapsed selector time.

    ``--fs_method none`` intentionally bypasses FeatureSelector and keeps the
    raw input columns, including constant columns.  ``top_k`` is ignored in
    this mode.
    """
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


def run_one(label, X_train, y_train, X_test, y_test, args, out_dir,
            *, category_train=None, split_mode):
    t0 = time.time()
    print(f"\n=== {label} ===")
    print(f"  train rows: {len(X_train):,}   test rows: {len(X_test):,}")

    # Feature selection, unless explicitly bypassed.
    cols, fs_seconds = _select_columns(X_train, y_train, args)

    # Internal val split (for early stopping)
    if split_mode == "time_ordered":
        X_tt, y_tt, X_tv, y_tv = _split_tail(X_train, y_train, args.ft_val_ratio)
    else:
        if category_train is None:
            raise ValueError("category_train is required for loco split mode.")
        X_tt, y_tt, X_tv, y_tv = _split_per_category(
            X_train, y_train, category_train, args.ft_val_ratio, args.seed,
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

    # Build + train
    device = default_device(args.ft_device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = FTTransformer(
        n_features=len(cols),
        d_token=args.ft_d_token,
        n_blocks=args.ft_n_blocks,
        n_heads=args.ft_n_heads,
        d_ffn=args.ft_d_ffn,
        dropout=args.ft_dropout,
        attn_dropout=args.ft_attn_dropout,
    )

    n_params = sum(p.numel() for p in model.parameters())
    amp_active = args.ft_amp and device.type == "cuda"
    print(f"  fitting FT-Transformer on {describe_device(device)}"
          f"{' [amp]' if amp_active else ''}  ({n_params/1e6:.2f}M params, "
          f"d_token={args.ft_d_token}, n_blocks={args.ft_n_blocks}, "
          f"n_heads={args.ft_n_heads}, d_ffn={args.ft_d_ffn}) ...")
    t_train = time.time()
    model, history, std = train_ft_transformer(
        model, Xtt, ytt, Xtv, ytv,
        lr=args.ft_lr,
        weight_decay=args.ft_weight_decay,
        max_epochs=args.ft_max_epochs,
        patience=args.ft_patience,
        batch_size=args.ft_batch_size,
        device=device,
        input_noise_std=args.ft_input_noise_std,
        grad_clip=args.ft_grad_clip,
        seed=args.seed,
        use_amp=args.ft_amp,
        verbose=True,
    )
    train_seconds = time.time() - t_train

    # Predictions on the FULL train set (not just train-train) and test set
    yp_tr = ft_predict(model, Xtr_all, std, device=device)
    yp_te = ft_predict(model, Xte, std, device=device)
    m_tr = compute_metrics(ytr_all, yp_tr)
    m_te = compute_metrics(yte, yp_te)
    print(f"  train: RMSE={m_tr['rmse']:.5f}  MAPE={m_tr['mape']:.2f}%  "
          f"R²={m_tr['r2']:.4f}")
    print(f"  test : RMSE={m_te['rmse']:.5f}  MAPE={m_te['mape']:.2f}%  "
          f"R²={m_te['r2']:.4f}")

    # Attention extraction on the test set
    cls_attn, ff_attn = extract_attention(model, Xte, std, device=device)
    feat_scores = pd.Series(cls_attn, index=cols).sort_values(ascending=False)
    top30 = feat_scores.head(30).index.tolist()
    top30_idx = [cols.index(f) for f in top30]
    ff_top30 = ff_attn[np.ix_(top30_idx, top30_idx)]
    # Zero the diagonal so the heatmap highlights cross-feature attention.
    np.fill_diagonal(ff_top30, 0.0)

    # Save artifacts
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    plot_training_curve(history, os.path.join(label_dir, "training_curve.png"))
    plot_pred_vs_true(yte, yp_te, os.path.join(label_dir, "pred_vs_true.png"),
                      title=f"Predicted vs True ({label}, test)")
    plot_top_features(
        feat_scores, os.path.join(label_dir, "cls_attention_top30.png"),
        title=f"Top features ({label}) -- CLS attention",
    )
    plot_interaction_heatmap(
        ff_top30, top30, os.path.join(label_dir, "interaction_heatmap.png"),
        title=f"Feature-feature attention top-30 ({label})",
    )
    feat_scores.to_csv(os.path.join(label_dir, "cls_attention.csv"))
    pd.DataFrame(ff_top30, index=top30, columns=top30).to_csv(
        os.path.join(label_dir, "interaction_matrix_top30.csv"))
    pd.DataFrame({"y_true": yte, "y_pred": yp_te}).to_csv(
        os.path.join(label_dir, "test_predictions.csv"), index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "selected_cols": list(cols),
        "x_mean": std.x_mean,
        "x_std": std.x_std,
        "y_mean": std.y_mean,
        "y_std": std.y_std,
        "config": {
            "n_features": len(cols),
            "d_token": args.ft_d_token,
            "n_blocks": args.ft_n_blocks,
            "n_heads": args.ft_n_heads,
            "d_ffn": args.ft_d_ffn,
            "dropout": args.ft_dropout,
            "attn_dropout": args.ft_attn_dropout,
        },
    }, os.path.join(label_dir, "model.pt"))

    print(f"  done in {time.time() - t0:.1f}s")
    return {
        "label": label,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features_kept": len(cols),
        "metrics_train": m_tr,
        "metrics_test": m_te,
        "feat_scores": feat_scores,
        "top30": top30,
        "ff_top30": ff_top30,
        "best_epoch": int(history["best_epoch"]),
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
    # Sum per-feature CLS attention across folds.
    summed = {}
    for r in results:
        for f, v in r["feat_scores"].items():
            summed[f] = summed.get(f, 0.0) + float(v)
    gs = pd.Series(summed).sort_values(ascending=False)
    gs.to_csv(os.path.join(global_dir, "cls_attention_summed.csv"))
    plot_top_features(
        gs, os.path.join(global_dir, "top_features.png"), top_n=30,
        title="Global top features (summed CLS attention across folds)",
    )

    # Sum feature-feature attention across folds, restricted to the
    # global top-30 features.
    top30 = gs.head(30).index.tolist()
    M = np.zeros((30, 30), dtype=np.float64)
    idx_map = {f: i for i, f in enumerate(top30)}
    for r in results:
        local_top30 = r["top30"]
        local_M = r["ff_top30"]
        for i_local, f_i in enumerate(local_top30):
            if f_i not in idx_map:
                continue
            for j_local, f_j in enumerate(local_top30):
                if f_j not in idx_map or i_local == j_local:
                    continue
                M[idx_map[f_i], idx_map[f_j]] += local_M[i_local, j_local]
    plot_interaction_heatmap(
        M, top30, os.path.join(global_dir, "interaction_heatmap.png"),
        title="Global feature-feature attention top-30 (summed across folds)",
    )
    pd.DataFrame(M, index=top30, columns=top30).to_csv(
        os.path.join(global_dir, "interaction_matrix_top30.csv"))
    return gs


def write_report(args, results, global_scores, out_dir, device):
    path = os.path.join(out_dir, "report.md")
    lines = []
    lines.append(f"# FT-Transformer on c906-db -- split = `{args.split}`\n")
    lines.append("## Hyperparameters\n")
    lines.append(f"- top_k (features): {args.top_k}"
                 f"{' (ignored; feature selection bypassed)' if args.fs_method == 'none' else ''}")
    lines.append(f"- presim_subdir: {args.presim_subdir}")
    lines.append(f"- seed: {args.seed}")
    if args.split == "time_ordered":
        lines.append(f"- test_ratio: {args.test_ratio}")
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
    lines.append(f"- ft_d_token: {args.ft_d_token}")
    lines.append(f"- ft_n_blocks: {args.ft_n_blocks}")
    lines.append(f"- ft_n_heads: {args.ft_n_heads}")
    lines.append(f"- ft_d_ffn: {args.ft_d_ffn}")
    lines.append(f"- ft_dropout: {args.ft_dropout}")
    lines.append(f"- ft_attn_dropout: {args.ft_attn_dropout}")
    lines.append(f"- ft_lr: {args.ft_lr}")
    lines.append(f"- ft_weight_decay: {args.ft_weight_decay}")
    lines.append(f"- ft_batch_size: {args.ft_batch_size}")
    lines.append(f"- ft_max_epochs: {args.ft_max_epochs}")
    lines.append(f"- ft_patience: {args.ft_patience}")
    lines.append(f"- ft_val_ratio: {args.ft_val_ratio}")
    lines.append(f"- ft_input_noise_std: {args.ft_input_noise_std}")
    lines.append(f"- ft_grad_clip: {args.ft_grad_clip}")
    lines.append(f"- ft_amp: {args.ft_amp}")
    lines.append(f"- device_used: {describe_device(device)}")
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
    lines.append("| rank | feature | summed CLS attention |")
    lines.append("|---:|---|---:|")
    for i, (name, val) in enumerate(global_scores.head(20).items(), 1):
        lines.append(f"| {i} | `{name}` | {val:.5f} |")
    lines.append("")
    lines.append("## Files\n")
    lines.append("Each fold/category subfolder contains:\n")
    lines.append("- `training_curve.png` -- train/val MSE on standardized y")
    lines.append("- `pred_vs_true.png` -- scatter on the test split")
    lines.append("- `cls_attention_top30.png` -- top-30 features by CLS attention")
    lines.append("- `interaction_heatmap.png` -- top-30 feature-feature attention")
    lines.append("- `cls_attention.csv`, `interaction_matrix_top30.csv`, `test_predictions.csv`")
    lines.append("- `model.pt` -- state_dict + selected cols + standardizer + config")
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
        description="FT-Transformer on c906-db power data")
    parser.add_argument("--split", choices=["loco", "time_ordered"], required=True)
    parser.add_argument("--top_k", type=int, default=1000,
                        help="Number of features to keep; ignored when "
                             "--fs_method none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Used only when --split time_ordered")

    # Feature-selection knobs (copy of c906_rulefit.py)
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
                             "-1 = all cores. MCP/DEEP coord descent is "
                             "sequential by algorithm; numba is auto-used if "
                             "installed.")
    parser.add_argument("--presim_subdir", type=str, default="presim")

    # FT-Transformer hyperparameters
    parser.add_argument("--ft_d_token", type=int, default=32)
    parser.add_argument("--ft_n_blocks", type=int, default=3)
    parser.add_argument("--ft_n_heads", type=int, default=4)
    parser.add_argument("--ft_d_ffn", type=int, default=64)
    parser.add_argument("--ft_dropout", type=float, default=0.1)
    parser.add_argument("--ft_attn_dropout", type=float, default=0.1)
    parser.add_argument("--ft_lr", type=float, default=1e-3)
    parser.add_argument("--ft_weight_decay", type=float, default=1e-4)
    parser.add_argument("--ft_batch_size", type=int, default=256)
    parser.add_argument("--ft_max_epochs", type=int, default=300)
    parser.add_argument("--ft_patience", type=int, default=40)
    parser.add_argument("--ft_val_ratio", type=float, default=0.2,
                        help="Fraction of training rows held out for early stopping.")
    parser.add_argument("--ft_input_noise_std", type=float, default=0.02)
    parser.add_argument("--ft_grad_clip", type=float, default=1.0)
    parser.add_argument("--ft_device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="auto picks CUDA > MPS > CPU. Explicit cuda/mps "
                             "raises if unavailable instead of silently falling back.")
    parser.add_argument("--ft_amp", action="store_true",
                        help="Mixed-precision (autocast + GradScaler) on CUDA. "
                             "No-op on MPS / CPU.")

    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    out_suffix = ""
    if args.fs_method != "pearson":
        out_suffix += f"_{args.fs_method}"
    if args.presim_subdir != "presim":
        out_suffix += f"_{args.presim_subdir}"
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "output",
        f"ft_c906_{args.split}{out_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {os.path.abspath(out_dir)}")

    # Resolve and announce the training device early so users can verify GPU
    # is in use before training starts.
    device = default_device(args.ft_device)
    print(f"device: {describe_device(device)}"
          f"{' [amp]' if (args.ft_amp and device.type == 'cuda') else ''}")
    if args.ft_device == "auto" and device.type == "cpu":
        print("WARNING: --ft_device auto resolved to CPU (no CUDA or MPS available). "
              "Training will be slow. Install a GPU-enabled PyTorch build to "
              "use --ft_device cuda or mps.")

    if args.split == "loco":
        results = driver_loco(args, out_dir)
    else:
        results = driver_time_ordered(args, out_dir)

    global_scores = aggregate_global(results, out_dir)
    write_report(args, results, global_scores, out_dir, device=device)
    print("\nDone.")


if __name__ == "__main__":
    main()
