"""
3-layer MLP regression on the c906-db dataset.

Mirrors the shape of c906_ft_transformer.py but swaps the downstream model
for a small fully-connected MLP (input -> h1 -> h2 -> 1, with ReLU between
each linear layer = 3 weight matrices total).  Feature selection is
delegated to the same `FeatureSelector` class used by the RuleFit /
FT-Transformer pipelines.

Two split modes:

* loco          -- leave-one-category-out (5 folds over MMU/cache/csr/
                   exception/interrupt).
* time_ordered  -- per-category 80/20 split by ascending time_ps. One
                   model per category.

Outputs go to ../../output/mlp_c906_<split>[_<fs_method>][_<presim_subdir>]/.

Usage (from src/algorithm-newalg/):
  python c906_mlp.py --split time_ordered --fs_method none
  python c906_mlp.py --split time_ordered --fs_method pearson
  python c906_mlp.py --split time_ordered --fs_method mcp
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
import torch.nn as nn
import torch.nn.functional as F

from c906_rulefit_utils import (
    PREFIXES, load_c906_pair, compute_metrics, validate_presim_subdir,
)
from feature_selectors import FeatureSelector
from ft_transformer_model import Standardizer, default_device, describe_device


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


def plot_top_features(feat_scores, out_path, top_n=30, title="Top features",
                      xlabel="sum |W1| (first-layer input weights)"):
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
# Model
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """input -> h1 -> h2 -> 1 with ReLU between linear layers."""

    def __init__(self, n_features, hidden1=128, hidden2=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.drop(h)
        h = F.relu(self.fc2(h))
        h = self.drop(h)
        return self.fc3(h).squeeze(-1)


def _to_tensor(arr, dtype, device):
    return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)


def train_mlp(model, X_train, y_train, X_val, y_val, *, lr, weight_decay,
              max_epochs, patience, batch_size, device, seed, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    std = Standardizer.fit(X_train, y_train)
    Xtr_t = _to_tensor(std.transform_X(X_train), torch.float32, device)
    ytr_t = _to_tensor(std.transform_y(y_train), torch.float32, device)
    Xva_t = _to_tensor(std.transform_X(X_val), torch.float32, device)
    yva_t = _to_tensor(std.transform_y(y_val), torch.float32, device)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs))

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wait = 0
    n_train = Xtr_t.size(0)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_losses = []
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach()))
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(Xva_t)
            val_loss = float(F.mse_loss(pred_val, yva_t))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            wait = 0
        else:
            wait += 1

        if verbose and (epoch < 3 or epoch % 25 == 0 or improved):
            print(f"    epoch {epoch:3d}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}{' *' if improved else ''}")

        if wait >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} "
                      f"(best epoch {history['best_epoch']}, val={best_val:.5f})")
            break

    model.load_state_dict(best_state)
    return model, history, std


def predict_mlp(model, X, std, device, batch_size=512):
    model.eval()
    Xz_t = _to_tensor(std.transform_X(X), torch.float32, device)
    preds = []
    with torch.no_grad():
        for start in range(0, Xz_t.size(0), batch_size):
            preds.append(model(Xz_t[start:start + batch_size]).cpu().numpy())
    yz = np.concatenate(preds, axis=0) if preds else np.zeros((0,))
    return std.inverse_y(yz)


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
        X_tt, y_tt, X_tv, y_tv = _split_tail(X_train, y_train, args.mlp_val_ratio)
    else:
        if category_train is None:
            raise ValueError("category_train is required for loco split mode.")
        X_tt, y_tt, X_tv, y_tv = _split_per_category(
            X_train, y_train, category_train, args.mlp_val_ratio, args.seed,
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

    device = default_device(args.mlp_device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SmallMLP(
        n_features=len(cols),
        hidden1=args.mlp_hidden1,
        hidden2=args.mlp_hidden2,
        dropout=args.mlp_dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  fitting 3-layer MLP on {describe_device(device)}  "
          f"({n_params/1e3:.1f}k params, "
          f"h1={args.mlp_hidden1}, h2={args.mlp_hidden2}, "
          f"dropout={args.mlp_dropout}) ...")
    print("  preprocessing: z-scored selected X and target y before MLP")
    t_train = time.time()
    model, history, std = train_mlp(
        model, Xtt, ytt, Xtv, ytv,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        max_epochs=args.mlp_max_epochs,
        patience=args.mlp_patience,
        batch_size=args.mlp_batch_size,
        device=device,
        seed=args.seed,
        verbose=True,
    )
    train_seconds = time.time() - t_train

    yp_tr = predict_mlp(model, Xtr_all, std, device=device)
    yp_te = predict_mlp(model, Xte, std, device=device)
    m_tr = compute_metrics(ytr_all, yp_tr)
    m_te = compute_metrics(yte, yp_te)
    print(f"  train: RMSE={m_tr['rmse']:.5f}  MAPE={m_tr['mape']:.2f}%  "
          f"R²={m_tr['r2']:.4f}")
    print(f"  test : RMSE={m_te['rmse']:.5f}  MAPE={m_te['mape']:.2f}%  "
          f"R²={m_te['r2']:.4f}")

    # First-layer weight magnitude per input feature as importance proxy.
    W1 = model.fc1.weight.detach().cpu().numpy()  # (h1, n_features)
    importance = np.abs(W1).sum(axis=0)
    feat_scores = pd.Series(importance, index=cols).sort_values(ascending=False)

    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    plot_training_curve(history, os.path.join(label_dir, "training_curve.png"))
    plot_pred_vs_true(yte, yp_te, os.path.join(label_dir, "pred_vs_true.png"),
                      title=f"Predicted vs True ({label}, test)")
    plot_top_features(
        feat_scores, os.path.join(label_dir, "feature_importance_top30.png"),
        title=f"Top features ({label}) -- MLP first-layer |W|",
    )
    feat_scores.to_csv(os.path.join(label_dir, "feature_importance.csv"))
    pd.DataFrame({"y_true": yte, "y_pred": yp_te}).to_csv(
        os.path.join(label_dir, "test_predictions.csv"), index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "selected_cols": list(cols),
        "preprocessing": "zscore_selected_X_and_y_before_mlp",
        "x_mean": std.x_mean,
        "x_std": std.x_std,
        "y_mean": std.y_mean,
        "y_std": std.y_std,
        "config": {
            "n_features": len(cols),
            "hidden1": args.mlp_hidden1,
            "hidden2": args.mlp_hidden2,
            "dropout": args.mlp_dropout,
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
    summed = {}
    for r in results:
        for f, v in r["feat_scores"].items():
            summed[f] = summed.get(f, 0.0) + float(v)
    gs = pd.Series(summed).sort_values(ascending=False)
    gs.to_csv(os.path.join(global_dir, "feature_importance_summed.csv"))
    plot_top_features(
        gs, os.path.join(global_dir, "top_features.png"), top_n=30,
        title="Global top features (summed first-layer |W| across folds)",
    )
    return gs


def write_report(args, results, global_scores, out_dir, device):
    path = os.path.join(out_dir, "report.md")
    lines = []
    lines.append(f"# 3-layer MLP on c906-db -- split = `{args.split}`\n")
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
                 "on the internal train-train split before MLP; "
                 "predictions are inverse-transformed before metrics/plots")
    lines.append(f"- mlp_hidden1: {args.mlp_hidden1}")
    lines.append(f"- mlp_hidden2: {args.mlp_hidden2}")
    lines.append(f"- mlp_dropout: {args.mlp_dropout}")
    lines.append(f"- mlp_lr: {args.mlp_lr}")
    lines.append(f"- mlp_weight_decay: {args.mlp_weight_decay}")
    lines.append(f"- mlp_batch_size: {args.mlp_batch_size}")
    lines.append(f"- mlp_max_epochs: {args.mlp_max_epochs}")
    lines.append(f"- mlp_patience: {args.mlp_patience}")
    lines.append(f"- mlp_val_ratio: {args.mlp_val_ratio}")
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
    lines.append("| rank | feature | summed sum|W1| |")
    lines.append("|---:|---|---:|")
    for i, (name, val) in enumerate(global_scores.head(20).items(), 1):
        lines.append(f"| {i} | `{name}` | {val:.5f} |")
    lines.append("")
    lines.append("## Files\n")
    lines.append("Each fold/category subfolder contains:\n")
    lines.append("- `training_curve.png` -- train/val MSE on standardized y")
    lines.append("- `pred_vs_true.png` -- scatter on the test split")
    lines.append("- `feature_importance_top30.png` -- top-30 features by first-layer |W|")
    lines.append("- `feature_importance.csv`, `test_predictions.csv`")
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
        description="3-layer MLP regression on c906-db power data")
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

    # MLP-specific
    parser.add_argument("--mlp_hidden1", type=int, default=128)
    parser.add_argument("--mlp_hidden2", type=int, default=64)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_max_epochs", type=int, default=300)
    parser.add_argument("--mlp_patience", type=int, default=40)
    parser.add_argument("--mlp_val_ratio", type=float, default=0.2,
                        help="Fraction of training rows held out for early stopping.")
    parser.add_argument("--mlp_device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                        help="auto picks CUDA > MPS > CPU.")

    args = parser.parse_args()
    try:
        args.presim_subdir = validate_presim_subdir(args.presim_subdir)
    except ValueError as exc:
        parser.error(str(exc))
    warnings.filterwarnings("ignore", category=FutureWarning)

    out_suffix = ""
    if args.fs_method != "pearson":
        out_suffix += f"_{args.fs_method}"
    if args.presim_subdir != "presim":
        out_suffix += f"_{args.presim_subdir}"
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "output",
        f"mlp_c906_{args.split}{out_suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"output dir: {os.path.abspath(out_dir)}")

    device = default_device(args.mlp_device)
    print(f"device: {describe_device(device)}")
    if args.mlp_device == "auto" and device.type == "cpu":
        print("WARNING: --mlp_device auto resolved to CPU (no CUDA or MPS available). "
              "Training will be slower than on GPU.")

    if args.split == "loco":
        results = driver_loco(args, out_dir)
    else:
        results = driver_time_ordered(args, out_dir)

    global_scores = aggregate_global(results, out_dir)
    write_report(args, results, global_scores, out_dir, device=device)
    print("\nDone.")


if __name__ == "__main__":
    main()
