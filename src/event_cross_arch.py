"""
Cross-Architecture Transfer on the ArchPower Event Dataset.

Exposes reusable functions for training either a GBDT (XGBoost) or Linear
(Ridge) model on ALL samples of one architecture (BOOM or XS) and evaluating
on ALL samples of the other. The companion notebook
`script/event_cross_arch.ipynb` drives these functions as a UI; running this
file directly executes the CLI workflow (train + evaluate + markdown report).

The event-only dataset is architecture-normalized by resource functions at
construction time, so both arches share the same 81 event features -- making
a cross-arch generalization experiment meaningful.

Normalization uses source-arch (training) statistics only, then applies them
to the target arch. This treats the target as unseen at training time.

Usage:
    cd src
    # BOOM -> XS with XGBoost
    python event_cross_arch.py --model gbdt --train_arch BOOM --test_arch XS

    # XS -> BOOM with Ridge
    python event_cross_arch.py --model linear --train_arch XS --test_arch BOOM
"""

import argparse
import os

import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge

from event_utils import (
    COMP_NAMES, load_event_dataset, denormalize,
    compute_metrics, generate_report,
)

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_ALPHA = 1.0
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_data(train_arch, test_arch):
    """Load both arches and normalize using source-arch statistics."""
    feat_src, label_src = load_event_dataset(train_arch)
    feat_tgt, label_tgt = load_event_dataset(test_arch)

    feat_mean = feat_src.mean(axis=0, keepdims=True)
    feat_std = feat_src.std(axis=0, keepdims=True) + 1e-8
    label_mean = label_src.mean(axis=0, keepdims=True)
    label_std = label_src.std(axis=0, keepdims=True) + 1e-8

    X_train = (feat_src - feat_mean) / feat_std
    X_test = (feat_tgt - feat_mean) / feat_std
    y_train = (label_src - label_mean) / label_std

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "X_test_raw": feat_tgt,
        "y_train_raw": label_src,
        "y_test_raw": label_tgt,
        "label_mean": label_mean,
        "label_std": label_std,
    }


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------

def train_gbdt(data, n_estimators=DEFAULT_N_ESTIMATORS,
               max_depth=DEFAULT_MAX_DEPTH,
               learning_rate=DEFAULT_LEARNING_RATE,
               seed=DEFAULT_SEED, feature_names=None, verbose=True):
    """Fit one XGBRegressor per power component on the source arch."""
    n_targets = data["y_train"].shape[1]
    models = []
    for i in range(n_targets):
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            verbosity=0,
        )
        model.fit(data["X_train"], data["y_train"][:, i])
        if feature_names is not None:
            model.get_booster().feature_names = list(feature_names)
        models.append(model)
        if verbose:
            print(f"  [{i+1:2d}/{n_targets}] {COMP_NAMES[i]:12s} trained")
    return models


def predict_gbdt(models, data):
    """Return target-arch predictions in original (denormalized) scale."""
    pred_norm = np.column_stack([m.predict(data["X_test"]) for m in models])
    return denormalize(pred_norm, data["label_mean"], data["label_std"])


def train_linear(data, alpha=DEFAULT_ALPHA):
    """Fit a multi-output Ridge regressor on the source arch."""
    model = Ridge(alpha=alpha)
    model.fit(data["X_train"], data["y_train"])
    return model


def predict_linear(model, data):
    """Return target-arch predictions in original (denormalized) scale."""
    pred_norm = model.predict(data["X_test"])
    return denormalize(pred_norm, data["label_mean"], data["label_std"])


# ---------------------------------------------------------------------------
# Plot: predicted vs true scatter
# ---------------------------------------------------------------------------

def plot_pred_vs_true(pred, true, metrics, train_arch, test_arch,
                      out_dir, filename="pred_vs_true.png"):
    """4x3 grid of predicted-vs-true scatter plots on the target arch."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    for i, name in enumerate(COMP_NAMES):
        ax = axes[i // 3, i % 3]
        yt, yp = true[:, i], pred[:, i]
        ax.scatter(yp, yt, c="C1", s=30, alpha=0.8,
                   label=f"{test_arch} ({len(yt)})")
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}  (R^2 = {metrics['r2'][i]:.3f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{train_arch} \u2192 {test_arch}", fontsize=14, y=1.002)
    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(train_arch, test_arch, model_kind, data, metrics,
                 hyperparams, n_params, out_path):
    """Thin wrapper over event_utils.generate_report with cross-arch fields."""
    n_train = data["X_train"].shape[0]
    n_test = data["X_test"].shape[0]
    n_val_configs = n_test // 8
    n_configs = n_val_configs  # all target configs are held out
    if model_kind == "gbdt":
        model_name = f"XGBoost GBDT ({train_arch} \u2192 {test_arch})"
        model_description = (
            "12 independent XGBRegressor models, trained on source arch")
    else:
        model_name = f"Ridge Linear ({train_arch} \u2192 {test_arch})"
        model_description = (
            "sklearn Ridge (multi-output), trained on source arch")
    extra = [("train_arch", train_arch), ("test_arch", test_arch)]
    extra.extend(list(hyperparams.items()))
    generate_report(
        arch=f"{train_arch} \u2192 {test_arch}",
        model_name=model_name,
        model_description=model_description,
        n_params=n_params,
        n_train=n_train,
        n_val=n_test,
        n_configs=n_configs,
        n_val_configs=n_val_configs,
        val_ratio=1.0,
        metrics=metrics,
        out_path=out_path,
        extra_model_info=extra,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-arch transfer on ArchPower Event Dataset")
    parser.add_argument("--model", default="gbdt", choices=["gbdt", "linear"],
                        help="Model family: gbdt (XGBoost) or linear (Ridge)")
    parser.add_argument("--train_arch", default="BOOM", choices=["BOOM", "XS"])
    parser.add_argument("--test_arch", default="XS", choices=["BOOM", "XS"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_estimators", type=int,
                        default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--learning_rate", type=float,
                        default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Ridge regularization strength")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.train_arch == args.test_arch:
        print(f"[warn] train_arch == test_arch ({args.train_arch}); "
              f"running as same-arch sanity check.")

    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "output",
        f"cross_{args.model}_{args.train_arch}_to_{args.test_arch}",
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/3] Loading source={args.train_arch}, "
          f"target={args.test_arch} ...")
    data = build_data(args.train_arch, args.test_arch)
    n_train = data["X_train"].shape[0]
    n_test = data["X_test"].shape[0]
    n_targets = data["y_train"].shape[1]
    print(f"  train: {n_train} samples  test: {n_test} samples  "
          f"features: {data['X_train'].shape[1]}  targets: {n_targets}")

    if args.model == "gbdt":
        print(f"[2/3] Training {n_targets} XGBRegressor models "
              f"(n_estimators={args.n_estimators}, "
              f"max_depth={args.max_depth}) ...")
        models = train_gbdt(
            data,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        pred = predict_gbdt(models, data)
        n_params = "N/A (tree ensemble)"
        hyperparams = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        }
    else:
        print(f"[2/3] Training Ridge regression (alpha={args.alpha}) ...")
        model = train_linear(data, alpha=args.alpha)
        pred = predict_linear(model, data)
        n_params = int(model.coef_.size + model.intercept_.size)
        print(f"  parameters: {n_params:,}")
        hyperparams = {"alpha": args.alpha}

    print("[3/3] Evaluating on target arch ...")
    true = data["y_test_raw"]
    metrics = compute_metrics(pred, true)

    print(f"\n  {'Component':12s} {'RMSE':>10s} {'MAPE%':>8s} {'R2':>8s}")
    for i, name in enumerate(COMP_NAMES):
        print(f"  {name:12s} {metrics['rmse'][i]:10.6f} "
              f"{metrics['mape'][i]:8.2f}% {metrics['r2'][i]:8.4f}")

    write_report(
        train_arch=args.train_arch,
        test_arch=args.test_arch,
        model_kind=args.model,
        data=data,
        metrics=metrics,
        hyperparams=hyperparams,
        n_params=n_params,
        out_path=os.path.join(out_dir, "report.md"),
    )

    print(f"\nDone!  All outputs in: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
