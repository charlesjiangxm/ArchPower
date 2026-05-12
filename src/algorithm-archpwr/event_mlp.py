"""
3-Hidden-Layer MLP on the ArchPower Event Dataset.

Trains a multi-output MLPRegressor (81 event features -> 12 power components)
with 3 hidden layers and writes a markdown report with MAPE, RMSE, R2.

Usage:
    cd src
    python event_mlp.py                                  # defaults: BOOM, (128,64,32)
    python event_mlp.py --arch XS                        # XiangShan architecture
    python event_mlp.py --hidden_layers 256,128,64       # larger network
"""

import argparse
import os

import numpy as np
from sklearn.neural_network import MLPRegressor

from event_utils import (
    COMP_NAMES, prepare_data_numpy, denormalize,
    compute_metrics, generate_report,
)


def count_mlp_params(model):
    """Count total trainable parameters from fitted MLPRegressor."""
    n = sum(c.size for c in model.coefs_)
    n += sum(b.size for b in model.intercepts_)
    return n


def main():
    parser = argparse.ArgumentParser(
        description="3-Layer MLP on ArchPower Event Dataset")
    parser.add_argument("--arch", default="BOOM", choices=["BOOM", "XS"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_layers", type=str, default="128,64,32",
                        help="Comma-separated hidden layer sizes (default: 128,64,32)")
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.001,
                        help="L2 regularization strength")
    args = parser.parse_args()

    hidden_sizes = tuple(int(x) for x in args.hidden_layers.split(","))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output",
                           f"mlp_{args.arch}")
    os.makedirs(out_dir, exist_ok=True)

    # --- data ---
    print(f"[1/3] Loading {args.arch} event dataset ...")
    data = prepare_data_numpy(args.arch, val_ratio=args.val_ratio,
                              seed=args.seed)
    n_train = data["X_train"].shape[0]
    n_val = data["X_val"].shape[0]
    print(f"  train: {n_train} samples  val: {n_val} samples  "
          f"features: {data['X_train'].shape[1]}")

    # --- train ---
    print(f"[2/3] Training MLP {hidden_sizes} (max_iter={args.max_iter}) ...")
    model = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        alpha=args.alpha,
        max_iter=args.max_iter,
        early_stopping=False,
        random_state=args.seed,
    )
    model.fit(data["X_train"], data["y_train"])

    n_params = count_mlp_params(model)
    print(f"  parameters: {n_params:,}")
    print(f"  converged in {model.n_iter_} iterations")

    # --- evaluate ---
    print("[3/3] Evaluating ...")
    pred_norm = model.predict(data["X_val"])
    pred_orig = denormalize(pred_norm, data["label_mean"], data["label_std"])
    true_orig = data["y_val_raw"]
    metrics = compute_metrics(pred_orig, true_orig)

    print(f"\n  {'Component':12s} {'RMSE':>10s} {'MAPE%':>8s} {'R2':>8s}")
    for i, name in enumerate(COMP_NAMES):
        print(f"  {name:12s} {metrics['rmse'][i]:10.6f} "
              f"{metrics['mape'][i]:8.2f}% {metrics['r2'][i]:8.4f}")

    # --- report ---
    n_val_configs = n_val // 8
    n_configs = n_train // 8 + n_val_configs
    generate_report(
        arch=args.arch,
        model_name="3-Layer MLP",
        model_description=f"sklearn MLPRegressor {hidden_sizes}",
        n_params=n_params,
        n_train=n_train, n_val=n_val,
        n_configs=n_configs, n_val_configs=n_val_configs,
        val_ratio=args.val_ratio,
        metrics=metrics,
        out_path=os.path.join(out_dir, "report.md"),
        extra_model_info=[
            ("hidden_layers", hidden_sizes),
            ("activation", "relu"),
            ("solver", "adam"),
            ("alpha (L2)", args.alpha),
            ("max_iter", args.max_iter),
            ("actual_iter", model.n_iter_),
        ],
    )

    print(f"\nDone!  All outputs in: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
