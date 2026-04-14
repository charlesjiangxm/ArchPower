"""
Ridge Regression baseline on the ArchPower Event Dataset.

Trains a single multi-output Ridge model (81 event features -> 12 power
components) and writes a markdown report with MAPE, RMSE, R2.

Usage:
    cd src
    python event_linear.py                    # defaults: BOOM, alpha=1.0
    python event_linear.py --arch XS          # XiangShan architecture
    python event_linear.py --alpha 0.1        # lower regularization
"""

import argparse
import os

import numpy as np
from sklearn.linear_model import Ridge

from event_utils import (
    COMP_NAMES, prepare_data_numpy, denormalize,
    compute_metrics, generate_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Ridge Regression on ArchPower Event Dataset")
    parser.add_argument("--arch", default="BOOM", choices=["BOOM", "XS"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization strength")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output",
                           f"linear_{args.arch}")
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
    print(f"[2/3] Training Ridge regression (alpha={args.alpha}) ...")
    model = Ridge(alpha=args.alpha)
    model.fit(data["X_train"], data["y_train"])

    n_params = model.coef_.size + model.intercept_.size
    print(f"  parameters: {n_params:,}")

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
        model_name="Ridge Regression (Linear)",
        model_description="sklearn Ridge (multi-output)",
        n_params=n_params,
        n_train=n_train, n_val=n_val,
        n_configs=n_configs, n_val_configs=n_val_configs,
        val_ratio=args.val_ratio,
        metrics=metrics,
        out_path=os.path.join(out_dir, "report.md"),
        extra_model_info=[
            ("alpha", args.alpha),
        ],
    )

    print(f"\nDone!  All outputs in: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
