"""
XGBoost GBDT model on the ArchPower Event Dataset.

Trains 12 independent XGBRegressor models (one per power component) on the
81 event features and writes a markdown report with MAPE, RMSE, R2.

Usage:
    cd src
    python event_gbdt.py                      # defaults: BOOM
    python event_gbdt.py --arch XS            # XiangShan architecture
    python event_gbdt.py --n_estimators 200   # more trees
"""

import argparse
import os

import numpy as np
import xgboost as xgb

from event_utils import (
    COMP_NAMES, prepare_data_numpy, denormalize,
    compute_metrics, generate_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost GBDT on ArchPower Event Dataset")
    parser.add_argument("--arch", default="BOOM", choices=["BOOM", "XS"])
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output",
                           f"gbdt_{args.arch}")
    os.makedirs(out_dir, exist_ok=True)

    # --- data ---
    print(f"[1/3] Loading {args.arch} event dataset ...")
    data = prepare_data_numpy(args.arch, val_ratio=args.val_ratio,
                              seed=args.seed)
    n_train = data["X_train"].shape[0]
    n_val = data["X_val"].shape[0]
    n_targets = data["y_train"].shape[1]
    print(f"  train: {n_train} samples  val: {n_val} samples  "
          f"features: {data['X_train'].shape[1]}  targets: {n_targets}")

    # --- train ---
    print(f"[2/3] Training {n_targets} XGBRegressor models "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}) ...")
    models = []
    for i in range(n_targets):
        model = xgb.XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.seed,
            verbosity=0,
        )
        model.fit(data["X_train"], data["y_train"][:, i])
        models.append(model)
        print(f"  [{i+1}/{n_targets}] {COMP_NAMES[i]} trained")

    # --- evaluate ---
    print("[3/3] Evaluating ...")
    pred_norm = np.column_stack([m.predict(data["X_val"]) for m in models])
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
        model_name="XGBoost GBDT",
        model_description="12 independent XGBRegressor models (one per component)",
        n_params="N/A (tree ensemble)",
        n_train=n_train, n_val=n_val,
        n_configs=n_configs, n_val_configs=n_val_configs,
        val_ratio=args.val_ratio,
        metrics=metrics,
        out_path=os.path.join(out_dir, "report.md"),
        extra_model_info=[
            ("n_estimators", args.n_estimators),
            ("max_depth", args.max_depth),
            ("learning_rate", args.learning_rate),
        ],
    )

    print(f"\nDone!  All outputs in: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
