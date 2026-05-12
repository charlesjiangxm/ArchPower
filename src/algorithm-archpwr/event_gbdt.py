"""
XGBoost GBDT library for the ArchPower Event Dataset.

Exposes reusable functions for training, evaluation, importance analysis,
tree visualization, SHAP attribution, and report generation. The companion
notebook `script/train_archpwr.ipynb` drives these functions as a UI; running
this file directly (`python event_gbdt.py --arch BOOM`) executes the CLI
workflow (train + evaluate + markdown report).

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

DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.3
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42

IMP_TYPES = ("gain", "cover", "weight")


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------

def train_models(data, n_estimators=DEFAULT_N_ESTIMATORS,
                 max_depth=DEFAULT_MAX_DEPTH,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 seed=DEFAULT_SEED, feature_names=None, verbose=True):
    """Fit one XGBRegressor per power component."""
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


def predict_both(models, data):
    """Return (train_pred, val_pred) in original (denormalized) scale."""
    train_norm = np.column_stack([m.predict(data["X_train"]) for m in models])
    val_norm = np.column_stack([m.predict(data["X_val"]) for m in models])
    train_pred = denormalize(train_norm, data["label_mean"], data["label_std"])
    val_pred = denormalize(val_norm, data["label_mean"], data["label_std"])
    return train_pred, val_pred


# ---------------------------------------------------------------------------
# Plot: predicted vs true scatter
# ---------------------------------------------------------------------------

def plot_pred_vs_true(train_true, train_pred, val_true, val_pred,
                      val_metrics, out_dir, filename="pred_vs_true.png"):
    """4x3 grid of predicted-vs-true scatter plots, one per component."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    for i, name in enumerate(COMP_NAMES):
        ax = axes[i // 3, i % 3]
        yt_tr, yp_tr = train_true[:, i], train_pred[:, i]
        yt_va, yp_va = val_true[:, i], val_pred[:, i]
        ax.scatter(yp_tr, yt_tr, c="C0", s=20, alpha=0.5,
                   label=f"train ({len(yt_tr)})")
        ax.scatter(yp_va, yt_va, c="C1", s=35, alpha=0.85,
                   label=f"test ({len(yt_va)})")
        lo = float(min(yt_tr.min(), yt_va.min(), yp_tr.min(), yp_va.min()))
        hi = float(max(yt_tr.max(), yt_va.max(), yp_tr.max(), yp_va.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name}  (val R^2 = {val_metrics['r2'][i]:.3f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Feature importance (gain / cover / weight)
# ---------------------------------------------------------------------------

def compute_feature_importance(models, feature_names):
    """Return (n_components, n_features, len(IMP_TYPES)) importance array."""
    feat_list = list(feature_names)

    def _one(booster, imp_type):
        score = booster.get_score(importance_type=imp_type)
        arr = np.zeros(len(feat_list))
        for k, v in score.items():
            if k in feat_list:
                arr[feat_list.index(k)] = v
        return arr

    return np.stack(
        [np.stack([_one(m.get_booster(), it) for it in IMP_TYPES], axis=1)
         for m in models],
        axis=0,
    )


def plot_per_component_importance(imp, feature_names, out_dir, top_k=15):
    """Write one importance_<comp>.png per component (disk only)."""
    import matplotlib.pyplot as plt

    feat_list = list(feature_names)
    for ci, cname in enumerate(COMP_NAMES):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ti, it in enumerate(IMP_TYPES):
            ax = axes[ti]
            v = imp[ci, :, ti]
            order = np.argsort(v)[::-1][:top_k]
            names = [feat_list[j] for j in order]
            ax.barh(range(top_k), v[order][::-1], color=f"C{ti}")
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(names[::-1], fontsize=8)
            ax.set_title(f"{cname} -- {it}")
            ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"importance_{cname}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {len(COMP_NAMES)} per-component importance figures to {out_dir}/")


def plot_global_importance(imp, feature_names, out_dir, top_k=20,
                           filename="importance_global.png"):
    """Plot mean-across-components importance. Returns figure for inline display."""
    import matplotlib.pyplot as plt

    feat_list = list(feature_names)
    imp_global = imp.mean(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ti, it in enumerate(IMP_TYPES):
        ax = axes[ti]
        v = imp_global[:, ti]
        order = np.argsort(v)[::-1][:top_k]
        names = [feat_list[j] for j in order]
        ax.barh(range(top_k), v[order][::-1], color=f"C{ti}")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(names[::-1], fontsize=9)
        ax.set_title(f"Global mean -- {it}")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Tree visualization
# ---------------------------------------------------------------------------

def plot_tree(model, comp_name, tree_index, out_dir):
    """Render a single tree via graphviz; fall back to text dump on failure."""
    import matplotlib.pyplot as plt

    booster = model.get_booster()
    try:
        fig, ax = plt.subplots(figsize=(32, 14))
        xgb.plot_tree(booster, num_trees=tree_index, rankdir="LR", ax=ax)
        ax.set_title(f"{comp_name} -- tree {tree_index}", fontsize=16)
        out_path = os.path.join(out_dir, f"tree_{comp_name}_t{tree_index}.png")
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"saved: {out_path}")
        return fig
    except Exception as e:
        print(f"graphviz rendering failed ({type(e).__name__}: {e})")
        print("falling back to text dump\n")
        text_dump = booster.get_dump(dump_format="text")[tree_index]
        print(text_dump)
        out_path = os.path.join(out_dir, f"tree_{comp_name}_t{tree_index}.txt")
        with open(out_path, "w") as f:
            f.write(text_dump)
        print(f"saved text tree to: {out_path}")
        return None


# ---------------------------------------------------------------------------
# SHAP attribution
# ---------------------------------------------------------------------------

def compute_shap_values(models, X_val):
    """Return list of SHAP value arrays (one per component model)."""
    import shap
    return [shap.TreeExplainer(m).shap_values(X_val) for m in models]


def plot_shap_per_component(shap_values, X_val_raw, feature_names, out_dir):
    """Write shap_beeswarm_<comp>.png and shap_bar_<comp>.png per component."""
    import matplotlib.pyplot as plt
    import shap

    for ci, cname in enumerate(COMP_NAMES):
        sv = shap_values[ci]

        plt.figure()
        shap.summary_plot(sv, X_val_raw, feature_names=list(feature_names),
                          max_display=20, show=False)
        plt.gcf().suptitle(f"SHAP beeswarm -- {cname}", y=1.02)
        plt.savefig(os.path.join(out_dir, f"shap_beeswarm_{cname}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close("all")

        plt.figure()
        shap.summary_plot(sv, X_val_raw, feature_names=list(feature_names),
                          plot_type="bar", max_display=20, show=False)
        plt.gcf().suptitle(f"SHAP mean(|SHAP|) -- {cname}", y=1.02)
        plt.savefig(os.path.join(out_dir, f"shap_bar_{cname}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close("all")
    print(f"saved {2 * len(COMP_NAMES)} per-component SHAP figures")


def plot_global_shap(shap_values, feature_names, out_dir, top_k=20,
                     filename="shap_global.png"):
    """Plot mean |SHAP| across all component models. Returns figure."""
    import matplotlib.pyplot as plt

    feat_list = list(feature_names)
    sv_global = np.stack([np.abs(s).mean(axis=0) for s in shap_values],
                         axis=0).mean(axis=0)
    order = np.argsort(sv_global)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_k), sv_global[order][::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feat_list[j] for j in order[::-1]], fontsize=10)
    ax.set_xlabel(f"Mean |SHAP| across {len(shap_values)} components")
    ax.set_title(f"Global SHAP attribution (top {top_k} features)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(arch, data, metrics, val_ratio, hyperparams, out_path):
    """Thin wrapper over event_utils.generate_report with GBDT-specific fields."""
    n_train = data["X_train"].shape[0]
    n_val = data["X_val"].shape[0]
    n_val_configs = n_val // 8
    n_configs = n_train // 8 + n_val_configs
    generate_report(
        arch=arch,
        model_name="XGBoost GBDT",
        model_description="12 independent XGBRegressor models (one per component)",
        n_params="N/A (tree ensemble)",
        n_train=n_train, n_val=n_val,
        n_configs=n_configs, n_val_configs=n_val_configs,
        val_ratio=val_ratio,
        metrics=metrics,
        out_path=out_path,
        extra_model_info=list(hyperparams.items()),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost GBDT on ArchPower Event Dataset")
    parser.add_argument("--arch", default="BOOM", choices=["BOOM", "XS"])
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--learning_rate", type=float,
                        default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output",
                           f"gbdt_{args.arch}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/3] Loading {args.arch} event dataset ...")
    data = prepare_data_numpy(args.arch, val_ratio=args.val_ratio,
                              seed=args.seed)
    n_train = data["X_train"].shape[0]
    n_val = data["X_val"].shape[0]
    n_targets = data["y_train"].shape[1]
    print(f"  train: {n_train} samples  val: {n_val} samples  "
          f"features: {data['X_train'].shape[1]}  targets: {n_targets}")

    print(f"[2/3] Training {n_targets} XGBRegressor models "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}) ...")
    models = train_models(
        data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print("[3/3] Evaluating ...")
    _, val_pred = predict_both(models, data)
    metrics = compute_metrics(val_pred, data["y_val_raw"])

    print(f"\n  {'Component':12s} {'RMSE':>10s} {'MAPE%':>8s} {'R2':>8s}")
    for i, name in enumerate(COMP_NAMES):
        print(f"  {name:12s} {metrics['rmse'][i]:10.6f} "
              f"{metrics['mape'][i]:8.2f}% {metrics['r2'][i]:8.4f}")

    write_report(
        arch=args.arch,
        data=data,
        metrics=metrics,
        val_ratio=args.val_ratio,
        hyperparams={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        out_path=os.path.join(out_dir, "report.md"),
    )

    print(f"\nDone!  All outputs in: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
