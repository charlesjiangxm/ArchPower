"""
Shared utilities for event-dataset power modeling scripts.

Provides consistent data loading, train/test splitting, metric computation,
and markdown report generation across all event-dataset model scripts.
"""

import os
import numpy as np
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# Component / label names (consistent with build_event_dataset.py)
# ---------------------------------------------------------------------------
COMP_NAMES = [
    "Total", "BP", "ICache", "IFU", "RNU", "LSU",
    "DCache", "Regfile", "ISU", "ROB", "FU-Pool", "Others",
]


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_event_dataset(arch, base_dir=None):
    """Load raw event-only dataset for one architecture."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "event_dataset")
    feat = np.load(os.path.join(base_dir, f"{arch}_event_feature.npy"))
    label = np.load(os.path.join(base_dir, f"{arch}_event_label.npy"))
    return feat, label


def prepare_data_numpy(arch="BOOM", val_ratio=0.2, seed=42, base_dir=None):
    """
    Load, normalize, and split into train / val (by config group).

    The split logic exactly matches event_attention.py:prepare_data() so that
    all models are evaluated on the same validation set.

    Returns dict with numpy arrays + normalization stats + raw labels.
    """
    feat, label = load_event_dataset(arch, base_dir)
    n_workloads = 8
    n_configs = feat.shape[0] // n_workloads

    feat_mean = feat.mean(axis=0, keepdims=True)
    feat_std = feat.std(axis=0, keepdims=True) + 1e-8
    feat_norm = (feat - feat_mean) / feat_std

    label_mean = label.mean(axis=0, keepdims=True)
    label_std = label.std(axis=0, keepdims=True) + 1e-8
    label_norm = (label - label_mean) / label_std

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_configs)
    n_val = max(1, int(n_configs * val_ratio))
    val_cfgs = perm[:n_val]
    train_cfgs = perm[n_val:]

    train_idx = np.concatenate([np.arange(c * 8, (c + 1) * 8) for c in train_cfgs])
    val_idx = np.concatenate([np.arange(c * 8, (c + 1) * 8) for c in val_cfgs])

    return {
        "X_train": feat_norm[train_idx],
        "y_train": label_norm[train_idx],
        "X_val": feat_norm[val_idx],
        "y_val": label_norm[val_idx],
        "X_train_raw": feat[train_idx],
        "y_train_raw": label[train_idx],
        "X_val_raw": feat[val_idx],
        "y_val_raw": label[val_idx],
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "label_mean": label_mean,
        "label_std": label_std,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def denormalize(pred_norm, label_mean, label_std):
    """Convert normalized predictions back to original scale."""
    return pred_norm * label_std + label_mean


def compute_metrics(pred_orig, true_orig):
    """
    Compute per-component MAPE, RMSE, R2.

    Parameters
    ----------
    pred_orig, true_orig : ndarray, shape (N, 12)
        Predictions and ground truth in original (denormalized) scale.

    Returns
    -------
    dict with keys 'rmse', 'mape', 'r2', each a 12-element ndarray.
    """
    mse = ((pred_orig - true_orig) ** 2).mean(axis=0)
    rmse = np.sqrt(mse)
    mape = np.mean(
        np.abs(pred_orig - true_orig) / (np.abs(true_orig) + 1e-12), axis=0
    ) * 100
    r2 = np.array([
        r2_score(true_orig[:, i], pred_orig[:, i])
        for i in range(true_orig.shape[1])
    ])
    return {"rmse": rmse, "mape": mape, "r2": r2}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(arch, model_name, model_description, n_params,
                    n_train, n_val, n_configs, n_val_configs,
                    val_ratio, metrics, out_path,
                    extra_model_info=None):
    """
    Write a standardized markdown report.

    Parameters
    ----------
    arch : str
    model_name : str              Display name (e.g., "FT-Transformer")
    model_description : str       One-line model type
    n_params : int or str         Parameter count (or descriptive string)
    n_train, n_val : int          Sample counts
    n_configs, n_val_configs : int
    val_ratio : float
    metrics : dict                Output of compute_metrics()
    out_path : str                Path to write report.md
    extra_model_info : list of (key, value) tuples, optional
    """
    lines = [
        f"# {model_name} -- {arch} Power Prediction Report\n",
        "## Data Split\n",
        "| Property | Value |",
        "|---|---|",
        f"| Architecture | {arch} |",
        f"| Split method | Config-level holdout (val_ratio={val_ratio}) |",
        f"| Total configs | {n_configs} |",
        f"| Training configs | {n_configs - n_val_configs} |",
        f"| Testing configs | {n_val_configs} |",
        f"| Samples per config | 8 (workloads) |",
        f"| Training samples | {n_train} |",
        f"| Testing samples | {n_val} |",
        f"| Features | 81 (event-only) |",
        f"| Targets | 12 (power components) |",
        "",
        "## Model\n",
        "| Property | Value |",
        "|---|---|",
        f"| Model type | {model_description} |",
        f"| Parameters | {n_params if isinstance(n_params, str) else f'{n_params:,}'} |",
    ]
    if extra_model_info:
        for key, val in extra_model_info:
            lines.append(f"| {key} | {val} |")
    lines.append("")
    lines.append("## Testing Metrics (original scale)\n")
    lines.append("| Component | MAPE (%) | RMSE | R2 |")
    lines.append("|---|---|---|---|")
    for i, name in enumerate(COMP_NAMES):
        lines.append(
            f"| {name} | {metrics['mape'][i]:.2f} "
            f"| {metrics['rmse'][i]:.6f} "
            f"| {metrics['r2'][i]:.4f} |"
        )
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to {out_path}")
