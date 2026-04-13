"""
Build event-only dataset by removing hardware-parameter dependence from power
labels using PANDA's resource functions.

For each component, the resource function R_c(hw) captures the dominant
hardware-driven scaling of power. Dividing the raw component power by R_c(hw)
yields a transformed label that depends only on workload (event) variation.

Output (saved to ../dataset/event_dataset/):
  {BOOM,XS}_event_feature.npy  — event-only features  (N, 81)
  {BOOM,XS}_event_label.npy    — transformed labels    (N, 12)
      column 0  = total (sum of components 1–11)
      columns 1–11 = per-component transformed power
"""

import numpy as np
import os

# ---------------------------------------------------------------------------
# Component definitions (mirrored from DataLoader.py)
# ---------------------------------------------------------------------------
COMP_NAMES = [
    "BP", "ICache", "IFU", "RNU", "LSU",
    "DCache", "Regfile", "ISU", "ROB", "FU-Pool", "Others",
]

# encode_table from PANDA.py — indices are local to each component's feature
ENCODE_TABLE = {
    "BP":      [0],
    "ICache":  [3, 4],
    "DCache":  [0, 1],
    "ISU":     [0],
    "Others":  [1],
    "IFU":     [1],
    "ROB":     [1],
    "Regfile": [1, 2],
    "RNU":     [1],
    "LSU":     [0],
}

# ---------------------------------------------------------------------------
# Resource-function helpers (from PANDA.py)
# ---------------------------------------------------------------------------

def compute_reserve_station_entries(decodewidth_init):
    """ISU-specific: map decodewidth to total reservation-station entries."""
    decodewidth = int(decodewidth_init + 0.01)
    isu_params = [
        [8,  decodewidth,  8, decodewidth,  8, decodewidth],
        [12, decodewidth, 20, decodewidth, 16, decodewidth],
        [16, decodewidth, 32, decodewidth, 24, decodewidth],
        [24, decodewidth, 40, decodewidth, 32, decodewidth],
        [24, decodewidth, 40, decodewidth, 32, decodewidth],
    ]
    p = isu_params[decodewidth - 1]
    return p[0] + p[2] + p[4]


def resource_function(comp_name, comp_feat):
    """
    Compute the resource-function scale factor R_c(hw) for each sample.

    Parameters
    ----------
    comp_name : str
        Component name (must be in COMP_NAMES).
    comp_feat : ndarray, shape (N, D_c)
        Per-component feature matrix.

    Returns
    -------
    scale : ndarray, shape (N,)
        R_c(hw) for every sample.  The caller does: label' = label / scale.
        Returns None for components with no resource function (FU-Pool).
    """
    if comp_name in ("BP", "ICache", "DCache", "RNU", "ROB", "IFU", "LSU"):
        scale = np.ones(comp_feat.shape[0])
        for idx in ENCODE_TABLE[comp_name]:
            scale = scale * comp_feat[:, idx]
        return scale

    if comp_name == "Regfile":
        scale = np.zeros(comp_feat.shape[0])
        for idx in ENCODE_TABLE[comp_name]:
            scale = scale + comp_feat[:, idx]
        return scale

    if comp_name == "ISU":
        idx = ENCODE_TABLE[comp_name][0]
        dw = comp_feat[:, idx]
        return np.array([compute_reserve_station_entries(d) for d in dw],
                        dtype=float)

    if comp_name == "Others":
        # logic_bias is effectively 0 in PANDA.py (Python scoping issue)
        idx = ENCODE_TABLE[comp_name][0]
        return comp_feat[:, idx] + 0.0  # + logic_bias (=0)

    # FU-Pool: no resource function
    return None


# ---------------------------------------------------------------------------
# Identify event-feature columns in the global feature.npy (101-d)
# ---------------------------------------------------------------------------

def find_event_feature_indices(feat, n_workloads=8):
    """Return sorted list of column indices that vary within config groups."""
    n_samples, n_cols = feat.shape
    n_configs = n_samples // n_workloads
    event_cols = []
    for col in range(n_cols):
        is_hw = True
        for cfg in range(n_configs):
            block = feat[cfg * n_workloads:(cfg + 1) * n_workloads, col]
            if not np.all(block == block[0]):
                is_hw = False
                break
        if not is_hw:
            event_cols.append(col)
    return sorted(event_cols)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_event_dataset(uarch, feat_all, label_all, comp_feats):
    """
    Build the event-only dataset for one architecture.

    Parameters
    ----------
    uarch : str
        'BOOM' or 'XS'.
    feat_all : ndarray (N_arch, 101)
    label_all : ndarray (N_arch, 12)   — power_group=0 (total power)
    comp_feats : dict[str, ndarray]     — per-component feature matrices

    Returns
    -------
    event_feat : ndarray (N_arch, 81)
    event_label : ndarray (N_arch, 12)
    """
    n = feat_all.shape[0]

    # Identify event-only columns from the global feature matrix
    event_cols = find_event_feature_indices(feat_all)
    event_feat = feat_all[:, event_cols]

    # Transform per-component labels
    transformed_comp = np.zeros((n, len(COMP_NAMES)))
    for i, comp_name in enumerate(COMP_NAMES):
        comp_label = label_all[:, i + 1]  # columns 1..11 are per-component
        cf = comp_feats[comp_name]
        scale = resource_function(comp_name, cf)
        if scale is not None:
            transformed_comp[:, i] = comp_label / scale
        else:
            transformed_comp[:, i] = comp_label  # FU-Pool: identity

    # Total = sum of transformed components
    transformed_total = transformed_comp.sum(axis=1)

    # Assemble label: [total, comp_0, comp_1, ..., comp_10]
    event_label = np.column_stack([transformed_total, transformed_comp])

    return event_feat, event_label


def main():
    base = os.path.join(os.path.dirname(__file__), '..', 'dataset')
    out_dir = os.path.join(base, 'event_dataset')
    os.makedirs(out_dir, exist_ok=True)

    # Load raw data
    feat_all = np.load(os.path.join(base, 'feature.npy'))
    label_raw = np.load(os.path.join(base, 'label.npy'))
    label_all = label_raw.reshape((label_raw.shape[0], 12, 5))[:, :, 0]  # power_group=0

    # Load per-component features
    comp_feats_all = {}
    for comp_name in COMP_NAMES:
        comp_feats_all[comp_name] = np.load(
            os.path.join(base, 'component_feature', f'{comp_name}.npy'))

    # Process each architecture
    archs = {
        'BOOM': (slice(0, 120), slice(0, 120)),
        'XS':   (slice(120, 200), slice(120, 200)),
    }

    for arch, (s_feat, s_label) in archs.items():
        feat = feat_all[s_feat]
        lab  = label_all[s_label]
        cfs  = {c: comp_feats_all[c][s_feat] for c in COMP_NAMES}

        event_feat, event_label = build_event_dataset(arch, feat, lab, cfs)

        # Save
        np.save(os.path.join(out_dir, f'{arch}_event_feature.npy'), event_feat)
        np.save(os.path.join(out_dir, f'{arch}_event_label.npy'), event_label)

        # Report
        print(f'[{arch}]')
        print(f'  event_feature shape: {event_feat.shape}')
        print(f'  event_label   shape: {event_label.shape}')
        print(f'  event feature columns: {find_event_feature_indices(feat)[:5]}... '
              f'({event_feat.shape[1]} total)')
        print(f'  label range: [{event_label.min():.4f}, {event_label.max():.4f}]')
        print()

    print(f'Saved to {os.path.abspath(out_dir)}/')


if __name__ == '__main__':
    main()
