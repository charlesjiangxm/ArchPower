"""
Event-to-Event Attention Discovery using FT-Transformer
on the ArchPower Event Dataset.

Uses a lightweight Feature Tokenizer Transformer to learn event (feature)
interactions for CPU power prediction, then extracts and visualizes
the attention weight matrices as event-to-event interaction maps.

Pipeline:
  1. Load & normalize event-only dataset
  2. (Optional) Self-supervised pre-training via masked feature modeling
  3. Supervised training: multi-target regression (12 power components)
  4. Extract attention maps → 81×81 event interaction matrices
  5. Visualize: heatmaps, top interactions, CLS attention bar chart

Usage:
    cd src
    python event_attention.py                         # defaults: BOOM, d=32, L=2
    python event_attention.py --arch XS --d_token 48  # XS with larger embedding
    python event_attention.py --no-pretrain            # skip self-supervised phase
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

from event_utils import generate_report as _generate_report

# ---------------------------------------------------------------------------
# Component / label names (consistent with build_event_dataset.py)
# ---------------------------------------------------------------------------
COMP_NAMES = [
    "Total", "BP", "ICache", "IFU", "RNU", "LSU",
    "DCache", "Regfile", "ISU", "ROB", "FU-Pool", "Others",
]

# Event feature names (81 features, matching global indices from feature.npy
# after removing 20 hardware-parameter columns).
# See doc/feature_description.md for full details.
EVENT_NAMES = [
    # Global 14–32 (local 0–18)
    "totalIpc",                       # 14
    "totalCpi",                       # 15
    "numCycles",                      # 16
    "idleCycles",                     # 17
    "BTBLookups",                     # 18
    "condPredicted",                  # 19
    "condIncorrect",                  # 20
    "intAluAccesses",                 # 21
    "fpAluAccesses",                  # 22
    "numLoadInsts",                   # 23
    "functionCalls",                  # 24
    "numSquashedInsts",               # 25
    "committedInsts",                 # 26
    "commit.numDist::mean",           # 27
    "commit.memRefs",                 # 28
    "numBranches",                    # 29
    "decode.runCycles",               # 30
    "decode.blockedCycles",           # 31
    "decode.decodedInsts",            # 32
    # Global 35–59 (local 19–43)
    "fetch.insts",                    # 35
    "fetch.branches",                 # 36
    "fetch.cycles",                   # 37
    "numRefs",                        # 38
    "numStoreInsts",                  # 39
    "numInsts",                       # 40
    "intIQWakeup",                    # 41
    "numBranches_dup",                # 42
    "numIssuedDist::mean",            # 43
    "intIQReads",                     # 44
    "intIQWrites",                    # 45
    "intIQWakeup_dup",                # 46
    "fpIQReads",                      # 47
    "fpIQWrites",                     # 48
    "fpIQWakeup",                     # 49
    "intAluAccesses_dup",             # 50
    "fpAluAccesses_dup",              # 51
    "issuedTotal",                    # 52
    "IssuedMemRead",                  # 53
    "IssuedMemWrite",                 # 54
    "IssuedFloatMemRead",             # 55
    "IssuedFloatMemWrite",            # 56
    "IssuedIntAlu",                   # 57
    "IssuedIntMult",                  # 58
    "IssuedIntDiv",                   # 59
    # Global 62 (local 44)
    "fuBusy",                         # 62
    # Global 65–100 (local 45–80)
    "conflictingLoads",               # 65
    "conflictingStores",              # 66
    "insertedLoads",                  # 67
    "insertedStores",                 # 68
    "rename.intLookups",              # 69
    "rename.renamedOperands",         # 70
    "rename.fpLookups",               # 71
    "rename.renamedInsts",            # 72
    "rename.runCycles",               # 73
    "rename.blockCycles",             # 74
    "rename.committedMaps",           # 75
    "rob.reads",                      # 76
    "rob.writes",                     # 77
    "mem_ctrls.readReqs",             # 78
    "mem_ctrls.writeReqs",            # 79
    "mem_ctrls.bytesReadSys",         # 80
    "icache.overallAccesses",         # 81
    "icache.overallMisses",           # 82
    "icache.mshrHits",                # 83
    "icache.mshrMisses",              # 84
    "icache.tags.totalRefs",          # 85
    "icache.tagAccesses",             # 86
    "dcache.ReadReq.accesses",        # 87
    "dcache.WriteReq.accesses",       # 88
    "dcache.ReadReq.misses",          # 89
    "dcache.WriteReq.misses",         # 90
    "dcache.overallAccesses",         # 91
    "dcache.overallMisses",           # 92
    "dcache.MshrHits",                # 93
    "dcache.MshrMisses",              # 94
    "dcache.tags.totalRefs",          # 95
    "dcache.tagAccesses",             # 96
    "intRegfileReads",                # 97
    "fpRegfileReads",                 # 98
    "intRegfileWrites",               # 99
    "fpRegfileWrites",                # 100
]

# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class NumericalFeatureTokenizer(nn.Module):
    """
    Embed each continuous feature independently into d-dimensional space.
    Each feature has its own linear projection (no weight sharing).
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.weights = nn.Parameter(torch.empty(n_features, d_token))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) → tokens: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with extractable attention."""
    def __init__(self, d_token: int, n_heads: int, d_ffn: int,
                 dropout: float = 0.3, attn_dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=attn_dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x, need_weights=False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            need_weights=need_weights,
            average_attn_weights=False,  # keep per-head weights
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class EventFTTransformer(nn.Module):
    """
    FT-Transformer for event-to-event interaction discovery.

    Input:  (batch, n_features) continuous event features
    Output: (batch, n_targets) power predictions  [+ attention maps]
    """
    def __init__(self, n_features=81, n_targets=12, d_token=32,
                 n_blocks=2, n_heads=4, d_ffn=64,
                 dropout=0.3, attn_dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.xavier_uniform_(self.cls_token)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout, attn_dropout)
            for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_targets)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, n_features)
            return_attention: if True return (pred, list_of_attn_weights)
        """
        batch = x.size(0)
        tokens = self.tokenizer(x)                          # (B, F, d)
        cls = self.cls_token.expand(batch, -1, -1)           # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)             # (B, 1+F, d)

        attn_maps = []
        for block in self.blocks:
            tokens, aw = block(tokens, need_weights=return_attention)
            if return_attention and aw is not None:
                attn_maps.append(aw)                         # (B, H, 1+F, 1+F)

        cls_out = self.final_norm(tokens[:, 0, :])           # (B, d)
        pred = self.head(cls_out)                            # (B, T)

        if return_attention:
            return pred, attn_maps
        return pred


class MaskedFeaturePretrainer(nn.Module):
    """
    Self-supervised pre-training: reconstruct randomly-masked features
    from the remaining context.  Teaches feature-to-feature dependencies
    without requiring labels.
    """
    def __init__(self, base_model: EventFTTransformer, mask_ratio=0.15):
        super().__init__()
        self.base = base_model
        self.mask_ratio = mask_ratio
        self.recon_head = nn.Linear(base_model.d_token, 1)

    def forward(self, x):
        bsz, n_feat = x.shape
        mask = torch.rand(bsz, n_feat, device=x.device) < self.mask_ratio

        x_masked = x.clone()
        x_masked[mask] = 0.0

        tokens = self.base.tokenizer(x_masked)
        cls = self.base.cls_token.expand(bsz, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for block in self.base.blocks:
            tokens, _ = block(tokens)

        tokens = self.base.final_norm(tokens)
        feat_tokens = tokens[:, 1:, :]                       # (bsz, n_feat, d)
        recon = self.recon_head(feat_tokens).squeeze(-1)      # (bsz, n_feat)

        loss = F.mse_loss(recon[mask], x[mask])
        return loss


# ---------------------------------------------------------------------------
# Data loading & preparation
# ---------------------------------------------------------------------------

def load_event_dataset(arch: str, base_dir: str = None):
    """Load raw event-only dataset for one architecture."""
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "event_dataset")
    feat = np.load(os.path.join(base_dir, f"{arch}_event_feature.npy"))
    label = np.load(os.path.join(base_dir, f"{arch}_event_label.npy"))
    return feat, label


def prepare_data(arch="BOOM", val_ratio=0.2, seed=42, base_dir=None):
    """
    Load, normalize, and split into train / val (by config group).
    Returns dict with tensors + normalization stats.
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

    def _t(a):
        return torch.FloatTensor(a)

    return {
        "X_train": _t(feat_norm[train_idx]),
        "y_train": _t(label_norm[train_idx]),
        "X_val": _t(feat_norm[val_idx]),
        "y_val": _t(label_norm[val_idx]),
        "X_all": _t(feat_norm),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "label_mean": label_mean,
        "label_std": label_std,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def pretrain(model, X_all, epochs=200, lr=1e-3, mask_ratio=0.15):
    """Self-supervised masked-feature pre-training."""
    pretrainer = MaskedFeaturePretrainer(model, mask_ratio=mask_ratio)
    optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=lr, weight_decay=1e-4)
    pretrainer.train()
    for epoch in range(epochs):
        loss = pretrainer(X_all)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
        optimizer.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  [pretrain] epoch {epoch:4d}  loss={loss.item():.6f}")
    return model


def train_supervised(model, data, epochs=500, lr=1e-3, weight_decay=1e-3,
                     patience=50, noise_std=0.02):
    """
    Supervised training with early stopping, noise augmentation, and
    cosine learning-rate schedule.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        X = data["X_train"]
        y = data["y_train"]

        # Gaussian noise augmentation
        X_aug = X + torch.randn_like(X) * noise_std
        pred = model(X_aug)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(data["X_val"])
            val_loss = F.mse_loss(val_pred, data["y_val"]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [train] early stop at epoch {epoch}  "
                      f"best_val_loss={best_val_loss:.6f}")
                break

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  [train] epoch {epoch:4d}  "
                  f"train_loss={loss.item():.6f}  val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ---------------------------------------------------------------------------
# Attention extraction & analysis
# ---------------------------------------------------------------------------

def extract_attention(model, X, device="cpu"):
    """
    Forward-pass all samples and collect attention maps.

    Returns dict with aggregated matrices:
        layer_{l}_mean         : (F, F)  mean over samples & heads
        layer_{l}_head_{h}     : (F, F)  mean over samples for head h
        layer_{l}_cls_attn     : (F,)    CLS-to-feature attention
    """
    model.eval()
    X_t = X if isinstance(X, torch.Tensor) else torch.FloatTensor(X)
    X_t = X_t.to(device)

    with torch.no_grad():
        _, attn_maps = model(X_t, return_attention=True)

    results = {}
    for li, aw in enumerate(attn_maps):
        # aw: (N, H, 1+F, 1+F)
        feat_attn = aw[:, :, 1:, 1:]          # (N, H, F, F)
        results[f"layer_{li}_mean"] = feat_attn.mean(dim=[0, 1]).cpu().numpy()
        n_heads = aw.size(1)
        for h in range(n_heads):
            results[f"layer_{li}_head_{h}"] = feat_attn[:, h].mean(dim=0).cpu().numpy()

        cls_attn = aw[:, :, 0, 1:]            # (N, H, F)
        results[f"layer_{li}_cls_attn"] = cls_attn.mean(dim=[0, 1]).cpu().numpy()

    # Attention rollout (product across layers)
    if len(attn_maps) > 1:
        rollout = attn_maps[0].mean(dim=1)     # (N, 1+F, 1+F) — avg over heads
        for aw in attn_maps[1:]:
            rollout = torch.bmm(aw.mean(dim=1), rollout)
        feat_rollout = rollout[:, 1:, 1:]
        results["rollout_mean"] = feat_rollout.mean(dim=0).cpu().numpy()

    return results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def plot_attention_heatmap(attn_matrix, title="Event-to-Event Attention",
                           out_path=None, feature_names=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = attn_matrix.shape[0]
    names = feature_names if feature_names is not None else [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(attn_matrix, cmap="viridis", ax=ax,
                xticklabels=names, yticklabels=names)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key (attended-to event)")
    ax.set_ylabel("Query (attending event)")
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6, rotation=0)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {out_path}")
    plt.close(fig)
    return fig


def plot_top_interactions(attn_matrix, k=20, title="Top Event Interactions",
                          out_path=None, feature_names=None):
    import matplotlib.pyplot as plt

    n = attn_matrix.shape[0]
    names = feature_names if feature_names is not None else [f"e{i}" for i in range(n)]

    A = attn_matrix.copy()
    np.fill_diagonal(A, 0)
    flat = np.argsort(A.flatten())[-k:][::-1]
    rows, cols = np.unravel_index(flat, A.shape)

    labels = [f"{names[r]} → {names[c]}" for r, c in zip(rows, cols)]
    values = [A[r, c] for r, c in zip(rows, cols)]

    fig, ax = plt.subplots(figsize=(10, max(6, k * 0.35)))
    ax.barh(range(k), values[::-1], color="steelblue")
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Attention Weight")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {out_path}")
    plt.close(fig)
    return fig


def plot_cls_attention(cls_attn, title="CLS → Feature Attention",
                       out_path=None, feature_names=None):
    import matplotlib.pyplot as plt

    n = len(cls_attn)
    names = feature_names if feature_names is not None else [f"e{i}" for i in range(n)]
    top_idx = np.argsort(cls_attn)[-20:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                              gridspec_kw={"height_ratios": [2, 1]})

    # Full bar chart with event names on x-axis
    axes[0].bar(range(n), cls_attn, color="teal", width=0.8)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(names, fontsize=5, rotation=90)
    axes[0].set_ylabel("Attention weight")
    axes[0].set_title(title, fontsize=13)

    # Top-20 zoomed with event names
    axes[1].barh(range(len(top_idx)), cls_attn[top_idx][::-1], color="darkorange")
    axes[1].set_yticks(range(len(top_idx)))
    axes[1].set_yticklabels([names[i] for i in top_idx[::-1]], fontsize=8)
    axes[1].set_xlabel("Attention weight")
    axes[1].set_title("Top-20 attended features", fontsize=11)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {out_path}")
    plt.close(fig)
    return fig


def plot_correlation_vs_attention(attn_matrix, X_np, out_path=None):
    """Scatter plot: Pearson |correlation| vs attention weight for all pairs."""
    import matplotlib.pyplot as plt

    corr = np.corrcoef(X_np.T)
    F = attn_matrix.shape[0]
    upper = np.triu_indices(F, k=1)
    corr_vals = np.abs(corr[upper])
    attn_sym = (attn_matrix + attn_matrix.T) / 2
    attn_vals = attn_sym[upper]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(corr_vals, attn_vals, alpha=0.3, s=10)
    ax.set_xlabel("|Pearson Correlation|")
    ax.set_ylabel("Attention Weight (symmetrized)")
    ax.set_title("Correlation vs Learned Attention")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  saved → {out_path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(model, data):
    """Compute MSE in original (un-normalized) label space."""
    model.eval()
    label_mean = data["label_mean"]
    label_std = data["label_std"]

    with torch.no_grad():
        pred_val = model(data["X_val"]).cpu().numpy()
        pred_all = model(data["X_all"]).cpu().numpy()

    pred_val_orig = pred_val * label_std + label_mean
    true_val_orig = data["y_val"].numpy() * label_std + label_mean
    pred_all_orig = pred_all * label_std + label_mean
    true_all_orig = data["y_all_raw"] if "y_all_raw" in data else None

    mse_per_comp = ((pred_val_orig - true_val_orig) ** 2).mean(axis=0)
    rmse_per_comp = np.sqrt(mse_per_comp)
    mape_per_comp = np.mean(
        np.abs(pred_val_orig - true_val_orig) / (np.abs(true_val_orig) + 1e-12),
        axis=0,
    ) * 100
    r2_per_comp = np.array([
        r2_score(true_val_orig[:, i], pred_val_orig[:, i])
        for i in range(true_val_orig.shape[1])
    ])

    print("\n  === Validation Metrics (original scale) ===")
    print(f"  {'Component':12s} {'RMSE':>10s} {'MAPE%':>8s} {'R2':>8s}")
    for i, name in enumerate(COMP_NAMES):
        print(f"  {name:12s} {rmse_per_comp[i]:10.6f} {mape_per_comp[i]:8.2f}% {r2_per_comp[i]:8.4f}")

    return {"rmse": rmse_per_comp, "mape": mape_per_comp, "r2": r2_per_comp}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Event-to-Event Attention Discovery (FT-Transformer)")
    parser.add_argument("--arch", default="BOOM", choices=["BOOM", "XS"],
                        help="Architecture to analyze")
    parser.add_argument("--d_token", type=int, default=32,
                        help="Token embedding dimension")
    parser.add_argument("--n_blocks", type=int, default=2,
                        help="Number of transformer blocks")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--d_ffn", type=int, default=64,
                        help="FFN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attn_dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrain", action="store_true",
                        help="Skip self-supervised pre-training")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output",
                           f"attention_{args.arch}")
    os.makedirs(out_dir, exist_ok=True)

    # --- data ---
    print(f"[1/5] Loading {args.arch} event dataset …")
    data = prepare_data(args.arch, val_ratio=args.val_ratio, seed=args.seed)
    print(f"  train: {data['X_train'].shape[0]} samples  "
          f"val: {data['X_val'].shape[0]} samples  "
          f"features: {data['X_train'].shape[1]}")

    # --- model ---
    print(f"[2/5] Building FT-Transformer "
          f"(d={args.d_token}, L={args.n_blocks}, H={args.n_heads}) …")
    model = EventFTTransformer(
        n_features=data["X_train"].shape[1],
        n_targets=data["y_train"].shape[1],
        d_token=args.d_token,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        d_ffn=args.d_ffn,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # --- pre-train ---
    if not args.no_pretrain:
        print(f"[3/5] Self-supervised pre-training ({args.pretrain_epochs} epochs) …")
        model = pretrain(model, data["X_all"],
                         epochs=args.pretrain_epochs, lr=args.lr,
                         mask_ratio=args.mask_ratio)
    else:
        print("[3/5] Skipping pre-training (--no-pretrain)")

    # --- supervised training ---
    print(f"[4/5] Supervised training ({args.epochs} epochs, patience={args.patience}) …")
    model, best_val = train_supervised(
        model, data,
        epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        noise_std=args.noise_std,
    )
    print(f"  best validation loss (normalized): {best_val:.6f}")

    # --- evaluate ---
    metrics = evaluate_predictions(model, data)

    # --- report ---
    n_train = data["X_train"].shape[0]
    n_val = data["X_val"].shape[0]
    n_val_configs = n_val // 8
    n_configs = n_train // 8 + n_val_configs
    _generate_report(
        arch=args.arch,
        model_name="FT-Transformer (Event Attention)",
        model_description="Feature Tokenizer Transformer",
        n_params=n_params,
        n_train=n_train, n_val=n_val,
        n_configs=n_configs, n_val_configs=n_val_configs,
        val_ratio=args.val_ratio,
        metrics=metrics,
        out_path=os.path.join(out_dir, "report.md"),
        extra_model_info=[
            ("d_token", args.d_token),
            ("n_blocks", args.n_blocks),
            ("n_heads", args.n_heads),
            ("d_ffn", args.d_ffn),
            ("dropout", args.dropout),
            ("attn_dropout", args.attn_dropout),
            ("pretrain_epochs", 0 if args.no_pretrain else args.pretrain_epochs),
            ("supervised_epochs", args.epochs),
            ("patience", args.patience),
        ],
    )

    # --- attention extraction ---
    print(f"[5/5] Extracting & visualizing attention maps …")
    attn = extract_attention(model, data["X_all"])

    # Save raw attention matrices
    for key, mat in attn.items():
        np.save(os.path.join(out_dir, f"{key}.npy"), mat)
    print(f"  saved {len(attn)} attention matrices to {out_dir}/")

    # --- plots ---
    # Last-layer mean attention heatmap
    last_layer = args.n_blocks - 1
    mean_key = f"layer_{last_layer}_mean"
    if mean_key in attn:
        plot_attention_heatmap(
            attn[mean_key],
            title=f"{args.arch} Event Attention — Layer {last_layer} (mean over heads)",
            out_path=os.path.join(out_dir, "heatmap_last_layer_mean.png"),
            feature_names=EVENT_NAMES,
        )
        plot_top_interactions(
            attn[mean_key], k=20,
            title=f"{args.arch} Top-20 Event Interactions (Layer {last_layer})",
            out_path=os.path.join(out_dir, "top_interactions.png"),
            feature_names=EVENT_NAMES,
        )

    # CLS attention
    cls_key = f"layer_{last_layer}_cls_attn"
    if cls_key in attn:
        plot_cls_attention(
            attn[cls_key],
            title=f"{args.arch} CLS → Feature Attention (Layer {last_layer})",
            out_path=os.path.join(out_dir, "cls_attention.png"),
            feature_names=EVENT_NAMES,
        )

    # Per-head heatmaps (last layer)
    for h in range(args.n_heads):
        hk = f"layer_{last_layer}_head_{h}"
        if hk in attn:
            plot_attention_heatmap(
                attn[hk],
                title=f"{args.arch} Attention — Layer {last_layer} Head {h}",
                out_path=os.path.join(out_dir, f"heatmap_L{last_layer}_H{h}.png"),
                feature_names=EVENT_NAMES,
            )

    # Rollout
    if "rollout_mean" in attn:
        plot_attention_heatmap(
            attn["rollout_mean"],
            title=f"{args.arch} Attention Rollout (all layers)",
            out_path=os.path.join(out_dir, "heatmap_rollout.png"),
            feature_names=EVENT_NAMES,
        )

    # Correlation vs attention scatter
    feat_raw, _ = load_event_dataset(args.arch)
    plot_correlation_vs_attention(
        attn[mean_key], feat_raw,
        out_path=os.path.join(out_dir, "correlation_vs_attention.png"),
    )

    print(f"\nDone!  All outputs in: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
