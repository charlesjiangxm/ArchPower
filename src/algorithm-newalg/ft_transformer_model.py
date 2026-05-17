"""
Minimal FT-Transformer for the c906 power-modeling pipeline.

Self-contained PyTorch implementation of the architecture from
Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data"
(NeurIPS 2021).  Single-target regression; numerical features only
(c906 has no categorical signals).

Public API
----------
- FTTransformer(nn.Module): the model.
- default_device(pref): pick mps/cuda/cpu.
- Standardizer: float64 z-score for X and y, fit on training rows only.
- train_ft_transformer(...): AdamW + CosineAnnealingLR + early stopping.
- predict(model, X, std, device): inverse-standardized predictions.
- extract_attention(model, X, std, device): returns (cls_attn, ff_attn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def default_device(pref="auto"):
    """Resolve a torch device.

    - `auto`: prefer CUDA > MPS > CPU. CUDA is preferred when both are
      present (e.g., a CUDA workstation running PyTorch nightly with MPS
      build flags) because dense MHSA throughput is consistently higher
      on CUDA than on MPS.
    - `cuda` / `mps`: require the backend to be available, raise otherwise.
    - `cpu`: always succeeds.
    """
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--ft_device cuda requested but torch.cuda.is_available() is False. "
                "Pass --ft_device auto to fall back, or install a CUDA-enabled PyTorch."
            )
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "--ft_device mps requested but torch.backends.mps.is_available() is False."
            )
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device):
    """Human-readable label for the resolved device, e.g.
    'cuda (NVIDIA A100-SXM4-80GB)' or 'mps (Apple Silicon GPU)' or 'cpu'."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon GPU)"
    return "cpu"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NumericalFeatureTokenizer(nn.Module):
    """Tokenize numerical features per the FT-Transformer paper, sec 3.3.

    For each feature j and each example i:  T_ij = b_j + x_ij * W_j,
    where W_j, b_j are vectors of length d_token.
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, F) -> (B, F, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    """PreNorm Transformer block (MHSA + FFN), batch_first.

    If `is_first=True`, the first LayerNorm before MHSA is omitted, per
    the FT-Transformer paper section 3.3 ("we found it to be necessary
    to remove the first normalization from the first Transformer layer").
    """

    def __init__(self, d_token, n_heads, d_ffn, dropout, attn_dropout,
                 is_first=False):
        super().__init__()
        self.is_first = is_first
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        # MHSA branch (PreNorm; first block skips norm1).
        x_for_attn = x if self.is_first else self.norm1(x)
        attn_out, attn_w = self.attn(
            x_for_attn, x_for_attn, x_for_attn,
            need_weights=return_attention,
            average_attn_weights=False,  # keep per-head for richer extraction
        )
        x = x + self.attn_dropout(attn_out)

        # FFN branch.
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.ffn_dropout(ffn_out)
        return x, attn_w


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for single-target regression.

    Forward returns ``y`` of shape ``(B,)``. If ``return_attention=True``,
    also returns a list of per-block attention tensors of shape
    ``(B, n_heads, 1+F, 1+F)`` (in MultiheadAttention's per-head layout
    with ``average_attn_weights=False``).
    """

    def __init__(self, n_features, d_token=32, n_blocks=3, n_heads=4,
                 d_ffn=64, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.xavier_uniform_(self.cls_token)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout, attn_dropout,
                             is_first=(i == 0))
            for i in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, 1)

    def forward(self, x, return_attention=False):
        # x: (B, F)
        tokens = self.tokenizer(x)                          # (B, F, d)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)  # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, 1+F, d)

        attn_list = [] if return_attention else None
        for block in self.blocks:
            tokens, attn_w = block(tokens, return_attention=return_attention)
            if return_attention:
                attn_list.append(attn_w)

        cls_out = self.final_norm(tokens[:, 0, :])          # (B, d)
        y = self.head(F.relu(cls_out)).squeeze(-1)           # (B,)
        if return_attention:
            return y, attn_list
        return y


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0, ddof=0)
        x_std[x_std == 0] = 1.0
        y_mean = float(y.mean())
        y_std = float(y.std(ddof=0))
        if y_std == 0:
            y_std = 1.0
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    def transform_X(self, X):
        return (np.asarray(X, dtype=np.float64) - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (np.asarray(y, dtype=np.float64) - self.y_mean) / self.y_std

    def inverse_y(self, yz):
        return np.asarray(yz, dtype=np.float64) * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _to_tensor(arr, dtype, device):
    return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)


def train_ft_transformer(
    model,
    X_train, y_train,
    X_val, y_val,
    *,
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=300,
    patience=40,
    batch_size=256,
    device=None,
    input_noise_std=0.02,
    grad_clip=1.0,
    seed=42,
    use_amp=False,
    verbose=True,
):
    """Fit `model` with AdamW + cosine LR + early stopping. Returns
    (best_model, history_dict, standardizer).

    Parameters
    ----------
    use_amp : bool
        Enable mixed-precision autocast on CUDA (no-op on MPS / CPU).
        Combines `torch.amp.autocast` with `GradScaler` for an ~2x
        forward/backward speed-up on Ampere+ GPUs.
    """
    device = device or default_device("auto")
    torch.manual_seed(seed)
    np.random.seed(seed)

    std = Standardizer.fit(X_train, y_train)
    Xtr_z = std.transform_X(X_train)
    ytr_z = std.transform_y(y_train)
    Xva_z = std.transform_X(X_val)
    yva_z = std.transform_y(y_val)

    Xtr_t = _to_tensor(Xtr_z, torch.float32, device)
    ytr_t = _to_tensor(ytr_z, torch.float32, device)
    Xva_t = _to_tensor(Xva_z, torch.float32, device)
    yva_t = _to_tensor(yva_z, torch.float32, device)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_epochs))

    amp_enabled = bool(use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    if use_amp and not amp_enabled and verbose:
        print(f"    [train] --ft_amp requested but device={device.type}; "
              f"autocast is CUDA-only -- running in float32.")

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
            if input_noise_std > 0:
                xb = xb + torch.randn_like(xb) * input_noise_std
            yb = ytr_t[idx]
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(xb)
                loss = F.mse_loss(pred, yb)
            if amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict(model, X, std, device=None, batch_size=512):
    """Standardize X, forward in batches, inverse-transform y."""
    device = device or default_device("auto")
    model.eval()
    Xz = std.transform_X(X)
    Xz_t = _to_tensor(Xz, torch.float32, device)
    preds = []
    with torch.no_grad():
        for start in range(0, Xz_t.size(0), batch_size):
            preds.append(model(Xz_t[start:start + batch_size]).cpu().numpy())
    yz = np.concatenate(preds, axis=0) if preds else np.zeros((0,))
    return std.inverse_y(yz)


def extract_attention(model, X, std, device=None, batch_size=64):
    """Return (cls_attn, ff_attn) averaged across heads, layers, and examples.

    cls_attn : (F,)   -- CLS-token's attention to each feature.
    ff_attn  : (F, F) -- feature-feature attention block.
    """
    device = device or default_device("auto")
    model.eval()
    Xz = std.transform_X(X)
    Xz_t = _to_tensor(Xz, torch.float32, device)
    n = Xz_t.size(0)
    F_dim = model.n_features

    sum_cls = np.zeros(F_dim, dtype=np.float64)
    sum_ff = np.zeros((F_dim, F_dim), dtype=np.float64)
    n_seen = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = Xz_t[start:start + batch_size]
            _, attn_list = model(xb, return_attention=True)
            B = xb.size(0)

            # Average over (heads, layers).
            stacked = torch.stack(attn_list, dim=0)        # (L, B, H, 1+F, 1+F)
            mean_lh = stacked.mean(dim=(0, 2))             # (B, 1+F, 1+F)
            cls_b = mean_lh[:, 0, 1:].sum(dim=0)            # (F,)
            ff_b = mean_lh[:, 1:, 1:].sum(dim=0)            # (F, F)
            sum_cls += cls_b.cpu().numpy()
            sum_ff += ff_b.cpu().numpy()
            n_seen += B

    if n_seen == 0:
        return np.zeros(F_dim), np.zeros((F_dim, F_dim))
    return sum_cls / n_seen, sum_ff / n_seen
