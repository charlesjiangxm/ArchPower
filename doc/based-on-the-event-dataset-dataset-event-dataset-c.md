# Using Attention Mechanisms to Discover Event-to-Event Interactions in the ArchPower Event Dataset

## Executive Summary

This report investigates how to apply **self-attention mechanisms** to the ArchPower event-only dataset (`dataset/event_dataset/`) to discover which CPU performance events (features) "attend to" — i.e., interact with or influence — which other events in determining power consumption. The dataset has 81 continuous event features and 12 power labels (total + 11 components), with only 120 BOOM and 80 XS samples — a classic "wide, short" tabular regime. We recommend the **FT-Transformer** (Feature Tokenizer + Transformer) architecture as the primary approach, supplemented by aggressive regularization, data augmentation, and careful attention-weight extraction for interpretability. We also discuss alternative architectures (SAINT, TabNet, custom lightweight attention modules) and provide a concrete implementation blueprint with PyTorch code, tailored to this specific dataset.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Dataset Characteristics & Challenges](#2-dataset-characteristics--challenges)
3. [Architecture Survey: Attention for Tabular Data](#3-architecture-survey-attention-for-tabular-data)
4. [Recommended Approach: FT-Transformer](#4-recommended-approach-ft-transformer)
5. [Handling Small Data: Regularization & Augmentation](#5-handling-small-data-regularization--augmentation)
6. [Attention Weight Extraction & Interpretation](#6-attention-weight-extraction--interpretation)
7. [Concrete Implementation Blueprint](#7-concrete-implementation-blueprint)
8. [Alternative Approaches](#8-alternative-approaches)
9. [Experimental Design](#9-experimental-design)
10. [Confidence Assessment](#10-confidence-assessment)
11. [Footnotes](#11-footnotes)

---

## 1. Problem Formulation

### What We Want to Learn

Given the event-only dataset where:
- **Features** `X ∈ ℝ^{N×81}`: 81 event parameters (performance counter statistics from gem5 simulation) that vary across workloads
- **Labels** `Y ∈ ℝ^{N×12}`: hardware-normalized power for [Total, BP, ICache, IFU, RNU, LSU, DCache, Regfile, ISU, ROB, FU-Pool, Others]

We want to discover the **event-to-event interaction structure**: which pairs (or groups) of events jointly influence power consumption. Formally, we seek an **attention matrix** `A ∈ ℝ^{81×81}` where `A[i,j]` indicates how strongly event `i` "attends to" (interacts with) event `j` when the model predicts power.

### Why This Matters

In CPU power modeling, understanding event interactions reveals:
- Which microarchitectural activities are **correlated in their power impact** (e.g., cache misses driving memory traffic)
- **Bottleneck coupling**: how one pipeline stage's activity amplifies another's power
- **Feature redundancy**: which events carry overlapping information
- Guidance for **feature selection** in simpler downstream models

---

## 2. Dataset Characteristics & Challenges

### Data Summary

| Property | BOOM | XS |
|---|---|---|
| Samples | 120 | 80 |
| Features | 81 (continuous) | 81 (continuous) |
| Labels | 12 (continuous) | 12 (continuous) |
| Sample/Feature ratio | 1.48 | 0.99 |
| Combined ratio | 2.47 | — |

### Key Observations from Data Analysis

1. **Extreme scale differences**: Feature standard deviations span 9 orders of magnitude (0.0001 to 93,544)[^1]. **Normalization is mandatory**.

2. **High feature correlation**: 312 feature pairs have |corr| > 0.8; 121 pairs exceed |corr| > 0.95. There are 16 correlation clusters (|r| > 0.9), the largest containing 10 features[^2]. This is expected — CPU events are mechanistically coupled (e.g., instruction fetch, decode, and issue are part of the same pipeline).

3. **Component-specific feature relevance differs markedly**:
   - ISU power is driven by features 77, 30, 79, 28, 12 (IPC/instruction counts)
   - DCache power by features 42, 43, 68, 10, 23 (cache access/miss rates)
   - FU-Pool power by features 56, 49, 52, 18, 57 (instruction mix)[^3]

4. **Very small sample size**: 120/80 samples for 81 features makes overfitting the dominant risk. Standard deep-learning transformers (with millions of parameters) cannot be naively applied.

5. **Only 8 unique workloads**: Within each architecture, the 8 workloads repeat across configurations. After hardware normalization, the 120 BOOM samples contain approximately 8 distinct "event patterns" with residual variation from normalization artifacts.

### Implications for Architecture Design

- **Must use a lightweight model** — small embedding dimensions, few layers, few heads
- **Must normalize features** — z-score or quantile normalization
- **Should consider multi-target training** — predicting all 12 labels simultaneously provides implicit regularization
- **Cross-architecture training** may be beneficial if event semantics align between BOOM and XS

---

## 3. Architecture Survey: Attention for Tabular Data

### 3.1 FT-Transformer (Gorishniy et al., NeurIPS 2021)

The **Feature Tokenizer Transformer** treats each feature as a token[^4]:

```
Input: x ∈ ℝ^81 (one sample)
  ↓
Feature Tokenizer: each feature x_i → token t_i = W_i · x_i + b_i ∈ ℝ^d
  ↓
Prepend [CLS] token
  ↓
Transformer Encoder (L layers of Multi-Head Self-Attention + FFN)
  ↓
[CLS] output → Linear head → ŷ ∈ ℝ^12
```

**Why it fits our problem:**
- Designed for **all-continuous features** (unlike TabTransformer which focuses on categoricals)
- Each feature gets its own linear projection — no shared weights across features
- Self-attention explicitly computes feature-to-feature interactions
- [CLS] token aggregates information for prediction
- Attention weights are directly interpretable as event-to-event interaction scores

### 3.2 SAINT (Somepalli et al., 2021)

SAINT adds **inter-sample attention** (row-wise) on top of self-attention (column-wise)[^5]:

```
Feature Tokens → Self-Attention (which events interact?) 
              → Intersample Attention (which samples are similar?)
              → Repeat L times
```

**Pros**: Inter-sample attention lets the model "borrow" information from similar workloads, potentially helpful with 8 unique workload patterns.  
**Cons**: More parameters, more complex to train, inter-sample attention adds significant memory overhead.

### 3.3 TabNet (Arik & Pfister, 2021)

TabNet uses **sequential attention** for dynamic feature selection[^6]:

```
Step 1: Attend to most relevant features → partial prediction
Step 2: Attend to next most relevant features → refine prediction
...
Step K: Final prediction = sum of partial predictions
```

**Pros**: Built-in feature importance, sparse attention.  
**Cons**: Attention is "sample→feature" (which features matter for this sample), NOT "feature→feature" (which features interact with each other). **Less suitable for our goal**.

### 3.4 Custom Lightweight Attention Module

For extremely small datasets, a **single self-attention layer** without the full transformer overhead may suffice:

```
Input features → Linear embedding → Single Multi-Head Attention → Pool → Predict
```

This is the minimal viable architecture for extracting event-to-event attention.

### Comparison Table

| Architecture | Feature→Feature Attention | Small-Data Friendly | Interpretability | Complexity |
|---|---|---|---|---|
| **FT-Transformer** | ✅ Direct | Medium | High (attention maps) | Medium |
| SAINT | ✅ + inter-sample | Medium | Medium | High |
| TabNet | ❌ (sample→feature) | Good | High (feature masks) | Medium |
| Custom 1-layer | ✅ Direct | Best | Highest | Low |

**Recommendation**: Start with **FT-Transformer** (or its lightweight variant) as it directly provides feature-to-feature attention maps, has well-tested implementations, and handles all-continuous features natively.

---

## 4. Recommended Approach: FT-Transformer

### Architecture for ArchPower Event Dataset

```
┌──────────────────────────────────────────────────┐
│                   INPUT LAYER                     │
│  x ∈ ℝ^81 (one sample, 81 event features)       │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│              FEATURE TOKENIZER                    │
│  For each feature i = 0..80:                     │
│    t_i = W_i · x_i + b_i    (W_i ∈ ℝ^{d×1})   │
│  Prepend: t_CLS (learnable parameter)            │
│  Result: T ∈ ℝ^{82 × d}                         │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│          TRANSFORMER ENCODER × L                  │
│  ┌─────────────────────────────────────────┐     │
│  │ LayerNorm → Multi-Head Self-Attention   │     │
│  │ (heads=h, dim=d) + Residual             │     │
│  │                                         │     │
│  │ LayerNorm → FFN(d → 4d → d)             │     │
│  │ + Dropout + Residual                    │     │
│  └─────────────────────────────────────────┘     │
│  Repeat L times                                   │
│                                                   │
│  ★ EXTRACT: attention_weights[l][head]            │
│    shape: (82, 82) per head per layer             │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│               PREDICTION HEAD                     │
│  Take CLS token output: h_CLS ∈ ℝ^d             │
│  Linear(d → 12) → ŷ ∈ ℝ^12                      │
│  (predict all 12 component powers simultaneously)│
└──────────────────────────────────────────────────┘
```

### Hyperparameter Recommendations for Small Data

Given only 120–200 samples and 81 features:

| Hyperparameter | Recommended Value | Rationale |
|---|---|---|
| `d_token` (embedding dim) | 32–64 | Small to limit parameters |
| `n_blocks` (transformer layers) | 1–2 | Shallow to prevent overfitting |
| `n_heads` | 4–8 | More heads = more diverse attention patterns |
| `ffn_d_hidden` | 64–128 | Small FFN |
| `dropout` | 0.3–0.5 | Aggressive for small data |
| `attention_dropout` | 0.2–0.3 | Regularize attention |
| Total parameters | ~50K–200K | Much less than typical 1M+ FT-Transformer |

**Parameter count estimate**: With d=32, L=2, h=4, FFN=64:
- Feature tokenizer: 81 × (32 + 1) + 32 = ~2,705 params
- Per transformer block: ~4 × (32² × 4) + 2 × (32 × 64) = ~20K params
- Head: 32 × 12 = 384 params
- Total: ~43K params — reasonable for 120 samples

---

## 5. Handling Small Data: Regularization & Augmentation

### 5.1 Regularization Strategies

Given the extreme sample/parameter ratio, combine multiple strategies[^7]:

1. **High dropout** (0.3–0.5) in both attention and FFN layers
2. **Weight decay** (L2 regularization): λ = 1e-3 to 1e-2
3. **Early stopping** with patience 20–50 epochs on validation loss
4. **Label smoothing** (for regression: add small Gaussian noise to labels during training)
5. **Attention sparsity**: Apply top-k masking or temperature scaling to attention logits to encourage sparse, interpretable patterns[^8]
6. **Gradient clipping**: max_norm = 1.0

### 5.2 Data Augmentation for Tabular Data

| Technique | Description | Implementation |
|---|---|---|
| **Gaussian noise injection** | Add `N(0, σ·std_col)` noise to each feature | σ = 0.01–0.05; applied per-epoch |
| **Feature mixup** | `x' = λ·x_a + (1-λ)·x_b`, `y' = λ·y_a + (1-λ)·y_b` | λ ~ Beta(0.2, 0.2) |
| **Cross-architecture pooling** | Train on combined BOOM+XS data (200 samples) | Requires feature alignment (already identical) |
| **Config-aware augmentation** | Interpolate between samples from same config group | Preserves within-config structure |

### 5.3 Training Strategy

```
Phase 1: Pre-train with reconstruction loss (self-supervised)
  - Mask 15% of feature tokens randomly
  - Train model to reconstruct masked features from context
  - This learns feature interactions WITHOUT labels
  - Use ALL 200 samples (BOOM + XS)

Phase 2: Fine-tune with power prediction loss (supervised)
  - Multi-target MSE: L = Σ_c (ŷ_c - y_c)²
  - With early stopping on validation set
```

Self-supervised pre-training is particularly valuable here because:
- It doesn't require labels, so all 200 samples can be used
- Masked feature reconstruction directly encourages the model to learn **which features predict which other features** — exactly the event-to-event interaction structure we want

---

## 6. Attention Weight Extraction & Interpretation

### 6.1 Extracting Attention Weights

After training, attention weights from each layer and head give a `(82×82)` matrix (including CLS). We focus on the `(81×81)` submatrix (features only):

```python
def extract_attention_maps(model, X_batch):
    """
    Run forward pass and collect attention weights from all layers/heads.
    
    Returns:
        attn_maps: list of tensors, one per layer
                   each shape: (batch, n_heads, 82, 82)
    """
    model.eval()
    attn_maps = []
    hooks = []
    
    # Register hooks on attention layers
    for layer in model.transformer.layers:
        def hook_fn(module, input, output, attn_output=None):
            # PyTorch MultiheadAttention returns (output, attn_weights)
            # when need_weights=True
            pass
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))
    
    with torch.no_grad():
        # Forward pass with attention output
        output = model(X_batch, return_attention=True)
    
    # Clean up hooks
    for h in hooks:
        h.remove()
    
    return attn_maps
```

### 6.2 Aggregation Strategies

Raw attention is per-sample, per-head, per-layer. To get a global event-to-event interaction matrix:

| Strategy | Formula | Interpretation |
|---|---|---|
| **Mean over samples & heads** | `A = mean(attn[:, :, 1:, 1:], dim=[0,1])` | Average interaction strength |
| **Max over heads** | `A = max(attn[:, :, 1:, 1:], dim=1)` then mean over samples | Strongest interaction across any head |
| **Attention rollout** | `A = Π_l attn_l` (matrix product across layers) | Effective attention through all layers |
| **CLS-to-features** | `A_cls = attn[:, :, 0, 1:]` | Which features most influence final prediction |
| **Per-component** | Train separate models per component, compare attention maps | Component-specific interactions |

### 6.3 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention_heatmap(attn_matrix, feature_names=None, title="Event-to-Event Attention"):
    """
    Plot 81×81 attention matrix as heatmap.
    """
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(attn_matrix, 
                xticklabels=feature_names if feature_names else range(81),
                yticklabels=feature_names if feature_names else range(81),
                cmap='viridis', 
                ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Key (attended-to event)")
    ax.set_ylabel("Query (attending event)")
    plt.tight_layout()
    return fig


def plot_top_interactions(attn_matrix, k=20, feature_names=None):
    """
    Bar chart of top-k strongest event-to-event interactions.
    """
    # Zero out diagonal (self-attention)
    A = attn_matrix.copy()
    np.fill_diagonal(A, 0)
    
    # Find top-k
    flat_idx = np.argsort(A.flatten())[-k:][::-1]
    rows, cols = np.unravel_index(flat_idx, A.shape)
    
    labels = []
    values = []
    for r, c in zip(rows, cols):
        name_r = feature_names[r] if feature_names else f"f{r}"
        name_c = feature_names[c] if feature_names else f"f{c}"
        labels.append(f"{name_r} → {name_c}")
        values.append(A[r, c])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(k), values[::-1])
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels[::-1])
    ax.set_xlabel("Attention Weight")
    ax.set_title(f"Top-{k} Event-to-Event Interactions")
    plt.tight_layout()
    return fig
```

### 6.4 Interpreting the Attention Map

The attention matrix reveals several types of relationships:

1. **Symmetric strong attention** (`A[i,j] ≈ A[j,i]` both high): Events i and j are **mutually dependent** — they jointly determine some power component. Example: instruction fetch rate and branch prediction accuracy are coupled through the pipeline.

2. **Asymmetric attention** (`A[i,j] >> A[j,i]`): Event i **depends on context from** event j, but not vice versa. This suggests a causal or hierarchical relationship.

3. **Cluster structure**: Groups of events with high mutual attention correspond to **functional units** or **pipeline stages** that operate together.

4. **CLS attention**: Which events the [CLS] token attends to most strongly are the **most power-relevant** events overall.

### 6.5 Validation of Discovered Interactions

To ensure attention weights reflect genuine interactions (not artifacts):

1. **Permutation test**: Randomly shuffle feature `j`, re-measure model error. If error increases when feature `j` was strongly attended-to by feature `i`, the interaction is real.

2. **Compare with Pearson/Spearman correlation**: Attention should capture interactions **beyond** simple linear correlation (the 16 correlation clusters we already know about[^2]).

3. **Domain validation**: Check if discovered interactions align with known CPU microarchitecture relationships (e.g., cache miss → memory access → store buffer pressure).

---

## 7. Concrete Implementation Blueprint

### 7.1 Complete PyTorch Implementation

```python
"""
FT-Transformer for Event-to-Event Attention Discovery
on ArchPower Event Dataset

Usage:
    python event_attention.py --arch BOOM --epochs 500 --d_token 32
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math


class NumericalFeatureTokenizer(nn.Module):
    """
    Embed each continuous feature independently into d-dimensional space.
    Each feature has its own linear projection (no weight sharing).
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        # Separate linear layer per feature
        self.weights = nn.Parameter(torch.randn(n_features, d_token))
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
        Returns:
            tokens: (batch, n_features, d_token)
        """
        # x[:, :, None] → (batch, n_features, 1)
        # self.weights[None] → (1, n_features, d_token)
        return x.unsqueeze(-1) * self.weights.unsqueeze(0) + self.biases.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""
    def __init__(self, d_token: int, n_heads: int, d_ffn: int, 
                 dropout: float = 0.3, attn_dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=attn_dropout, batch_first=True
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
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm, need_weights=need_weights
        )
        x = x + attn_out
        
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, attn_weights


class EventFTTransformer(nn.Module):
    """
    FT-Transformer for event-to-event interaction discovery.
    
    Input:  (batch, 81) continuous event features
    Output: (batch, 12) power predictions + attention maps
    """
    def __init__(self, n_features=81, n_targets=12, d_token=32,
                 n_blocks=2, n_heads=4, d_ffn=64,
                 dropout=0.3, attn_dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        nn.init.xavier_uniform_(self.cls_token)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout, attn_dropout)
            for _ in range(n_blocks)
        ])
        
        # Final layer norm + prediction head
        self.final_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_targets)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, 81) raw event features
            return_attention: if True, also return attention maps
        Returns:
            pred: (batch, 12) power predictions
            attn_maps: list of (batch, n_heads, 82, 82) if return_attention
        """
        batch_size = x.size(0)
        
        # Tokenize features → (batch, 81, d_token)
        tokens = self.tokenizer(x)
        
        # Prepend [CLS] → (batch, 82, d_token)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        
        # Pass through transformer blocks
        attn_maps = []
        for block in self.blocks:
            tokens, attn_w = block(tokens, need_weights=return_attention)
            if return_attention and attn_w is not None:
                attn_maps.append(attn_w)
        
        # Extract [CLS] token → prediction
        cls_output = self.final_norm(tokens[:, 0, :])
        pred = self.head(cls_output)
        
        if return_attention:
            return pred, attn_maps
        return pred


def prepare_data(arch='BOOM', val_ratio=0.2, seed=42):
    """Load and prepare the event dataset with z-score normalization."""
    base = 'dataset/event_dataset'
    feat = np.load(f'{base}/{arch}_event_feature.npy')
    label = np.load(f'{base}/{arch}_event_label.npy')
    
    # Z-score normalization (per feature)
    feat_mean = feat.mean(axis=0, keepdims=True)
    feat_std = feat.std(axis=0, keepdims=True) + 1e-8
    feat_norm = (feat - feat_mean) / feat_std
    
    # Also normalize labels for stable training
    label_mean = label.mean(axis=0, keepdims=True)
    label_std = label.std(axis=0, keepdims=True) + 1e-8
    label_norm = (label - label_mean) / label_std
    
    # Train/val split (by config group to avoid leakage)
    n_workloads = 8
    n_configs = feat.shape[0] // n_workloads
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_configs)
    n_val_configs = max(1, int(n_configs * val_ratio))
    val_configs = perm[:n_val_configs]
    train_configs = perm[n_val_configs:]
    
    train_idx = np.concatenate([np.arange(c*8, (c+1)*8) for c in train_configs])
    val_idx = np.concatenate([np.arange(c*8, (c+1)*8) for c in val_configs])
    
    return {
        'X_train': torch.FloatTensor(feat_norm[train_idx]),
        'y_train': torch.FloatTensor(label_norm[train_idx]),
        'X_val': torch.FloatTensor(feat_norm[val_idx]),
        'y_val': torch.FloatTensor(label_norm[val_idx]),
        'feat_mean': feat_mean, 'feat_std': feat_std,
        'label_mean': label_mean, 'label_std': label_std,
    }


def train_model(model, data, epochs=500, lr=1e-3, weight_decay=1e-3,
                patience=50, noise_std=0.02):
    """Train with early stopping, noise augmentation, and mixup."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        X = data['X_train']
        y = data['y_train']
        
        # Gaussian noise augmentation
        X_aug = X + torch.randn_like(X) * noise_std
        
        # Forward
        pred = model(X_aug)
        loss = F.mse_loss(pred, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(data['X_val'])
            val_loss = F.mse_loss(val_pred, data['y_val']).item()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")
    
    model.load_state_dict(best_state)
    return model


def extract_and_analyze_attention(model, X_all):
    """Extract attention maps and compute aggregated interaction matrices."""
    model.eval()
    with torch.no_grad():
        _, attn_maps = model(X_all, return_attention=True)
    
    results = {}
    for layer_idx, attn in enumerate(attn_maps):
        # attn shape: (N, n_heads, 82, 82)
        # Extract feature-to-feature block (skip CLS at index 0)
        feat_attn = attn[:, :, 1:, 1:]  # (N, n_heads, 81, 81)
        
        # Aggregate: mean over samples and heads
        results[f'layer_{layer_idx}_mean'] = feat_attn.mean(dim=[0, 1]).numpy()
        
        # Per-head: mean over samples
        for head in range(attn.size(1)):
            results[f'layer_{layer_idx}_head_{head}'] = feat_attn[:, head].mean(dim=0).numpy()
        
        # CLS attention to features
        cls_attn = attn[:, :, 0, 1:]  # (N, n_heads, 81)
        results[f'layer_{layer_idx}_cls_attn'] = cls_attn.mean(dim=[0, 1]).numpy()
    
    return results
```

### 7.2 Self-Supervised Pre-Training (Masked Feature Modeling)

```python
class MaskedFeaturePretrainer(nn.Module):
    """
    Pre-train FT-Transformer by reconstructing randomly masked features.
    This teaches the model event-to-event dependencies without labels.
    """
    def __init__(self, base_model: EventFTTransformer, mask_ratio=0.15):
        super().__init__()
        self.base = base_model
        self.mask_ratio = mask_ratio
        # Reconstruction head: from d_token back to scalar
        self.recon_head = nn.Linear(base_model.tokenizer.weights.size(1), 1)
    
    def forward(self, x):
        batch_size, n_feat = x.shape
        
        # Create mask
        mask = torch.rand(batch_size, n_feat) < self.mask_ratio
        
        # Replace masked features with 0 (or learnable mask token)
        x_masked = x.clone()
        x_masked[mask] = 0.0
        
        # Get token representations from transformer
        tokens = self.base.tokenizer(x_masked)
        cls = self.base.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        
        for block in self.base.blocks:
            tokens, _ = block(tokens)
        
        tokens = self.base.final_norm(tokens)
        
        # Reconstruct masked features
        feature_tokens = tokens[:, 1:, :]  # skip CLS
        reconstructed = self.recon_head(feature_tokens).squeeze(-1)  # (batch, 81)
        
        # Loss only on masked positions
        loss = F.mse_loss(reconstructed[mask], x[mask])
        return loss
```

### 7.3 Complete Training Pipeline

```python
def main():
    # 1. Prepare data
    data = prepare_data('BOOM', val_ratio=0.2, seed=42)
    X_all = torch.FloatTensor(np.load('dataset/event_dataset/BOOM_event_feature.npy'))
    # Normalize
    X_all = (X_all - torch.FloatTensor(data['feat_mean'])) / torch.FloatTensor(data['feat_std'])
    
    # 2. Build model
    model = EventFTTransformer(
        n_features=81, n_targets=12,
        d_token=32, n_blocks=2, n_heads=4, d_ffn=64,
        dropout=0.3, attn_dropout=0.2
    )
    
    # 3. (Optional) Self-supervised pre-training
    pretrainer = MaskedFeaturePretrainer(model, mask_ratio=0.15)
    pretrain_optimizer = torch.optim.AdamW(pretrainer.parameters(), lr=1e-3)
    for epoch in range(200):
        pretrainer.train()
        loss = pretrainer(X_all)
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()
    
    # 4. Supervised fine-tuning
    model = train_model(model, data, epochs=500, lr=1e-3)
    
    # 5. Extract attention maps
    attention_results = extract_and_analyze_attention(model, X_all)
    
    # 6. Visualize
    mean_attn = attention_results['layer_1_mean']
    plot_attention_heatmap(mean_attn, title="BOOM Event-to-Event Attention (Layer 2, Mean)")
    
    # 7. Find top interactions
    plot_top_interactions(mean_attn, k=20)
    
    # 8. Save results
    np.save('attention_matrix_boom.npy', mean_attn)
    print("Done! Attention matrix saved.")
```

---

## 8. Alternative Approaches

### 8.1 Correlation-Aware Attention (No Deep Learning)

For a simpler baseline, compute a **mutual information** or **partial correlation** matrix directly:

```python
from sklearn.feature_selection import mutual_info_regression

# Mutual information between each feature pair
MI = np.zeros((81, 81))
for i in range(81):
    MI[i, :] = mutual_info_regression(X, X[:, i])
```

This gives "which features predict which other features" without neural networks. Compare with learned attention to validate.

### 8.2 Graph Attention Network (GAT)

Treat features as nodes in a graph, with edges representing interactions:

```
Feature nodes → GAT layers → Learned edge weights = interactions
```

The edge attention weights in GAT serve a similar role to self-attention in transformers[^9]. However, GAT requires a pre-defined graph topology (which features can interact), whereas self-attention considers all pairs.

### 8.3 Additive Attention / Bilinear Attention

Instead of full scaled dot-product attention, use simpler attention variants that are less parameter-hungry:

```python
# Bilinear attention: A[i,j] = t_i^T W t_j
class BilinearAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d, d))
    
    def forward(self, tokens):
        # tokens: (batch, n_feat, d)
        A = torch.matmul(tokens @ self.W, tokens.transpose(-1, -2))
        A = F.softmax(A / math.sqrt(tokens.size(-1)), dim=-1)
        return A
```

This has fewer parameters (only d² for the bilinear weight) and may be more appropriate for very small datasets.

---

## 9. Experimental Design

### 9.1 Evaluation Protocol

Since the goal is **interpretability** (discovering interactions), not just prediction accuracy:

| Metric | Purpose |
|---|---|
| **Validation MSE** | Ensure model actually learns meaningful predictions |
| **Attention stability** | Run with 5 random seeds; compute std of attention matrices |
| **Attention vs. correlation** | Compare discovered interactions with known correlation clusters[^2] |
| **Permutation importance** | For top-attended feature pairs, verify via permutation tests |
| **Domain validation** | Map discovered interactions to CPU microarchitecture knowledge |

### 9.2 Cross-Validation Strategy

Follow the ArchPower convention of **config-level splits**[^10]:

```
BOOM: 15 configs → 5-fold CV (3 configs per fold = 24 samples)
XS:   10 configs → 5-fold CV (2 configs per fold = 16 samples)
```

Report mean ± std of both prediction error and attention map stability.

### 9.3 Baselines to Compare Against

1. **Pearson correlation matrix** (linear interactions only)
2. **Mutual information matrix** (nonlinear but pairwise)
3. **XGBoost SHAP interaction values** (tree-based feature interactions)
4. **Random attention** (sanity check: does learned attention differ from uniform?)

### 9.4 Expected Outcomes

Based on the correlation analysis[^2][^3]:

- **Instruction pipeline cluster** (features 0, 12, 24, 25, 30, 36, 55, 77, 79): Strong mutual attention expected — these are IPC/instruction-count related events
- **Memory subsystem cluster** (features 42, 43, 10, 68, 23): Should attend to each other — cache miss/hit rates
- **Branch-related features** (features 3, 62, 63, 64): Branch misprediction events should form a tight cluster
- **Cross-cluster attention**: The interesting discovery will be **between-cluster** interactions not captured by simple correlation — e.g., how branch mispredictions interact with cache behavior to affect power

---

## 10. Confidence Assessment

### High Confidence
- **FT-Transformer is well-suited** for this feature-interaction discovery task — it's the standard approach for all-continuous tabular data with attention[^4]
- **Normalization is essential** given the 9-order-of-magnitude scale difference[^1]
- **Config-level train/test splits** are necessary to avoid information leakage[^10]

### Medium Confidence
- **The model will train successfully** on 120 samples with the recommended hyperparameters — this is at the edge of what transformers can handle, but the regularization + pre-training strategy should work
- **Attention weights will be interpretable** — attention-as-explanation is debated in NLP, but for tabular data where features have clear semantics, it's more justified[^11]
- **Multi-target training** (predicting all 12 components) will help vs. single-target — provides implicit regularization through shared representation

### Lower Confidence
- **Whether attention captures interactions beyond correlation** — with only 8 unique workload patterns, the model may primarily rediscover the known correlation structure rather than novel higher-order interactions
- **Optimal hyperparameters** — the recommended values are educated guesses; systematic tuning (e.g., with Optuna) is needed
- **Cross-architecture generalization** — whether BOOM and XS attention patterns are comparable depends on how similarly the event features map between the two architectures

### Assumptions Made
1. The 81 event features have consistent semantics between BOOM and XS (same column indices mean the same physical event)
2. PyTorch is available or can be installed in the environment
3. The primary goal is interaction **discovery** (interpretability), not pure prediction accuracy

---

## 11. Footnotes

[^1]: `dataset/event_dataset/BOOM_event_feature.npy` — computed `std` range [0.0001, 93544.49], ratio ~824M. See data analysis in this report.

[^2]: Feature correlation analysis on BOOM event features: 312 pairs with |corr|>0.8, 16 clusters with |corr|>0.9. Largest cluster: features {8, 31, 32, 33, 35, 39, 40, 51, 78, 80} (10 members).

[^3]: Feature-label correlation analysis showing component-specific top features. E.g., ISU: f77(+0.873), DCache: f42(+0.654), FU-Pool: f56(+0.886).

[^4]: Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). "Revisiting Deep Learning Models for Tabular Data." NeurIPS 2021. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959). Official code: [yandex-research/rtdl](https://github.com/yandex-research/rtdl).

[^5]: Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training." [arXiv:2106.01342](https://arxiv.org/abs/2106.01342). Code: [somepago/saint](https://github.com/somepago/saint).

[^6]: Arik, S. Ö., & Pfister, T. (2021). "TabNet: Attentive Interpretable Tabular Learning." AAAI 2021.

[^7]: Regularization techniques for attention models on small data: AttentionDrop ([arXiv:2504.12088](https://arxiv.org/abs/2504.12088)), TANGOS gradient orthogonalization ([OpenReview](https://openreview.net/forum?id=n6H86gW8u0d)), structured sparsity ([arXiv:2508.06016](https://arxiv.org/html/2508.06016v1)).

[^8]: Chen, Z., et al. (2025). "ExcelFormer: Making Neural Network Excel in Small Tabular Data." Uses semi-permeable attention for small datasets. [OpenReview](https://openreview.net/forum?id=sYv3OMboTF).

[^9]: "The Role of Feature Interactions in Graph-based Tabular Deep Learning." [arXiv:2510.04543](https://arxiv.org/html/2510.04543v2).

[^10]: Train/test split convention from ArchPower: split by CPU configuration, not individual samples. Each config expands to 8 samples. See `src/McPAT-Calib-CompGroup.py` and `doc/event_dataset_generation.md`.

[^11]: Attention-as-explanation is more justified in tabular settings than NLP because: (a) each token represents a single, interpretable feature, (b) there's no positional encoding ambiguity, and (c) the attention pattern directly corresponds to feature interaction.

---

## Key References

| Resource | Type | URL |
|---|---|---|
| FT-Transformer (Gorishniy 2021) | Paper | [arXiv:2106.11959](https://arxiv.org/abs/2106.11959) |
| Official RTDL code | Code | [github.com/yandex-research/rtdl](https://github.com/yandex-research/rtdl) |
| SAINT (Somepalli 2021) | Paper | [arXiv:2106.01342](https://arxiv.org/abs/2106.01342) |
| TabNet (Arik 2021) | Paper | [AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16826) |
| pytorch-tabular library | Library | [github.com/manujosephv/pytorch-tabular](https://github.com/manujosephv/pytorch-tabular) |
| ExcelFormer (small data) | Paper | [OpenReview](https://openreview.net/forum?id=sYv3OMboTF) |
| CAPSim (attention for CPU) | Paper | [arXiv:2510.10484](https://arxiv.org/html/2510.10484) |
| BertViz (attention visualization) | Tool | [github.com/jessevig/bertviz](https://github.com/jessevig/bertviz) |
| ArchPower dataset | Dataset | [github (this repo)](../README.md) |
| Event dataset generation | Doc | [`doc/event_dataset_generation.md`](../doc/event_dataset_generation.md) |
