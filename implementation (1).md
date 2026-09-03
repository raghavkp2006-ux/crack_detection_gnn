# Pipeline Crack Detection: Shallow GNN + Transformer Hybrid
## Implementation Plan

---

## 1. Project Overview

**Goal:** Detect cracks on pipeline surface images using a superpixel-graph representation, where graph edges are constructed via **Pixel Intensity Resemblance Method (PIRM)**. We improve on a deep-GNN baseline by combining a **shallow GNN** (to avoid over-smoothing) with a **positionally-encoded Transformer layer** (to recover long-range spatial continuity that PIRM edges cannot express).

**Core hypothesis:**
PIRM graphs connect superpixels by appearance similarity only, with no spatial/geometric awareness. This causes two failure modes in deep-GNN baselines:
1. **Fragmented detection** of long/diagonal cracks whose superpixels are far apart in graph-hop-distance.
2. **False positives** from intensity-similar-but-unrelated regions (shadows, stains, weld seams).

A shallow GNN (1–3 layers) avoids over-smoothing while capturing local appearance consistency. A Transformer layer with explicit positional encoding of superpixel centroids recovers global spatial reasoning that PIRM cannot provide, letting far-apart same-crack superpixels reinforce each other while down-weighting spatially-incoherent false positives.

**Task formulation:** Node-level binary classification (crack / no-crack) per superpixel → reconstructed into a full-image crack segmentation mask.

---

## 2. Repository / Folder Structure

```
pipeline-crack-gnn/
├── data/
│   ├── raw/                     # original pipe surface images
│   ├── masks/                   # ground-truth crack masks (if available)
│   ├── superpixels/             # cached SLIC segmentation outputs (.npy)
│   └── graphs/                  # cached PyG graph objects (.pt)
├── src/
│   ├── preprocessing/
│   │   ├── superpixel.py        # SLIC segmentation
│   │   ├── pirm_graph.py        # PIRM edge construction
│   │   └── features.py          # node feature extraction
│   ├── models/
│   │   ├── gnn_encoder.py       # shallow/deep GNN encoder
│   │   ├── positional_encoding.py
│   │   ├── transformer_block.py
│   │   └── hybrid_model.py      # full GNN + Transformer model
│   ├── train.py
│   ├── evaluate.py
│   ├── ablation.py
│   └── utils/
│       ├── metrics.py           # F1, IoU, Dirichlet energy
│       └── visualize.py
├── configs/
│   ├── baseline_deep_gnn.yaml
│   ├── shallow_gnn.yaml
│   ├── hybrid_no_pe.yaml
│   └── hybrid_full.yaml
├── notebooks/
│   └── exploratory.ipynb
├── implementation.md            # this file
└── requirements.txt
```

---

## 3. Environment Setup (Phase 0)

### Commands

```bash
python -m venv venv
source venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric
pip install scikit-image scikit-learn opencv-python
pip install numpy pandas matplotlib seaborn
pip install pyyaml tqdm
```

### `requirements.txt`

```
torch>=2.1.0
torch_geometric>=2.4.0
scikit-image>=0.22
scikit-learn>=1.3
opencv-python>=4.8
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
pyyaml>=6.0
tqdm>=4.66
```

---

## 4. Phase 1 — Data Preprocessing & Graph Construction

### 4.1 Superpixel Segmentation

Use SLIC (Simple Linear Iterative Clustering) via `skimage.segmentation.slic`.

**Parameters:**
| Parameter | Value | Notes |
|---|---|---|
| `n_segments` | 200–500 | tune based on image resolution; more segments = finer crack localization but larger graph |
| `compactness` | 10.0 | balances color proximity vs. spatial proximity |
| `sigma` | 1.0 | Gaussian smoothing before segmentation |
| `start_label` | 1 | |

```python
from skimage.segmentation import slic
from skimage.io import imread

image = imread(image_path)
segments = slic(image, n_segments=300, compactness=10.0, sigma=1.0, start_label=1)
```

Cache `segments` array per image to `data/superpixels/{image_id}.npy`.

### 4.2 Node Feature Extraction

Per superpixel, compute:
| Feature | Dimension | Description |
|---|---|---|
| Mean intensity (per channel, or grayscale) | 1–3 | primary PIRM feature |
| Std. dev. of intensity | 1 | texture roughness proxy |
| Mean gradient magnitude (Sobel) | 1 | edge strength — cracks have sharp gradients |
| LBP histogram (Local Binary Pattern) | 8–16 | texture descriptor |
| Centroid (x, y), normalized to [0,1] | 2 | **stored separately** — used later for positional encoding, NOT concatenated into GNN input features (to avoid the GNN implicitly learning spatial shortcuts that bypass the PIRM graph structure) |

**Total node feature dimension: `d_in = 12–20`** depending on which are included (finalize after ablation).

```python
def extract_node_features(image, segments):
    features = []
    centroids = []
    for label in np.unique(segments):
        mask = segments == label
        pixels = image[mask]
        mean_int = pixels.mean(axis=0)
        std_int = pixels.std(axis=0)
        grad = sobel_magnitude(image)[mask].mean()
        lbp_hist = compute_lbp_histogram(image, mask)
        ys, xs = np.where(mask)
        centroid = (xs.mean() / image.shape[1], ys.mean() / image.shape[0])
        features.append(np.concatenate([mean_int, std_int, [grad], lbp_hist]))
        centroids.append(centroid)
    return np.array(features), np.array(centroids)
```

### 4.3 PIRM Edge Construction

**Definition:** connect superpixels `i, j` if their intensity resemblance exceeds a threshold, or weight all pairs by a similarity kernel.

**Recommended formulation (weighted k-NN over intensity space, not full pairwise O(n²)):**

```python
from sklearn.neighbors import NearestNeighbors

def build_pirm_graph(intensity_features, k=8, sigma=0.1):
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(intensity_features)
    dists, idxs = nn.kneighbors(intensity_features)
    edge_index = []
    edge_weight = []
    for i in range(len(intensity_features)):
        for j_pos in range(1, k+1):  # skip self at position 0
            j = idxs[i, j_pos]
            d = dists[i, j_pos]
            w = np.exp(-(d**2) / (2 * sigma**2))   # Gaussian similarity kernel
            edge_index.append([i, j])
            edge_weight.append(w)
    return np.array(edge_index).T, np.array(edge_weight)
```

**Parameters:**
| Parameter | Value | Notes |
|---|---|---|
| `k` (neighbors per node) | 8 | tune 5–15; too high densifies graph and re-introduces over-smoothing risk |
| `sigma` (kernel bandwidth) | 0.1 (on normalized intensity scale) | controls how sharply similarity decays |
| Graph type | k-NN (directed → symmetrize) | full O(n²) PIRM is only feasible for n < ~2000 nodes; use k-NN for larger images |

Symmetrize: `edge_index = union(edge_index, edge_index.flip(0))`, average duplicate weights.

Save as PyTorch Geometric `Data` object:

```python
from torch_geometric.data import Data
import torch

data = Data(
    x=torch.tensor(node_features, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_weight, dtype=torch.float).unsqueeze(-1),
    pos=torch.tensor(centroids, dtype=torch.float),   # kept separately for positional encoding
    y=torch.tensor(node_labels, dtype=torch.long)      # crack / no-crack per superpixel
)
torch.save(data, f"data/graphs/{image_id}.pt")
```

---

## 5. Phase 2 — Baseline: Deep GNN (Reference Model)

Purpose: establish the over-smoothing problem empirically before proposing the fix.

**Architecture:**
| Parameter | Value |
|---|---|
| Layer type | GraphSAGE (`SAGEConv`) or GAT (`GATConv`) — pick one consistent with your prior baseline |
| Number of layers | 8 (deliberately deep, to demonstrate degradation) |
| Hidden dim | 64 |
| Activation | ReLU |
| Normalization | BatchNorm after each layer |
| Dropout | 0.3 |
| Residual connections | None (to let over-smoothing manifest clearly) |
| Output head | Linear(64 → 2) + softmax |

```python
import torch.nn as nn
from torch_geometric.nn import SAGEConv, BatchNorm

class DeepGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=8, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(BatchNorm(dims[i+1]))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        return self.head(x)
```

**Training config:**
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| LR | 1e-3 |
| Weight decay | 5e-4 |
| LR scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Loss | Weighted CrossEntropyLoss (class weights for crack/no-crack imbalance) |
| Epochs | 150 |
| Batch size | 1 image-graph per step, or batch multiple small graphs via `torch_geometric.loader.DataLoader` |
| Early stopping | patience=20 on val F1 |

**Required measurement:** Dirichlet energy per layer, logged at each epoch (see Section 8).

---

## 6. Phase 3 — Shallow GNN (Novelty Axis 1)

Identical architecture to Phase 2, with:
| Parameter | Value | Change from baseline |
|---|---|---|
| Number of layers | 2 (test 1, 2, 3 in ablation) | reduced from 8 |
| Residual/skip connection | Add `x = x + prev_x` after layer 2 if using ≥2 layers | mitigates residual info loss |
| Everything else | same as Phase 2 | |

This is trained and evaluated standalone to isolate the effect of depth reduction alone, before adding the transformer.

---

## 7. Phase 4 — Hybrid Model: Shallow GNN + Transformer (Full Proposed Model)

### 7.1 Positional Encoding

Use sinusoidal positional encoding applied to normalized centroid coordinates `(x, y)` stored in `data.pos`.

**Parameters:**
| Parameter | Value |
|---|---|
| PE dimension | 32 (16 for x, 16 for y, concatenated) |
| Frequency bands | 8 per axis |
| Combination with GNN embedding | Concatenate, then project via Linear(hidden_dim + 32 → hidden_dim) |

```python
import torch, math

def sinusoidal_pe(coords, num_freqs=8):
    # coords: [N, 2] normalized to [0,1]
    freqs = 2 ** torch.arange(num_freqs).float() * math.pi
    args = coords.unsqueeze(-1) * freqs  # [N, 2, num_freqs]
    pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [N, 2, 2*num_freqs]
    return pe.flatten(1)  # [N, 2*2*num_freqs] = [N, 32] when num_freqs=8
```

### 7.2 Transformer Block

**Parameters:**
| Parameter | Value | Notes |
|---|---|---|
| Number of transformer layers | 2 | keep shallow initially; ablate 1–4 |
| Attention type | Standard multi-head self-attention (full, not causal) | all superpixels attend to all others |
| Number of heads | 4 | hidden_dim=64 → 16 dim/head |
| Hidden dim | 64 (matches GNN output) | |
| FFN expansion | 4x (64 → 256 → 64) | standard transformer FFN ratio |
| Activation (FFN) | GELU | |
| Dropout | 0.1 | attention dropout + FFN dropout |
| Normalization | Pre-LN (LayerNorm before attention/FFN, not post) | more stable training for shallow transformer stacks |
| Positional bias option (advanced, optional Phase 4b) | Add relative distance between centroids as an additive bias term to attention logits, rather than only concatenating PE to input | stronger structural grounding; implement only after basic PE version works |

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim=64, heads=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [N, dim] -> treat as [1, N, dim] batch of 1 image-graph
        x_n = self.norm1(x).unsqueeze(0)
        attn_out, _ = self.attn(x_n, x_n, x_n)
        x = x + self.dropout(attn_out.squeeze(0))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

### 7.3 Full Hybrid Model

```python
class HybridGNNTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, gnn_layers=2, transformer_layers=2,
                 heads=4, pe_dim=32, dropout=0.1):
        super().__init__()
        # Shallow GNN encoder
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * gnn_layers
        for i in range(gnn_layers):
            self.convs.append(SAGEConv(dims[i], dims[i+1]))
            self.norms.append(BatchNorm(dims[i+1]))
        self.gnn_dropout = nn.Dropout(dropout)

        # Positional encoding fusion
        self.pe_proj = nn.Linear(hidden_dim + pe_dim, hidden_dim)

        # Transformer stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dropout=dropout)
            for _ in range(transformer_layers)
        ])

        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, pos):
        for conv, norm in zip(self.convs, self.norms):
            x = torch.relu(norm(conv(x, edge_index)))
            x = self.gnn_dropout(x)

        pe = sinusoidal_pe(pos)                       # [N, pe_dim]
        x = torch.cat([x, pe], dim=-1)
        x = self.pe_proj(x)                            # [N, hidden_dim]

        for block in self.transformer_blocks:
            x = block(x)

        return self.head(x)
```

**Training config:** same as Phase 2/3 (Adam, lr=1e-3, weight_decay=5e-4), but:
| Parameter | Value | Notes |
|---|---|---|
| LR warmup | 5 epochs linear warmup before ReduceLROnPlateau | transformers benefit from warmup even when small |
| Gradient clipping | max_norm=1.0 | stabilizes attention training |

---

## 8. Phase 5 — Evaluation & Ablation

### 8.1 Model Variants to Train (in order)

| ID | Model | Purpose |
|---|---|---|
| M1 | Deep GNN (8 layers), no transformer | baseline, shows over-smoothing |
| M2 | Shallow GNN (2 layers), no transformer | isolates depth-reduction effect |
| M3 | Shallow GNN (2 layers) + Transformer, **no positional encoding** | isolates raw attention effect |
| M4 | Shallow GNN (2 layers) + Transformer + **positional encoding** (full proposed model) | full novelty contribution |

### 8.2 Metrics

**Task performance (per-node classification → aggregate to image-level segmentation):**
- Precision, Recall, F1 (crack class)
- IoU (Intersection-over-Union) on reconstructed mask
- ROC-AUC

**Over-smoothing diagnostic (compute at each GNN layer output, for M1 vs M2):**

```python
def dirichlet_energy(x, edge_index):
    row, col = edge_index
    diff = x[row] - x[col]
    energy = (diff ** 2).sum(dim=-1).mean()
    return energy.item()

def avg_pairwise_cosine_sim(x, sample_size=1000):
    idx = torch.randperm(x.size(0))[:sample_size]
    xs = torch.nn.functional.normalize(x[idx], dim=-1)
    sim = xs @ xs.T
    return sim.mean().item()
```

Log both metrics per GNN layer, per epoch. Plot layer-index vs. energy for M1 (expect sharp decay) vs. M2 (expect energy stays higher).

### 8.3 Bucketed Analysis (key differentiating experiment)

Split test-set cracks by geometric property:
| Bucket | Criterion |
|---|---|
| Short/compact | crack bounding-box diagonal < median |
| Long/diagonal | crack bounding-box diagonal ≥ median, aspect ratio > 3:1 |
| Lighting-variable | cracks with intensity std along their length > threshold (compute from ground truth mask overlaid on image) |

Report F1/IoU separately per bucket for M1–M4. **Expected result supporting your hypothesis:** M4 > M3 > M2 ≈ M1 on "long/diagonal" and "lighting-variable" buckets; M2/M3/M4 roughly similar to M1 on "short/compact" bucket.

### 8.4 Commands

```bash
# Train each variant
python src/train.py --config configs/baseline_deep_gnn.yaml
python src/train.py --config configs/shallow_gnn.yaml
python src/train.py --config configs/hybrid_no_pe.yaml
python src/train.py --config configs/hybrid_full.yaml

# Run full ablation + bucketed analysis + plots
python src/ablation.py --models M1 M2 M3 M4 --output results/ablation_report/
```

---

## 9. Phase 6 — Writeup Structure

1. **Introduction:** problem statement, PIRM graph limitation, over-smoothing background.
2. **Related Work:** GNNs for defect detection, graph transformers (Graphormer, GraphGPS), over-smoothing literature.
3. **Method:**
   - PIRM graph construction (Section 4.3)
   - Shallow GNN encoder (Section 6)
   - Positional encoding + Transformer fusion (Section 7)
4. **Experiments:**
   - Dataset description, superpixel/graph stats (avg nodes/edges per image)
   - M1–M4 comparison table (Section 8.2 metrics)
   - Dirichlet energy plots (Section 8.2)
   - Bucketed analysis table (Section 8.3) — **this is your strongest evidence**
5. **Discussion:** where the model still fails (e.g., very short cracks, extreme lighting), compute cost comparison (M1 vs M4 training/inference time).
6. **Conclusion & Future Work:** relative position bias in attention (Section 7.2 advanced option), scaling to full pairwise PIRM instead of k-NN.

---

## 10. Full Hyperparameter Summary Table

| Component | Parameter | Value |
|---|---|---|
| SLIC | n_segments | 300 |
| SLIC | compactness | 10.0 |
| PIRM | k (neighbors) | 8 |
| PIRM | sigma (kernel bandwidth) | 0.1 |
| GNN (baseline) | layers | 8 |
| GNN (shallow) | layers | 2 |
| GNN | hidden_dim | 64 |
| GNN | dropout | 0.3 |
| Transformer | layers | 2 |
| Transformer | heads | 4 |
| Transformer | FFN expansion | 4x |
| Transformer | dropout | 0.1 |
| Positional encoding | dim | 32 |
| Positional encoding | freq bands | 8 |
| Optimizer | Adam, lr | 1e-3 |
| Optimizer | weight_decay | 5e-4 |
| Training | epochs | 150 |
| Training | early stopping patience | 20 |
| Transformer-specific | LR warmup epochs | 5 |
| Transformer-specific | grad clip norm | 1.0 |

---

## 11. Suggested Timeline

| Phase | Task | Estimated time |
|---|---|---|
| 0 | Environment setup | 0.5 day |
| 1 | Superpixel + PIRM graph pipeline | 2–3 days |
| 2 | Deep GNN baseline + Dirichlet energy logging | 2 days |
| 3 | Shallow GNN variant | 1 day |
| 4 | Transformer + positional encoding implementation | 3–4 days |
| 5 | Full ablation + bucketed analysis | 2–3 days |
| 6 | Writeup | 3–4 days |
