# Pipeline Crack Detection using Graph Neural Networks and Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A framework for crack segmentation using SLIC superpixels, feature-space kNN graphs, GraphSAGE encoders, positional encoding and Transformer attention.

---

## 📌 Table of Contents
- [Overview & Motivation](#-overview--motivation)
- [System Architecture](#-system-architecture)
- [Model Variants & Taxonomy](#-model-variants--taxonomy)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Workflow & Usage](#-workflow--usage)
  - [1. Running Smoke Tests](#1-running-smoke-tests)
  - [2. Preprocessing & Graph Construction](#2-preprocessing--graph-construction)
  - [3. Training](#3-training)
  - [4. Evaluation & Mask Reconstruction](#4-evaluation--mask-reconstruction)
  - [5. Systematic Ablation Study](#5-systematic-ablation-study)
- [Evaluation Metrics & Diagnostics](#-evaluation-metrics--diagnostics)
- [Configuration Reference](#-configuration-reference)

---

## 💡 Overview & Motivation

Detecting cracks in civil infrastructure, industrial pipes, and concrete surfaces via standard Convolutional Neural Networks (CNNs) faces several key limitations:
1. **Severe Pixel Imbalance:** Cracks represent tiny fractions of the overall surface area.
2. **Tortuous and Non-Local Topologies:** Cracks exhibit thin, winding geometries spanning long spatial distances. Standard CNN receptive fields struggle to capture continuity without downsampling artifacts.
3. **Over-smoothing in Deep GNNs:** Directly stacking deep GNN layers causes representations to collapse into indistinguishable states (high Dirichlet energy loss).

### Our Solution
This framework formulates crack detection as a **graph node classification problem on superpixel regions**:
- **Graph construction:** SLIC superpixel nodes connected by k=8 nearest neighbors on raw 15-D features.
- **Hybrid GNN-Transformer Architecture:** 
  - A **Shallow GNN** captures fine-grained local connectivity without suffering from over-smoothing.
  - **2D Fourier/Sinusoidal Positional Encoding** injects metric spatial coordinates to break graph permutation invariance.
  - **Multi-Head Transformer Self-Attention** enables long-range reasoning to reconnect broken or distant crack branches.

---

## 🏗 System Architecture

Each image is segmented with SLIC (target 300 superpixels, compactness 10,
sigma 1, connectivity enforcement disabled). Nodes follow `np.unique(segments)`.
The raw 15-D feature vector contains three mean RGB channels, intensity standard
deviation, mean Sobel magnitude, and ten uniform-LBP bins. Normalized centroid
coordinates are stored separately as `pos`; they are not part of the 15-D features.

A k=8 nearest-neighbor graph uses Euclidean distance on the raw, unnormalized
15-D features, then adds reverse edges. Gaussian edge weights are computed and
stored but are unused by the models. `pirm_graph.py` is a legacy filename for
this feature-space kNN builder.

The graph encoders use GraphSAGE (`SAGEConv`), BatchNorm, ReLU and dropout 0.3.
M1 has eight layers without residual connections; M2 has two layers with a
residual connection. M3/M4 use the two-layer encoder followed by two Transformer
blocks (four heads, width 64, FFN width 256, dropout 0.1). M4 concatenates 32-D
sinusoidal centroid encodings and projects them to width 64 before attention.
All variants use a single linear 64→2 classification head. Node probabilities
are projected to the original image through its superpixel map.

```text
RGB image → SLIC → raw 15-D features → symmetrized feature kNN graph
                                            ↓
                                     GraphSAGE encoder
                                            ↓
                            optional centroid PE + Transformer
                                            ↓
                                   linear node classifier
                                            ↓
                              full-resolution superpixel mask
```

Training uses one graph per batch, Adam (lr 0.001, weight decay 5e-4), at most
150 epochs and early-stopping patience 20. For each seed (42, 123, 456, 789,
2026), `get_splits(dataset, 0.7, 0.15, seed)` selects 210 training and 45 validation
graphs from the 300-image train directory; the remaining 45 are unused by the
paper's holdout evaluation. The independent test directory has 237 images.
Validation nodes alone select τ* from 0.10 to 0.90 in steps of 0.02.

Paper v2's primary comparison uses focal loss (gamma 2) for M1–M4, plus
DeepCrack-style (re-implemented). The full weighted-CE/focal grid is secondary.
Correct graph parameter counts: M1 60,930; M2 10,626; M3 110,594; M4 116,802.
See the [paper v2 workflow](paper_v2/README.md) for reproducible run order.

---

## 🔬 Model Variants & Taxonomy

The codebase supports four systematic configurations for comparative evaluation and ablation:

| Model ID | Architecture | GNN Depth | Transformer | Positional Encoding | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **M1** | `baseline_deep_gnn` | 8 Layers | ❌ None | ❌ None | Quantifies the over-smoothing bottleneck in deep GNNs. |
| **M2** | `shallow_gnn` | 2 Layers | ❌ None | ❌ None | Evaluates pure local message-passing without global attention. |
| **M3** | `hybrid_no_pe` | 2 Layers | 2 Layers | ❌ None | Tests whether global attention alone suffices without spatial position awareness. |
| **M4** | `hybrid_full` | 2 Layers | 2 Layers | ✅ 2D Sinusoidal | Full hybrid model combining local topology, global attention, and geometric coordinates. |

---

## 📂 Repository Structure

```text
pipeline-crack-gnn/
│
├── configs/                           # Experiment YAML configurations
│   ├── baseline_deep_gnn.yaml         # M1: 8-layer deep GraphSAGE
│   ├── shallow_gnn.yaml               # M2: 2-layer shallow GraphSAGE
│   ├── hybrid_no_pe.yaml              # M3: GNN + Transformer (no PE)
│   └── hybrid_full.yaml               # M4: GNN + PE + Transformer (Full)
│
├── src/
│   ├── models/                        # Neural network architectures
│   │   ├── gnn_encoder.py             # DeepGNN and ShallowGNN implementations
│   │   ├── positional_encoding.py     # 2D sinusoidal coordinate PE & fusion
│   │   ├── transformer_block.py       # Multi-head attention blocks & stacks
│   │   └── hybrid_model.py            # HybridGNNTransformer & build_model factory
│   │
│   ├── preprocessing/                 # Graph generation & feature pipeline
│   │   ├── superpixel.py              # SLIC superpixel segmentation & caching
│   │   ├── features.py                # 15D node features (Color, Sobel, LBP)
│   │   ├── pirm_graph.py              # Feature kNN graph construction & PyG data builder
│   │   └── build_dataset.py           # Batch preprocessing CLI runner
│   │
│   ├── utils/                         # Helper functions & utilities
│   │   ├── dataset.py                 # PyG dataset loader and train/val/test splits
│   │   ├── metrics.py                 # F1, IoU, ROC-AUC, Dirichlet energy, Cosine Sim
│   │   └── visualize.py               # Visualizer for superpixels, graphs & predictions
│   │
│   ├── train.py                       # Main training script with early stopping
│   ├── evaluate.py                    # Standalone test set evaluation & mask recovery
│   └── ablation.py                    # Multi-model ablation pipeline & bucket analysis
│
├── tests/
│   └── smoke_test.py                  # End-to-end integration test with synthetic data
│
└── requirements.txt                   # Dependency definitions
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/raghavkp2006-ux/crack_detection_gnn.git
cd crack_detection_gnn/pipeline-crack-gnn
```

### 2. Create and Activate Virtual Environment
```bash
# Using conda
conda create -n crack-gnn python=3.11 -y
conda activate crack-gnn

# Or using venv
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Workflow & Usage

### 1. Running Smoke Tests
Verify that all dependencies, model architectures, feature extractions, and forward passes operate seamlessly:
```bash
python -m tests.smoke_test
```

### 2. Preprocessing & Graph Construction
Converts a folder of raw images and binary masks into PyTorch Geometric graph structures:
```bash
python -m src.preprocessing.build_dataset \
    --raw_dir data/raw_images \
    --mask_dir data/masks \
    --superpixel_dir data/superpixels \
    --graph_dir data/graphs \
    --config configs/hybrid_full.yaml
```

### 3. Training
Train any model variant using its configuration file:
```bash
# Train the Full Hybrid Model (M4)
python -m src.train --config configs/hybrid_full.yaml

# Train the Baseline Deep GNN (M1)
python -m src.train --config configs/baseline_deep_gnn.yaml
```

### 4. Evaluation & Mask Reconstruction
Evaluate a trained checkpoint against test graphs, generate classification metrics, and project graph node predictions back into pixel masks:
```bash
python -m src.evaluate \
    --model_path results/M4_hybrid_full/best_model.pt \
    --config_path configs/hybrid_full.yaml \
    --test_graph_dir data/graphs \
    --superpixel_dir data/superpixels \
    --output_dir results/predictions
```

### 5. Systematic Ablation Study
Run an automated benchmark across all four variants (M1 to M4) and generate comparison tables across crack geometry buckets:
```bash
python -m src.ablation \
    --configs_dir configs \
    --output_dir results/ablation
```

---

## 📊 Evaluation Metrics & Diagnostics

The evaluation engine computes both standard predictive performance and mathematical diagnostics for representation health:

### Predictive Metrics
- **F1-Score (Macro & Binary):** Evaluates precision and recall under high class imbalance.
- **Intersection over Union (IoU / Jaccard Index):** Quantifies overlap of predicted crack segments with ground truth.
- **ROC-AUC:** Area under the ROC curve for probabilistic node classification.

### Over-smoothing Diagnostics
- **Dirichlet Energy:**
  $$E(X) = \frac{1}{2} \text{Tr}(X^T L X) = \frac{1}{2} \sum_{(i, j) \in E} A_{ij} \|x_i - x_j\|^2$$
  Monitors whether node features collapse toward a constant value as GNN depth increases.
- **Average Pairwise Cosine Distance:**
  Tracks the geometric separation between node embeddings in latent space.

### Geometric Bucket Analysis

Buckets use the white crack pixels of the original test mask, evaluated in order:
1. **Thin:** crack area fraction <0.025.
2. **Long:** otherwise, bounding-box aspect ratio max(h,w)/(min(h,w)+1e-5) ≥2.
3. **Branched:** all remaining nonempty masks.

Empty masks are recorded separately as Clean. Scores use node labels; bootstrap
intervals resample whole images. These are geometry buckets, not lighting classes.

---

## 🔧 Configuration Reference

Configuration keys match `configs/hybrid_full.yaml` and the model factory:

```yaml
model:
  type: hybrid
  in_dim: 15
  hidden_dim: 64
  gnn_layers: 2
  transformer_layers: 2
  heads: 4
  pe_dim: 32
  num_freqs: 8
  gnn_dropout: 0.3       # Factory currently uses model.dropout, default 0.3.
  transformer_dropout: 0.1
  use_pe: true
superpixels:
  n_segments: 300
  compactness: 10.0
  sigma: 1.0
graph:
  k: 8
  sigma: 0.1           # Stored Gaussian weights; models use unweighted edges.
labels:
  threshold: 0.3       # Existing graph-label overlap threshold; do not regenerate labels.
data:
  graph_dir: data/graphs/train
  train_ratio: 0.7
  val_ratio: 0.15
training:
  epochs: 150
  lr: 0.001
  weight_decay: 0.0005
  warmup_epochs: 5
  grad_clip_norm: 1.0
  batch_size: 1
  early_stopping_patience: 20
  seed: 42
output_dir: results/M4_hybrid_full
```

The multi-seed runner chooses weighted CE or focal loss explicitly. Hybrid models
use five warm-up epochs and gradient clipping as configured; graph-only models
use their own unchanged YAML settings. Splits vary by seed, while the independent
237-image holdout is fixed.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
