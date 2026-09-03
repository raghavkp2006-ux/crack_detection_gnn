# Pipeline Crack Detection using Graph Neural Networks and Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-orange.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework for automated surface and pipeline crack detection using **Position-Aware Interacting Region Map (PIRM) graphs**, **Shallow GNN encoders**, **2D Sinusoidal Positional Encoding**, and **Transformer Self-Attention**.

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
- **PIRM Graph Construction:** Groups homogeneous image regions into superpixel nodes (SLIC) and connects them via affinity kernels based on spatial proximity and feature similarity.
- **Hybrid GNN-Transformer Architecture:** 
  - A **Shallow GNN** captures fine-grained local connectivity without suffering from over-smoothing.
  - **2D Fourier/Sinusoidal Positional Encoding** injects metric spatial coordinates to break graph permutation invariance.
  - **Multi-Head Transformer Self-Attention** enables long-range reasoning to reconnect broken or distant crack branches.

---

## 🏗 System Architecture

```
Raw Image (H × W × 3)
         │
         ▼
[SLIC Superpixel Segmentation] ─────────► Segments Array (N superpixels)
         │
         ├───► [Feature Extraction] ────► Node Features X ∈ ℝ^{N × 15}
         │     (Color, Sobel, LBP, Normed Centroids)
         │
         └───► [PIRM Graph Generator] ──► Adjacency Edge Index & Edge Weights
                        │
                        ▼
            PyG Graph Data (x, edge_index, pos, y)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  [Shallow GNN (2-layer GCN)]    [2D Sinusoidal Positional Encoding]
  (Local Structural Encoding)    (Spatial Frequency Coordinate Maps)
         │                             │
         └──────────────┬──────────────┘
                        │ Feature Concatenation / Linear Projection
                        ▼
          [Transformer Multi-Head Attention Stack]
          (Global Long-Range Dependency Capture)
                        │
                        ▼
          [MLP Node Classification Head]
                        │
                        ▼
           Predicted Crack Probabilities per Node
                        │
                        ▼
       [Pixel Mask Reconstruction via Superpixel Lookup]
                        │
                        ▼
           Binary Crack Segmentation Mask (H × W)
```

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
crack_detection_gnn/
│
├── pipeline-crack-gnn/
│   ├── configs/                           # Experiment YAML configurations
│   │   ├── baseline_deep_gnn.yaml         # M1: 8-layer deep GCN
│   │   ├── shallow_gnn.yaml               # M2: 2-layer shallow GCN
│   │   ├── hybrid_no_pe.yaml              # M3: GNN + Transformer (no PE)
│   │   └── hybrid_full.yaml               # M4: GNN + PE + Transformer (Full)
│   │
│   ├── src/
│   │   ├── models/                        # Neural network architectures
│   │   │   ├── gnn_encoder.py             # DeepGNN and ShallowGNN implementations
│   │   │   ├── positional_encoding.py     # 2D sinusoidal coordinate PE & fusion
│   │   │   ├── transformer_block.py       # Multi-head attention blocks & stacks
│   │   │   └── hybrid_model.py            # HybridGNNTransformer & build_model factory
│   │   │
│   │   ├── preprocessing/                 # Graph generation & feature pipeline
│   │   │   ├── superpixel.py              # SLIC superpixel segmentation & caching
│   │   │   ├── features.py                # 15D node features (Color, Sobel, LBP)
│   │   │   ├── pirm_graph.py              # PIRM graph construction & PyG data builder
│   │   │   └── build_dataset.py           # Batch preprocessing CLI runner
│   │   │
│   │   ├── utils/                         # Helper functions & utilities
│   │   │   ├── dataset.py                 # PyG dataset loader and train/val/test splits
│   │   │   ├── metrics.py                 # F1, IoU, ROC-AUC, Dirichlet energy, Cosine Sim
│   │   │   └── visualize.py               # Visualizer for superpixels, graphs & predictions
│   │   │
│   │   ├── train.py                       # Main training script with early stopping
│   │   ├── evaluate.py                    # Standalone test set evaluation & mask recovery
│   │   └── ablation.py                    # Multi-model ablation pipeline & bucket analysis
│   │
│   ├── tests/
│   │   └── smoke_test.py                  # End-to-end integration test with synthetic data
│   │
│   └── requirements.txt                   # Dependency definitions
│
└── README.md                              # Main documentation
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
Key training features:
- Automatically calculates inverse-frequency class weights to combat extreme crack pixel sparsity.
- Implements validation F1-score early stopping and checkpoint saving.
- Logs Dirichlet energy and representation cosine similarity across epochs to track over-smoothing.

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
The ablation suite categorizes cracks into three geometric profiles:
1. **Short / Compact Cracks:** High localized curvature, short diagonal footprint.
2. **Long / Diagonal Cracks:** Elongated continuity requiring long-range contextual association.
3. **Lighting-Variable Cracks:** Regions with substantial contrast and illumination shifts.

---

## 🔧 Configuration Reference

Configurations are stored in YAML format inside `configs/`. Example configuration (`hybrid_full.yaml`):

```yaml
model:
  type: "hybrid"            # 'deep_gnn', 'shallow_gnn', or 'hybrid'
  in_dim: 15               # 15D node feature vector
  hidden_dim: 64           # Latent dimension
  gnn_layers: 2            # Shallow GNN message-passing layers
  transformer_layers: 2    # Multi-head self-attention layers
  heads: 4                 # Attention heads
  pe_dim: 32               # Positional encoding dimension
  num_freqs: 8             # Number of Fourier frequency bands
  use_pe: true             # Enable 2D spatial positional encoding
  dropout: 0.1

data:
  n_segments: 200          # Target superpixels per image
  compactness: 10.0        # SLIC compactness parameter
  k_neighbors: 8           # PIRM k-NN graph connectivity
  sigma: 0.1               # Radial basis kernel bandwidth
  batch_size: 16
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

training:
  epochs: 100
  lr: 0.001
  weight_decay: 1.0e-4
  patience: 20
  output_dir: "results/M4_hybrid_full"
```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
