# Defect Classification System (DCS) & Pipeline Crack GNN — Complete Project Summary

**Project:** Pipeline Surface Crack Detection Using Superpixel Graph Neural Networks & Transformer Hybrid  
**Workspace:** `c:\antigravity_projects\dcsnew`  
**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA Acceleration)  
**Deep Learning Stack:** PyTorch 2.x, PyTorch Geometric 2.x (**100% PyTorch — Zero TensorFlow**)  
**Backend Framework:** FastAPI, Ultralytics YOLOv11  
**Evaluation Benchmark:** DeepCrack (537 Images) & TITS Multi-Sensor Dataset  
**Date:** September 2026  

---

## 1. Executive Summary

This project encompasses two major interconnected engineering and research initiatives:
1. **DCS Backend Production Refactoring:** Modernization and security hardening of the FastAPI + YOLOv11 defect detection API for robotic pipeline rovers.
2. **Pipeline Crack Detection GNN + Transformer System:** Development, multi-seed statistical validation, and empirical benchmarking of a novel **Hybrid GNN + Transformer architecture (M4)** for pixel-level surface fissure detection on pipeline surfaces.

### Primary Breakthroughs & Scientific Milestones
- **Multi-Seed Rigor & Defensible Claims ($N=5$ Seeds):** M4 (Hybrid Full PE) achieves **F1 = 0.7378 ± 0.0200** vs. M1 (Deep GNN 8L) at **0.7159 ± 0.0314**. While overall differences are statistically non-significant after Bonferroni correction ($p = 0.2926$), M4 achieves decisive, statistically significant superiority on **long continuous cracks** (**F1 = 0.8120 vs. 0.7056, disjoint 95% CIs, $p < 0.001$**), while M1 specializes in **branched web networks** (**F1 = 0.7617 vs. 0.7115, disjoint 95% CIs, $p < 0.001$**).
- **Non-Circular Router Validation:** Independent validation confirmed morphology specialization. However, in production using predicted first-pass masks, the router achieves **F1 = 0.7578 [95% CI: 0.7170, 0.7969]**, performing comparably to standalone M4 ($p = 0.975$). We transparently report that first-pass mask noise (20.3% error) prevents the router from establishing a new production SOTA.
- **Empirical Proof of GNN Over-Smoothing:** Demonstrated that in an 8-layer Deep GNN, normalized Dirichlet energy collapses by **-42.8%** after layer 4, proving representation homogenization. The Hybrid M4 solves this by truncating GNN depth at 2 layers and delegating long-range reasoning to self-attention.
- **Root-Cause Resolution of Transformer Deficit:** Discovered that ~80% of prior transformer underperformance on sparse graphs was caused by loss imbalance (97.4% background token domination). Focal Loss ($\gamma=2$) unlocked the true structural power of the Transformer backbone.
- **Multi-Sensor Benchmark Rectification:** Corrected TITS evaluation methodology by isolating negative control sensors (LRIS, $N=13$) and establishing a leak-free 20% calibration / 80% holdout split. Accounted for sensor counts ($N=78$ total, $N=63$ holdout: 34 AIGLE_RN, 13 ESAR, 16 LCMS). Demonstrated that M4 leads on cross-sensor optical transfer (**F1 = 0.2889 vs 0.0000**), while acknowledging the overall cross-sensor transfer ceiling (micro F1 = 0.0991).

---

## 2. DCS Backend Modernization & Production Hardening

The Defect Classification System (DCS) backend was updated, secured, and calibrated for autonomous rover deployment:

| Module | Change Implemented | Rationale & Safety Impact |
|---|---|---|
| `backend/api.py` | Replaced wildcard `allow_origins=["*"]` with `ALLOWED_ORIGINS` environment variable (default `["http://localhost:3000"]`). | Prevents Cross-Origin Resource Sharing (CORS) exploits while allowing controlled dashboard connections. |
| `backend/inference.py` | Added `YOLO_CONF_THRESHOLD` environment variable (default: `0.25`). | Allows dynamic sensitivity tuning in noisy pipeline environments without modifying source code. |
| `backend/inference.py` | Added `DCS_ALLOW_SIMULATION` environment variable (default: `"false"`). Raises `ModelNotAvailableError` if model is missing and simulation is false. | Enforces strict failsafe: prevents rover from reporting synthetic mock defects during real pipeline inspection runs. |
| `backend/inference.py` | Converted hardcoded camera focal length (800.0 px) and rover standoff distance (1000.0 mm) to `ROVER_FOCAL_LENGTH_PX` and `ROVER_DISTANCE_MM`. | Supports dynamic field calibration for varying rover optical sensors and pipe diameter clearances. |
| Codebase Cleanup | Safely removed `backend/dataset_prep.py` and `backend/static/`. | Cleaned dead code and legacy frontend assets now superseded by Next.js. |

---

## 3. GNN + Transformer Architecture Suite

All models were implemented strictly in PyTorch and PyTorch Geometric under `pipeline-crack-gnn/src/models/`:

### M1: Deep GNN Baseline (`DeepGNN`)
- **Layers:** 8-layer deep `SAGEConv` + `BatchNorm` + ReLU (33,666 parameters).
- **Design Intent:** Kept intentionally without residual connections to serve as an empirical baseline for graph over-smoothing.

### M2: Shallow GNN (`ShallowGNN`)
- **Layers:** 2-layer shallow `SAGEConv` + `BatchNorm` + residual skip connection (8,706 parameters).
- **Design Intent:** Efficient local edge feature extractor with small receptive field.

### M3: Hybrid GNN + Transformer without Positional Encoding (`HybridGNNTransformer`, `use_pe=False`)
- **Layers:** 2-layer GNN encoder + 2-layer Pre-LN Transformer stack (114,755 parameters).
- **Design Intent:** Ablation baseline to isolate the necessity of spatial coordinates in all-to-all attention.

### M4: Hybrid GNN + Transformer with 2D Sinusoidal Positional Encoding (`HybridGNNTransformer`, `use_pe=True`)
- **Layers:** 2-layer GNN encoder + 2D Sinusoidal Positional Encoding (32 dims, 8 frequency octaves) + 2-layer Pre-LN Transformer stack + Linear Head (116,803 parameters).
- **Design Intent:** Combines local graph edge aggregation with physical coordinate grounding and global self-attention.

---

## 4. Master Benchmark Leaderboard (237 Holdout Images, 67,998 Unseen Nodes)

| Rank | Model Architecture | Loss Configuration | Operating $\tau^*$ | Precision | Recall | Test F1 | Test IoU | ROC-AUC | Parameters |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **M4 (Hybrid Full PE)** | **Focal Loss ($\gamma=2$)** | **0.66** | **0.7757** | 0.7548 | **0.7651** | **0.6195** | 0.9754 | 116,803 |
| **2** | **M1 (Deep GNN 8L)** | Weighted CE | 0.60 | 0.7584 | 0.7548 | **0.7566** | **0.6085** | 0.9764 | 33,666 |
| **3** | **Step 3 (M4 + Gate)** | Focal Loss ($\gamma=2$) | 0.70 | **0.7858** | 0.7171 | **0.7499** | **0.5998** | 0.9768 | 116,804 |
| **4** | **Step 4 (M4 + RelPos)** | Focal Loss ($\gamma=2$) | 0.64 | 0.7094 | **0.7828** | **0.7443** | **0.5927** | **0.9777** | 116,806 |
| **5** | **M3 (Hybrid No PE)** | Focal Loss ($\gamma=2$) | 0.74 | 0.7440 | 0.7036 | **0.7232** | **0.5664** | 0.9760 | 114,755 |
| **6** | **M2 (Shallow GNN 2L)** | Focal Loss ($\gamma=2$) | 0.70 | 0.6516 | 0.7285 | **0.6879** | **0.5243** | 0.9753 | 8,706 |

---

## 5. Non-Circular Router & Generalization Validation

Rather than deriving routing rules circularly on test data, thresholds were derived on an independent 45-image validation set ($AR \ge 1.5, A_{\text{frac}} < 0.015$).

| Configuration | Test F1 [95% CI] | Test IoU | Paired $t$-test vs M4 | Practical Implication |
|---|---|---|---|---|
| **Standalone M1** | 0.7566 [0.7156, 0.7952] | 0.6085 | $t = -1.456, p = 0.147$ | SOTA on branched web fissures |
| **Standalone M4** | 0.7651 [0.7218, 0.8043] | 0.6195 | Baseline | SOTA on continuous elongated cracks |
| **Oracle Router (GT Masks)** | 0.7657 [0.7225, 0.8051] | 0.6203 | $t = +0.814, p = 0.416$ | Theoretical upper limit of rule-based routing |
| **Production Router (Pred Masks)** | **0.7578 [0.7170, 0.7969]** | **0.6101** | **$t = -0.032, p = 0.975$** | Statistically comparable to standalone M4 |

**Honest Assessment:** Ground truth morphology routing achieves 0.7657 F1, confirming specialized capacity. However, in production, first-pass segmentation noise (20.3% error rate) causes routing mistakes, pulling F1 down to 0.7578. Therefore, standalone M4 is recommended for practical deployment.

---

## 6. TITS Multi-Sensor Transfer Benchmark (80% Holdout Split, $N=63$)

| Model Architecture | Optimal $\tau^*$ | AIGLE_RN ($N=34$) | ESAR ($N=13$) | LCMS 3D Laser ($N=16$) | Macro-Weighted F1 | Micro F1 |
|---|---|---|---|---|---|---|
| **M4 (Hybrid Full PE)** | **0.50** | **0.2889** | **0.1288** | 0.0174 | **0.1869** | **0.0991** |
| **M1 (Deep GNN 8L)** | 0.82 | 0.0000 | 0.0000 | **0.3546** | 0.0901 | 0.0507 |
| **M2 (Shallow GNN 2L)** | 0.84 | 0.0000 | 0.0000 | 0.1801 | 0.0457 | 0.0347 |
| **M3 (Hybrid No PE)** | 0.84 | 0.0000 | 0.0000 | 0.1306 | 0.0332 | 0.0325 |

- **Sensor Discrepancy Reconciliation:** Total positive sensor images are $N=78$ (42 AIGLE_RN, 16 ESAR, 20 LCMS). The 20% calibration split used $N=15$ images (8 AIGLE_RN, 3 ESAR, 4 LCMS), leaving exactly $N=63$ images for holdout evaluation (34 AIGLE_RN, 13 ESAR, 16 LCMS).
- **M1 Sensitivity:** M1's score of 0.0000 on AIGLE_RN under global calibration was caused by probability compression (mean 0.0756 vs M4's 0.3031). Under target-specific thresholds, M1 recovers signal (F1 = 0.0946), but M4 remains substantially superior on optical cameras.
- **Cross-Sensor Ceiling:** Micro F1 = 0.0991 highlights that zero-shot transfer across distinct camera and lighting setups remains a significant open problem.

---

## 7. Deliverables & Project Asset Directory

### Core Model & Training Implementations
- `src/models/hybrid_model.py` — Hybrid GNN + Transformer with Residual Gating & Relative Positional Bias
- `src/models/transformer_block.py` — Pre-LN Transformer Block with additive distance attention mask
- `src/models/positional_encoding.py` — 2D Sinusoidal Positional Encoding & feature fusion
- `src/models/gnn_encoder.py` — DeepGNN (8L) and ShallowGNN (2L) implementations
- `src/train.py` — Focal Loss, warmup scheduling, early stopping, and Dirichlet energy logging

### Visual Artifacts & Interactive Tooling
- `dashboard/index.html` — Interactive Web Dashboard comparing models and morphology router
- `dashboard/predictions_data.json` — Sampled prediction data for live inspection
- `results/plots/multiseed_benchmark_errorbars.png` — Multi-seed benchmark with error bars
- `results/plots/bucketed_significance_cis.png` — Morphology bootstrap confidence intervals
- `results/plots/dirichlet_energy_curves.png` — Dirichlet energy collapse curve
- `results/plots/gate_trajectory_analysis.png` — Residual gate trajectory analysis
