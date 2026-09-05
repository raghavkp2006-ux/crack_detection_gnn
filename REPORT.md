# Pipeline Surface Defect & Crack Detection: Comprehensive Technical & Empirical Report

**Project Title:** Graph Neural Networks & Transformer Hybrids for Pipeline Crack Detection and Rover Defect Classification  
**Repository:** `https://github.com/raghavkp2006-ux/crack_detection_gnn.git`  
**Evaluation Benchmarks:** DeepCrack (537 Images, 67,998 Holdout Nodes) & TITS Multi-Sensor Dataset (78 Positive Sensors, 13 Negative Control)  
**Deep Learning Stack:** PyTorch 2.x, PyTorch Geometric 2.x (**100% PyTorch — Zero TensorFlow**)  
**Production Backend:** FastAPI, Ultralytics YOLOv11  
**Hardware Accelerated:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA Acceleration)  
**Date:** September 2026  

---

## Table of Contents
1. [Executive Summary & Key Takeaways](#1-executive-summary--key-takeaways)
2. [Problem Formulation & Mathematical Foundation](#2-problem-formulation--mathematical-foundation)
3. [Model Architectures & Parameter Scales](#3-model-architectures--parameter-scales)
4. [6-Step Empirical Optimization Program](#4-the-6-step-empirical-optimization-program)
5. [Multi-Seed Master Benchmark Leaderboard ($N=5$ Seeds)](#5-multi-seed-master-benchmark-leaderboard-n5-seeds)
6. [Bucketed Morphology Specialization & Effect Sizes](#6-bucketed-morphology-specialization--effect-sizes)
7. [Non-Circular Morphology Router Validation & Error Analysis](#7-non-circular-morphology-router-validation--error-analysis)
8. [TITS Multi-Sensor Cross-Domain Benchmark & Calibration Stability](#8-tits-multi-sensor-cross-domain-benchmark--calibration-stability)
9. [Dirichlet Energy & Empirical Proof of Over-Smoothing](#9-dirichlet-energy--empirical-proof-of-over-smoothing)
10. [DCS Backend Modernization & Robotic Rover Hardening](#10-dcs-backend-modernization--robotic-rover-hardening)
11. [Sensor-Conditional Engineering Deployment Guidelines](#11-sensor-conditional-engineering-deployment-guidelines)
12. [Evidence-Based Retraining Assessment & Conclusion](#12-evidence-based-retraining-assessment--conclusion)
13. [Artifact & Code Manifest](#13-artifact--code-manifest)

---

## 1. Executive Summary & Key Takeaways

This report synthesizes the end-to-end development, mathematical formulation, multi-seed statistical validation, and cross-sensor evaluation of Graph Neural Networks (GNNs) and Transformer Hybrids for automated surface crack detection in industrial pipeline rovers.

### Primary Scientific Findings:
1. **Neither Architecture Wins Overall on Unweighted Aggregate Metrics:**
   - Across $N=5$ random seeds, **M4 (Hybrid Full PE)** averages **F1 = 0.7378 ± 0.0200** vs. **M1 (Deep GNN 8L)** at **F1 = 0.7159 ± 0.0314**.
   - Seed-level paired t-test yields $t = 1.2109, p = 0.2926$ ($d = 0.542$). Graph-level paired t-test across 237 holdout graphs yields $t = -1.4557, p = 0.1468$. Under family-wise Bonferroni correction ($\alpha_{\text{adj}} = 0.0125$), aggregate differences between M4 and M1 are **statistically non-significant**.
2. **Decisive, Defensible Morphology Specialization (Primary Contribution):**
   - **Continuous Long Cracks ($N=55$):** M4 demonstrates decisive superiority with **F1 = 0.8120 [95% CI: 0.8028, 0.8188]** vs. M1's **0.7056 [95% CI: 0.6640, 0.7393]** (**+10.64% F1 advantage, Cohen's $d = 1.480$ "Very Large", disjoint 95% CIs, $p < 0.001$**).
   - **Branched Web Networks ($N=98$):** M1 demonstrates decisive superiority with **F1 = 0.7617 [95% CI: 0.7506, 0.7722]** vs. M4's **0.7115 [95% CI: 0.6913, 0.7358]** (**+5.02% F1 advantage, Cohen's $d = 0.895$ "Large", disjoint 95% CIs, $p < 0.001$**).
   - **Thin Fissures ($N=84$):** M4 leads with F1 = 0.6747 vs M1's 0.6142, but **95% confidence intervals overlap** (0.6383 < 0.6673), making this lead statistically inconclusive.
   - **Significance:** These large-to-very-large effect sizes ($d = 1.480$ and $d = 0.895$) establish morphology specialization as the most robust, reproducible finding of this investigation.
3. **Non-Circular Router Validation & Error Diagnosis:**
   - Evaluated using independent validation thresholds ($AR \ge 1.5, A_{\text{frac}} < 0.015$), an Oracle Router achieves **F1 = 0.7657**.
   - In production inference using predicted first-pass masks from M1, the router achieves **F1 = 0.7578 [95% CI: 0.7170, 0.7969]**, statistically indistinguishable from standalone M4 ($p = 0.9747$).
   - **Error Concentration:** Routing errors concentrate in Branched Cracks (**34.88% misrouting rate** vs 12.50% on Long and 8.70% on Thin), where fragmented first-pass masks fool bounding-box aspect ratio heuristics. We do **not** claim a new SOTA for the router in production.
4. **TITS Cross-Sensor Benchmark & Modality Divergence:**
   - Under single-draw calibration ($N=15$), M4 achieves micro F1 = 0.0991 vs M1's 0.0507.
   - Stability analysis across **10 independent stratified calibration draws** reveals that all models cluster tightly in aggregate (M4: **0.0815 ± 0.0487**, M1: **0.0723 ± 0.0291**) with large threshold variance ($\tau^* = 0.58 \pm 0.32$ for M1).
   - **Modality Divergence:** M4 dominates on **optical surface cameras** (AIGLE_RN F1 = **0.1175 ± 0.1171** vs M1's 0.0390 ± 0.0522), whereas M1 dominates on **3D laser profilometry** (LCMS F1 = **0.2570 ± 0.1289** vs M4's 0.0739 ± 0.0429). Sensor modality must drive model selection.
5. **Empirical Proof of GNN Over-Smoothing:**
   - Normalized Dirichlet energy collapses by **-42.8%** from Layer 4 to Layer 8 in deep GNNs, providing quantitative proof of representation homogenization. M4 circumvents this by truncating GNN depth at 2 layers and handling global context via self-attention.

---

## 2. Problem Formulation & Mathematical Foundation

Pipeline surface crack inspection requires detecting thin, tortuous, and non-local topological defects amidst severe background imbalance (cracks occupy ~2.6% of image pixels).

```
Raw Image (H × W × 3)
         │
         ▼
SLIC Superpixel Segmentation (N ≈ 300 regions)
         │
         ▼
15-dim Feature Extraction + 2D Centroids (x, y)
         │
         ▼
PIRM Graph Construction (k-NN in feature space + Gaussian affinity)
         │
    ┌────┴────────────────────────┐
    ▼                             ▼
Shallow GNN (2 Layers)     2D Fourier Positional Encoding
Local Message Passing      Metric Spatial Coordinates
    └────┬────────────────────────┘
         ▼
Pre-LN Transformer Stack (2 Layers, 4 Heads)
Global Context & Long-Range Fissure Reconnection
         │
         ▼
Binary Node Classification (Crack vs. Non-Crack)
```

### 1. PIRM Graph Construction:
- **Segmentation:** Simple Linear Iterative Clustering (SLIC) segments each image into $N \approx 300$ superpixels.
- **Node Features ($d=15$):**
  - 3-channel mean color intensity (RGB)
  - 1 scalar standard deviation across channels
  - 1 Sobel gradient magnitude mean
  - 10-bin Local Binary Pattern (LBP) texture histogram
- **Graph Edges:** Pixel Intensity Resemblance Method (PIRM) constructs edges using $k$-nearest neighbors ($k=8$) in feature space with a Gaussian affinity kernel:
  $$W_{ij} = \exp\left( -\frac{\|f_i - f_j\|_2^2}{2\sigma^2} \right)$$

### 2. 2D Sinusoidal Positional Encoding:
To break graph permutation invariance and inject spatial physical coordinates:
$$\text{PE}_{(p, 2i)} = \sin\left( \frac{p}{10000^{2i/d_{\text{pe}}}} \right), \quad \text{PE}_{(p, 2i+1)} = \cos\left( \frac{p}{10000^{2i/d_{\text{pe}}}} \right)$$
computed independently across normalized centroid coordinates $(x_{\text{norm}}, y_{\text{norm}}) \in [0, 1]^2$ with 8 frequency octaves ($d_{\text{pe}} = 32$).

### 3. Pre-LN Multi-Head Self-Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + \text{attn\_bias} \right) V$$
where $\text{attn\_bias}_{ij} = -\frac{\|\text{pos}_i - \text{pos}_j\|_2}{\text{temperature}}$ enforces physical distance penalties.

---

## 3. Model Architectures & Parameter Scales

| Model Identifier | Model Architecture | Layer Configuration | Parameter Count | Core Functional Role |
|---|---|---|:---:|---|
| **M1** | `DeepGNN` | 8-layer `SAGEConv` + `BatchNorm` + ReLU (no residual skip) | 33,666 | Baseline for measuring Dirichlet energy over-smoothing. |
| **M2** | `ShallowGNN` | 2-layer `SAGEConv` + `BatchNorm` + residual skip connection | 8,706 | Ultra-low-latency edge feature extractor for mobile rovers. |
| **M3** | `HybridGNNTransformer` (No PE) | 2-layer GNN + 2-layer Pre-LN Transformer (without spatial PE) | 114,755 | Ablation baseline isolating coordinate contribution. |
| **M4** | `HybridGNNTransformer` (Full PE) | 2-layer GNN + 32-dim 2D Sinusoidal PE + 2-layer Pre-LN Transformer | 116,803 | Proposed champion architecture combining local and global reasoning. |

---

## 4. The 6-Step Empirical Optimization Program

### Step 1: Decision Threshold Calibration ($\tau^*$)
- **Context:** Superpixel crack nodes are sparse (~3.17% of total nodes). The standard decision rule ($\tau = 0.50$) caused severe recall suppression under weighted loss.
- **Action:** Swept thresholds $\tau \in [0.10, 0.90]$ in steps of 0.02 on the validation set.
- **Impact:** M2 jumped **+62.7% F1** ($0.3810 \to 0.6200$) on internal validation, and on holdout test M2 reached **0.7435 F1**, nearly matching the 8-layer deep GNN ($0.7566$).

### Step 2: Focal Loss Isolation ($\gamma=2.0$)
- **Context:** Dissected whether transformer underperformance on graphs was architectural or loss-driven.
- **Action:** Added `FocalLoss` ($\gamma=2.0$, $\alpha=[0.0256, 0.9744]$) to suppress easy background token gradients by up to $10,000\times$. Retrained M2, M3, and M4 holding all seeds constant.
- **Impact:** M4 became #1 across the entire holdout benchmark (**F1 = 0.7651, IoU = 0.6195**). M3 surged by $+0.2540$ F1 ($0.4692 \to 0.7232$), proving ~80% of prior transformer deficit was gradient flooding by background tokens.

### Step 3: Learned Residual Gate Diagnostics
- **Context:** Evaluated scalar gating between GNN and Transformer:
  $$x_{\text{out}} = \sigma(\text{gate}) \cdot x_{\text{trans}} + (1 - \sigma(\text{gate})) \cdot x_{\text{gnn}}$$
- **Observation:** Initialized at $\sigma(0.0) = 0.50$, the scalar gate hovered between 0.4939 and 0.5102 ($\Delta = 0.0162, \sigma = 0.00581$).
- **Conclusion:** Global scalar gating experiences weak aggregate gradient pull across thousands of nodes. True spatial adaptation takes place inside the attention matrix rather than at the macroscopic model level.

### Step 4: Relative Positional Bias in Self-Attention
- **Context:** Injected physical metric distance penalties directly into self-attention logits:
  $$\text{attn\_bias}_{ij} = -\frac{\|\text{pos}_i - \text{pos}_j\|_2}{\text{temperature}}$$
- **Observation:** Temperature parameter dynamically decayed from **$1.0 \to 0.479 \to 0.070 \to 0.001$** (clamping floor).
- **Conclusion:** Gradient descent actively learns to sharpen spatial locality, penalizing distant unrelated tokens while preserving connectivity along long continuous crack paths.

### Step 5: Morphology Bucketing & Disjoint Confidence Intervals
Categorized 237 holdout benchmark images into physical crack morphologies using aspect ratio ($AR$) and area fraction ($A_{\text{frac}}$):
- **Long / Elongated Cracks ($N=55$):** $AR \ge 2.0$
- **Thin / Fine Fissures ($N=84$):** $A_{\text{frac}} < 0.025$
- **Branched / Complex Networks ($N=98$):** $AR < 2.0, A_{\text{frac}} \ge 0.025$
Empirically proved that M4 and M1 specialize in disjoint crack morphologies with non-overlapping 95% bootstrap confidence intervals.

### Step 6: Dirichlet Energy Over-Smoothing Quantification
Quantified normalized Dirichlet energy layer-by-layer across all holdout graphs to prove feature collapse in deep GNNs.

---

## 5. Multi-Seed Master Benchmark Leaderboard ($N=5$ Seeds)

All models were evaluated across $N=5$ random seeds (42, 123, 456, 789, 2026) on the untouched 237 holdout images (67,998 unseen superpixel nodes):

| Model Architecture | Parameter Count | F1 Score (Mean ± Std) | IoU (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | ROC-AUC (Mean ± Std) |
|---|---|---|---|---|---|---|
| **M4 (Hybrid Full PE)** | 116,803 | **0.7378 ± 0.0200** | **0.5848 ± 0.0250** | 0.8099 ± 0.0837 | 0.6877 ± 0.0691 | 0.9728 ± 0.0031 |
| **M3 (Hybrid No PE)** | 114,755 | 0.7264 ± 0.0281 | 0.5709 ± 0.0340 | **0.8381 ± 0.0926** | 0.6535 ± 0.0833 | **0.9766 ± 0.0015** |
| **M2 (Shallow GNN 2L)** | 8,706 | 0.7199 ± 0.0157 | 0.5626 ± 0.0192 | 0.8061 ± 0.1119 | 0.6637 ± 0.0659 | 0.9753 ± 0.0016 |
| **M1 (Deep GNN 8L)** | 33,666 | 0.7159 ± 0.0314 | 0.5583 ± 0.0381 | 0.6896 ± 0.1153 | **0.7655 ± 0.0710** | 0.9608 ± 0.0138 |

### Hypothesis Testing & Effect Size Analysis (M4 vs. M1):
- **Seed-Level Paired t-test ($N=5$):** $t = 1.2109, p = 0.2926$ (Cohen's $d = 0.542$, Medium). Statistically non-significant.
- **Graph-Level Paired t-test ($N=237$):** $t = -1.4557, p = 0.1468$ (Cohen's $d = 0.144$, Negligible). Statistically non-significant.
- **Graph-Level Wilcoxon Signed-Rank:** $W = 12681.0, p = 0.4157$. Statistically non-significant.
- **Bonferroni-Corrected Threshold:** $\alpha_{\text{adj}} = 0.05 / 4 = 0.0125$. The overall aggregate difference between M4 and M1 is statistically indistinguishable.

---

## 6. Bucketed Morphology Specialization & Effect Sizes

Evaluating models within specific physical crack geometries reveals where each inductive bias succeeds:

| Morphology Class | M1 (Deep GNN 8L) Mean [95% CI] | M2 (Shallow GNN 2L) Mean [95% CI] | M3 (Hybrid No PE) Mean [95% CI] | M4 (Hybrid Full PE) Mean [95% CI] | Statistical Conclusion & Effect Size |
|---|---|---|---|---|---|
| **Long / Elongated ($N=55$)** | 0.7056 [0.6640, 0.7393] | 0.7961 [0.7867, 0.8038] | 0.7895 [0.7675, 0.8065] | **0.8120 [0.8028, 0.8188]** | **M4 definitively superior (+10.64% F1, $d=1.480$, disjoint CIs, $p < 0.001$)** |
| **Thin / Fine Fissures ($N=84$)** | 0.6142 [0.5563, 0.6673] | 0.6554 [0.6454, 0.6655] | 0.6716 [0.6445, 0.6906] | **0.6747 [0.6383, 0.7084]** | **Inconclusive (CIs overlap: 0.6383 < 0.6673, $d=0.380$)** |
| **Branched / Complex ($N=98$)** | **0.7617 [0.7506, 0.7722]** | 0.6925 [0.6763, 0.7105] | 0.7038 [0.6743, 0.7313] | 0.7115 [0.6913, 0.7358] | **M1 definitively superior (+5.02% F1, $d=0.895$, disjoint CIs, $p < 0.001$)** |

### Why This Specialization Occurs:
- **Long Continuous Fissures:** 2D Positional Encoding grounds distant superpixels in physical Cartesian coordinates. Pre-LN self-attention bridges spatial gaps across straight pipe walls where graph edge diffusion fails.
- **Branched Web Networks:** Multi-directional fissures create dense local graph cliques. Coordinate-free graph message passing (M1) aggregates neighborhood connectivity organically without being constrained by 2D Euclidean linear assumptions.

---

## 7. Non-Circular Morphology Router Validation & Error Analysis

To determine whether morphology specialization can be exploited dynamically at test time without oracle ground-truth masks:
1. **Independent Validation Derivation:** Thresholds were fit strictly on the 45-image validation split ($AR \ge 1.5, A_{\text{frac}} < 0.015$, Val F1 = 0.7462).
2. **Two-Stage Production Pipeline:** In inference, M1 generates a fast first-pass segmentation mask. We measure predicted $AR$ and $A_{\text{frac}}$, routing to M4 if $AR \ge 1.5$ or $A_{\text{frac}} < 0.015$, and to M1 otherwise.

### Router Performance on 237 Holdout Images:

| Architecture / Routing Configuration | Test F1 [95% CI] | Test IoU | Paired $t$-test vs M4 ($p$-value) | Paired $t$-test vs Max(M1, M4) ($p$-value) |
|---|---|---|---|---|
| **Standalone M1 (Deep GNN)** | 0.7566 [0.7156, 0.7952] | 0.6085 | $t = -1.456, p = 0.1468$ | $t = -6.442, p = 7.12 \times 10^{-10}$ |
| **Standalone M4 (Hybrid Full PE)** | 0.7651 [0.7218, 0.8043] | 0.6195 | Baseline | $t = -5.461, p = 1.15 \times 10^{-7}$ |
| **Oracle GT Router (Upper Bound)** | 0.7657 [0.7225, 0.8051] | 0.6203 | $t = +0.814, p = 0.4162$ | $t = -4.920, p = 1.54 \times 10^{-6}$ |
| **Production Predicted-Mask Router** | **0.7578 [0.7170, 0.7969]** | **0.6101** | **$t = -0.032, p = 0.9747$** | **$t = -6.792, p = 8.93 \times 10^{-11}$** |

### Routing Error Rate Breakdown by Morphology Bucket:

Note that "Routing Accuracy" and "Exact Category Match" are distinct metrics. The router collapses three morphology categories onto two models (Long_Elongated and Thin_Fine_Fissure both route to M4; only Branched_Complex routes to M1). A first-pass mask can therefore be mis-classified into the wrong category yet still be routed to the correct model, simply because two of the three categories share the same destination. This is why Long and Thin show materially higher Routing Accuracy than Category Accuracy, while Branched — the only category mapping to M1 — shows identical values for both, since any category error there is automatically also a routing error.

| True Morphology Bucket | Total Images ($N$) | Correctly Routed | Routing Accuracy | Routing Error Rate | Exact Category Match |
|---|---|---|---|---|---|
| **Long / Elongated** | 128 | 112 | 87.50% | **12.50%** | 78.12% |
| **Thin / Fine Fissure** | 23 | 21 | 91.30% | **8.70%** | 56.52% |
| **Branched / Complex** | 86 | 56 | 65.12% | **34.88%** | 65.12% |
| **Overall Aggregate** | **237** | **189** | **79.75%** | **20.25%** | **71.31%** |

### Key Diagnostic Discovery:
- **Routing errors are heavily concentrated in Branched Cracks (34.88% misrouting rate).**
- When M1 predicts a rough or broken mask on complex web fissures, disconnected fragments appear artificially elongated ($AR \ge 1.5$) or sparse ($A_{\text{frac}} < 0.015$). This falsely triggers the rule to route to M4 instead of M1.
- Because M1 is significantly superior on branched networks (+5.02% F1 over M4), misrouting 30 out of 86 branched graphs directly reduces aggregate production performance down to 0.7578.
- **Deployment Implication:** Standalone M4 remains the simplest, lowest-latency, and most dependable system for real-time robotic rover deployment.

---

## 8. TITS Multi-Sensor Cross-Domain Benchmark & Calibration Stability

### 1. Dataset Architecture & Reconciliation:
- **Negative Control:** LRIS ($N=13$, 3,892 nodes, 0 crack annotations) evaluated purely for False Positive Rate (FPR) and Specificity.
- **Dropped Sensor:** TEMPEST2 ($N=1$) dropped due to 0 crack annotations and lack of statistical validity.
- **Positive Sensors ($N=78$ Total Images):**
  - `AIGLE_RN` ($N=42$ total): 8 assigned to 20% calibration, 34 assigned to 80% holdout.
  - `ESAR` ($N=16$ total): 3 assigned to 20% calibration, 13 assigned to 80% holdout.
  - `LCMS` ($N=20$ total): 4 assigned to 20% calibration, 16 assigned to 80% holdout.
- **Calibration Split (20%, $N=15$):** Used strictly to identify $\tau^*$.
- **Holdout Evaluation Split (80%, $N=63$):** Completely blind evaluation across 18,655 superpixel nodes.

### 2. Multi-Draw Calibration Stability Analysis ($N_{\text{draws}} = 10$):
To quantify calibration instability directly, we evaluated all models across 10 independent stratified draws:

| Model Architecture | $\tau^*$ (Mean ± Std) | Micro F1 (Mean ± Std) | Micro IoU (Mean ± Std) | Micro ROC-AUC | AIGLE_RN F1 (Mean ± Std) | LCMS F1 (Mean ± Std) | ESAR F1 (Mean ± Std) |
|---|---|---|---|---|---|---|---|
| **M4 (Hybrid Full PE)** | 0.57 ± 0.21 | **0.0815 ± 0.0487** | **0.0432 ± 0.0265** | 0.6108 ± 0.0120 | **0.1175 ± 0.1171** | 0.0739 ± 0.0429 | 0.0780 ± 0.0786 |
| **M3 (Hybrid No PE)** | 0.67 ± 0.17 | 0.0789 ± 0.0405 | 0.0415 ± 0.0220 | 0.5899 ± 0.0076 | 0.0955 ± 0.0968 | 0.0952 ± 0.0630 | 0.0838 ± 0.0846 |
| **M2 (Shallow GNN 2L)** | 0.61 ± 0.23 | 0.0745 ± 0.0357 | 0.0391 ± 0.0193 | 0.5682 ± 0.0080 | 0.0827 ± 0.0844 | 0.1180 ± 0.0856 | **0.0864 ± 0.0872** |
| **M1 (Deep GNN 8L)** | 0.58 ± 0.32 | 0.0723 ± 0.0291 | 0.0377 ± 0.0157 | **0.6379 ± 0.0076** | 0.0390 ± 0.0522 | **0.2570 ± 0.1289** | 0.0349 ± 0.0585 |

### 3. Symmetric Modality Specialization (Equal Narrative Weight, Statistically Verified):
- **Optical Surface Cameras (AIGLE_RN):** M4 leads decisively (**0.1175 ± 0.1171 vs. M1's 0.0390 ± 0.0522**, 3.0x higher F1). Paired t-test across the 10 calibration draws confirms this is statistically significant ($t=2.659$, $p=0.0261$, Cohen's $d=0.841$, Large effect).
- **3D Laser Profilometry (LCMS):** M1 dominates decisively (**0.2570 ± 0.1289 vs. M4's 0.0739 ± 0.0429**, 3.5x higher F1). Paired t-test confirms this is statistically significant ($t=-5.052$, $p=0.0007$, Cohen's $d=-1.598$, Very Large effect) — the strongest single-sensor effect size in this report.
- **ESAR:** M4 leads on point estimate (0.0780 vs. 0.0349) but this does not reach significance ($t=2.068$, $p=0.0686$, Cohen's $d=0.654$) and should not be cited as a settled finding.
- **M1 Structural Instability:** M1's optimal threshold swings between $\tau^* = 0.10$ and $\tau^* = 0.88$ (std = 0.32). Because M1's probability mass is compressed near zero (mean = 0.0756), minor sampling shifts in calibration splits create threshold collapse.
- **Cross-Sensor Performance Ceiling:** Aggregate micro F1s below 0.10 across all architectures confirm that zero-shot transfer across distinct physical sensor domains remains an open challenge requiring target-domain fine-tuning or domain adaptation.

---

## 9. Dirichlet Energy & Empirical Proof of Over-Smoothing

To rigorously verify graph representation homogenization, we calculated normalized Dirichlet energy layer-by-layer across all holdout test graphs:
$$E_{\text{norm}}(H^{(l)}) = \frac{1}{|E|} \sum_{(i,j) \in E} \left\| \frac{h_i^{(l)}}{\|h_i^{(l)}\|_2} - \frac{h_j^{(l)}}{\|h_j^{(l)}\|_2} \right\|_2^2$$

- **Layer 1:** $E_{\text{norm}} = 0.0521$
- **Layer 2:** $E_{\text{norm}} = 0.0532$
- **Layer 4 (Peak):** $E_{\text{norm}} = \mathbf{0.0797}$
- **Layer 8:** $E_{\text{norm}} = \mathbf{0.0456}$ (**-42.8% collapse from peak**)

**Scientific Implication:** Deep stacking of graph convolution layers collapses feature variance across neighboring nodes. Truncating GNN depth at 2 layers ($E_{\text{norm}} = 0.0490$) and delegating long-range spatial context to Pre-LN Transformer attention (as implemented in M4) provides the optimal architectural balance.

---

## 10. DCS Backend Modernization & Robotic Rover Hardening

In parallel with GNN research, the Defect Classification System (DCS) FastAPI backend was refactored and secured for autonomous rover deployment:

| Module | Implemented Security & Architectural Change | Production Safety Rationale |
|---|---|---|
| `backend/api.py` | Configurable `ALLOWED_ORIGINS` environment variable (default: `["http://localhost:3000"]`). | Eliminates wildcard CORS vulnerabilities (`"*"`) while enabling secure dashboard telemetry. |
| `backend/inference.py` | Environment variable `YOLO_CONF_THRESHOLD` (default: `0.25`). | Enables runtime sensitivity tuning without modifying codebase or redeploying Docker containers. |
| `backend/inference.py` | `DCS_ALLOW_SIMULATION` failsafe flag (default: `"false"`). | Raises explicit `ModelNotAvailableError` if model weights are missing; prevents rover from generating synthetic mock defects in real pipeline pipes. |
| `backend/inference.py` | Calibrated camera focal length (`ROVER_FOCAL_LENGTH_PX`) and standoff distance (`ROVER_DISTANCE_MM`). | Dynamic field calibration supporting variable camera lenses and pipe diameter clearances. |
| Repository Hygiene | Removed deprecated `backend/dataset_prep.py` and `backend/static/`. | Eliminated dead code and legacy assets superseded by Next.js. |

---

## 11. Sensor-Conditional Engineering Deployment Guidelines

| Target Inspection Scenario | Recommended Model Architecture | Operating Threshold ($\tau^*$) | Rationale & Performance |
|---|---|:---:|---|
| **Optical Pipeline Wall Cameras** | **M4 (Hybrid Full PE)** | $\tau \approx 0.50$ | Dominates optical surface transfer (F1 = **0.1175 ± 0.1171**, 3x over M1). |
| **3D Laser Profilometry Scanners** | **M1 (Deep GNN 8L)** | $\tau \approx 0.30$ | Excels on coordinate-free depth/range topologies (F1 = **0.2570 ± 0.1289**, 3.5x over M4). |
| **Edge Compute / Micro-Rovers (<10 MB RAM)** | **M2 (Shallow GNN 2L)** | $\tau \approx 0.60$ | Ultra-compact (8,706 params, 150+ FPS, F1 = **0.0745 ± 0.0357**). |
| **Continuous Longitudinal Pipe Fissures** | **M4 (Hybrid Full PE)** | $\tau \approx 0.64$ | Proven superior on long cracks (**F1 = 0.8120 vs 0.7056**, $d=1.480$, $p<0.001$). |
| **Fatigue Web / Branched Networks** | **M1 (Deep GNN 8L)** | $\tau \approx 0.60$ | Proven superior on branched networks (**F1 = 0.7617 vs 0.7115**, $d=0.895$, $p<0.001$). |

---

## 12. Evidence-Based Retraining Assessment & Conclusion

Following completion of the script re-analyses in Next Steps (v4), we evaluated whether model retraining or additional data collection is justified:
1. **Calibration Instability Root Cause:** The multi-draw benchmark proved that cross-sensor instability is driven by calibration sample scarcity ($N=15$ images across 3 sensors) and severe modality distribution shifts (optical RGB vs laser range profilometry). Retraining backbones on DeepCrack cannot resolve cross-sensor physical domain gaps. The appropriate engineering next step is gathering 30–50 calibration images per target sensor or applying unsupervised domain adaptation.
2. **Router Degradation Root Cause:** Routing misclassifications are concentrated in Branched Cracks (34.88% error) due to bounding box aspect-ratio heuristic failures on fragmented masks. Retraining backbones does not fix this heuristic limitation. Future work should target graph-skeleton topological analysis or joint end-to-end routing gates.
3. **Definitive Conclusion (Scoped):** No model retraining or data re-collection is justified for the two questions investigated in this section — TITS calibration instability (driven by calibration sample scarcity, not model quality) and router degradation (driven by a bounding-box heuristic limitation, not model quality). The morphology specialization finding ($d=1.480$ on Long, $d=0.895$ on Branched) stands as a complete, statistically verified, and defensible scientific contribution independent of these two questions. Thin/Fine Fissure performance, where all four models cluster within a narrow, mediocre band (F1 $\approx$ 0.61–0.67), remains an open question outside the scope of this analysis and is a candidate for future targeted work.

---

## 13. Artifact & Code Manifest

All assets are versioned and directly inspectable in the repository:

### Core Scripts & Pipelines:
- `src/models/hybrid_model.py` — Complete Hybrid GNN + Transformer implementation
- `src/models/transformer_block.py` — Pre-LN Transformer with additive spatial distance bias
- `src/models/positional_encoding.py` — 2D Sinusoidal Positional Encoding
- `src/models/gnn_encoder.py` — DeepGNN (8L) and ShallowGNN (2L)
- `src/train.py` — Focal loss, warmup scheduling, and Dirichlet energy logging
- `run_multiseed_experiment.py` — Multi-seed training and hypothesis testing suite
- `run_noncircular_router_validation.py` — Non-circular validation and two-stage routing pipeline
- `analyze_routing_error_by_bucket.py` — Reproducible breakdown of router error rate by morphology category
- `run_tits_rigorous_evaluation.py` — TITS leak-free calibration evaluation
- `analyze_tits_modality_significance.py` — Paired t-tests and Cohen's d across 10 TITS calibration draws

### Structured Empirical Metrics (JSON):
- `results/tits_modality_significance.json` — Sensor modality divergence statistical significance report
- `results/tits_multidraw_calibration.json` — 10-draw calibration stability summary metrics
- `results/router_per_image_categories.json` — Per-image true vs predicted morphology category records
- `results/routing_error_by_bucket.json` — Morphology bucket routing error breakdown
- `results/noncircular_router_validation.json` — Non-circular router evaluation results
- `results/multiseed/multiseed_summary.json` — Full 5-seed benchmark results and CIs
- `results/full_evaluation_summary.json` — Comprehensive holdout metrics

### Visual Plots:
- `results/plots/multiseed_benchmark_errorbars.png` — Multi-seed master leaderboard
- `results/plots/bucketed_significance_cis.png` — Morphology bootstrap 95% confidence intervals
- `results/plots/dirichlet_energy_curves.png` — Layer-by-layer Dirichlet energy collapse curve
- `results/plots/gate_trajectory_analysis.png` — Residual gate trajectory dynamics

### Interactive Visual Tooling:
- `dashboard/index.html` — Interactive Web Dashboard for live model comparison and prediction inspection
- `dashboard/predictions_data.json` — Superpixel prediction telemetry data
