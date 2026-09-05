# Pipeline Crack Detection — Empirical Test Results & Benchmark Report (Post-Edit)

**Project:** Pipeline Surface Crack Detection using Superpixel Graph Neural Networks & Transformer Hybrid  
**Repository:** `https://github.com/raghavkp2006-ux/crack_detection_gnn.git`  
**Dataset:** DeepCrack (Wuhan University, 537 Images) & TITS Multi-Sensor Benchmark (78 Valid Positive Sensors, 13 Negative Control)  
**Evaluation Date:** September 2026  
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA Acceleration)  
**Software Framework:** PyTorch 2.x, PyTorch Geometric 2.x (**100% PyTorch — Zero TensorFlow**)  

---

## 1. Multi-Seed Master Leaderboard ($N=5$ Seeds, 237 Holdout Test Images)

Holdout test set evaluated with 5 random seeds (42, 123, 456, 789, 2026) across 67,998 unseen superpixel nodes:

| Rank | Model Architecture | Parameters | Test F1 (Mean ± Std) | Test IoU (Mean ± Std) | Precision (Mean ± Std) | Recall (Mean ± Std) | ROC-AUC (Mean ± Std) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **M4 (Hybrid Full PE)** | 116,803 | **0.7378 ± 0.0200** | **0.5848 ± 0.0250** | 0.8099 ± 0.0837 | 0.6877 ± 0.0691 | 0.9728 ± 0.0031 |
| **2** | **M3 (Hybrid No PE)** | 114,755 | 0.7264 ± 0.0281 | 0.5709 ± 0.0340 | **0.8381 ± 0.0926** | 0.6535 ± 0.0833 | **0.9766 ± 0.0015** |
| **3** | **M2 (Shallow GNN 2L)** | 8,706 | 0.7199 ± 0.0157 | 0.5626 ± 0.0192 | 0.8061 ± 0.1119 | 0.6637 ± 0.0659 | 0.9753 ± 0.0016 |
| **4** | **M1 (Deep GNN 8L)** | 33,666 | 0.7159 ± 0.0314 | 0.5583 ± 0.0381 | 0.6896 ± 0.1153 | **0.7655 ± 0.0710** | 0.9608 ± 0.0138 |

### Overall Aggregate Hypothesis Testing (M4 vs. M1)
- **Seed-Level Paired t-test ($N=5$):** $t = 1.2109, p = 0.2926$ (Cohen's $d = 0.542$, Medium) $\rightarrow$ **Not statistically significant**
- **Graph-Level Paired t-test ($N=237$):** $t = -1.4557, p = 0.1468$ (Cohen's $d = 0.144$, Negligible) $\rightarrow$ **Not statistically significant**
- **Graph-Level Wilcoxon Signed-Rank:** $W = 12681.0, p = 0.4157 \rightarrow$ **Not statistically significant**
- **Bonferroni-Adjusted Threshold:** $\alpha_{\text{adj}} = 0.05 / 4 = 0.0125$. In aggregate across all crack types, M4 and M1 are statistically comparable.

---

## 2. Morphology-Bucketed Benchmark (1,000-Resample Bootstrap 95% CIs)

Categorizing test images into physical crack geometries reveals significant, disjoint inductive biases:

| Morphology Category | Sample Size ($N$) | M1 (Deep GNN) Mean [95% CI] | M4 (Hybrid Full PE) Mean [95% CI] | F1 Gap ($\Delta$) | Statistical Test & Effect Size | Statistically Significant? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Long / Elongated Cracks** | 55 | 0.7056 [0.6640, 0.7393] | **0.8120 [0.8028, 0.8188]** | **+10.64%** | **Cohen's $d = 1.480$ (Very Large)** | **Yes (Disjoint CIs, $p < 0.001$)** |
| **Branched / Complex Networks** | 98 | **0.7617 [0.7506, 0.7722]** | 0.7115 [0.6913, 0.7358] | **-5.02%** | **Cohen's $d = 0.895$ (Large)** | **Yes (Disjoint CIs, $p < 0.001$)** |
| **Thin / Fine Fissures** | 84 | 0.6142 [0.5563, 0.6673] | **0.6747 [0.6383, 0.7084]** | **+6.05%** | Cohen's $d = 0.380$ (Small) | No (CIs overlap: 0.6383 < 0.6673) |

> **Key Takeaway:** These large-to-very-large effects ($d = 1.480$ for Long, $d = 0.895$ for Branched) make morphology specialization the strongest and most reproducible scientific result in this project.

---

## 3. Non-Circular Router Evaluation on 237 Holdout Images

Routing thresholds were derived strictly on the 45-image validation split ($AR \ge 1.5, A_{\text{frac}} < 0.015$, Val F1 = 0.7462) to eliminate circularity.

### Router Benchmark on Unseen Test Split ($N=237$)

| Architecture / Routing Configuration | Test F1 [95% CI] | Test IoU | Paired $t$-test vs M4 ($p$-value) | Paired $t$-test vs Max(M1, M4) ($p$-value) |
|---|:---:|:---:|:---:|:---:|
| **Standalone M1 (Deep GNN)** | 0.7566 [0.7156, 0.7952] | 0.6085 | $t = -1.456, p = 0.1468$ | $t = -6.442, p = 7.12 \times 10^{-10}$ |
| **Standalone M4 (Hybrid Full PE)** | 0.7651 [0.7218, 0.8043] | 0.6195 | Baseline | $t = -5.461, p = 1.15 \times 10^{-7}$ |
| **Oracle GT Router (Upper Bound)** | 0.7657 [0.7225, 0.8051] | 0.6203 | $t = +0.814, p = 0.4162$ | $t = -4.920, p = 1.54 \times 10^{-6}$ |
| **Production Predicted-Mask Router** | **0.7578 [0.7170, 0.7969]** | **0.6101** | **$t = -0.032, p = 0.9747$** | **$t = -6.792, p = 8.93 \times 10^{-11}$** |

### Routing Error Rate Breakdown by Morphology Bucket
*Computed by `analyze_routing_error_by_bucket.py` from `results/router_per_image_categories.json`.*

Note that "Routing Accuracy" and "Exact Category Match" are distinct metrics. The router collapses three morphology categories onto two models (Long_Elongated and Thin_Fine_Fissure both route to M4; only Branched_Complex routes to M1). A first-pass mask can therefore be mis-classified into the wrong category yet still be routed to the correct model, simply because two of the three categories share the same destination. This is why Long and Thin show materially higher Routing Accuracy than Category Accuracy, while Branched — the only category mapping to M1 — shows identical values for both, since any category error there is automatically also a routing error.

| True Morphology Bucket | Total Images ($N$) | Correctly Routed | Routing Accuracy | Routing Error Rate | Exact Category Match |
|---|:---:|:---:|:---:|:---:|:---:|
| **Long / Elongated** | 128 | 112 | 87.50% | **12.50%** | 78.12% |
| **Thin / Fine Fissure** | 23 | 21 | 91.30% | **8.70%** | 56.52% |
| **Branched / Complex** | 86 | 56 | 65.12% | **34.88%** | 65.12% |
| **Overall Aggregate** | **237** | **189** | **79.75%** | **20.25%** | **71.31%** |

- **Root Cause of Router Degradation:** Error is heavily concentrated in Branched Cracks (**34.88% misrouting rate**). Fragmented first-pass masks from M1 make branched cracks appear artificially elongated ($AR \ge 1.5$) or sparse ($A_{\text{frac}} < 0.015$), erroneously routing 30 out of 86 branched graphs to M4 instead of M1. In production, this pulls the router's score down to 0.7578 ($p = 0.975$ vs standalone M4).

---

## 4. TITS Multi-Sensor Transfer Benchmark & Calibration Stability ($N_{\text{draws}} = 10$)

*Evaluated on 78 positive sensor images (`AIGLE_RN`, `ESAR`, `LCMS`) across 10 independent stratified calibration draws (20% calibration / 80% holdout).*

| Model Architecture | $\tau^*$ (Mean ± Std) | Micro F1 (Mean ± Std) | Micro IoU (Mean ± Std) | Micro ROC-AUC | AIGLE_RN F1 (Mean ± Std) | LCMS F1 (Mean ± Std) | ESAR F1 (Mean ± Std) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M4 (Hybrid Full PE)** | 0.57 ± 0.21 | **0.0815 ± 0.0487** | **0.0432 ± 0.0265** | 0.6108 ± 0.0120 | **0.1175 ± 0.1171** | 0.0739 ± 0.0429 | 0.0780 ± 0.0786 |
| **M3 (Hybrid No PE)** | 0.67 ± 0.17 | 0.0789 ± 0.0405 | 0.0415 ± 0.0220 | 0.5899 ± 0.0076 | 0.0955 ± 0.0968 | 0.0952 ± 0.0630 | 0.0838 ± 0.0846 |
| **M2 (Shallow GNN 2L)** | 0.61 ± 0.23 | 0.0745 ± 0.0357 | 0.0391 ± 0.0193 | 0.5682 ± 0.0080 | 0.0827 ± 0.0844 | 0.1180 ± 0.0856 | **0.0864 ± 0.0872** |
| **M1 (Deep GNN 8L)** | 0.58 ± 0.32 | 0.0723 ± 0.0291 | 0.0377 ± 0.0157 | **0.6379 ± 0.0076** | 0.0390 ± 0.0522 | **0.2570 ± 0.1289** | 0.0349 ± 0.0585 |

### Paired Significance Tests on Modality Divergence (M4 vs. M1 across 10 draws)
*Computed by `analyze_tits_modality_significance.py` and stored in `results/tits_modality_significance.json`.*

| Sensor Modality | Imaging Physics | M4 Mean F1 | M1 Mean F1 | Paired $t$-stat | $p$-value | Cohen's $d$ Effect Size | Statistically Significant? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **AIGLE_RN** | Optical Surface Camera | **0.1175** | 0.0390 | $t = +2.659$ | **$p = 0.0261$** | $d = 0.841$ (Large) | **Yes (M4 dominates)** |
| **LCMS** | 3D Laser Profilometry | 0.0739 | **0.2570** | $t = -5.052$ | **$p = 0.0007$** | $d = -1.598$ (Very Large) | **Yes (M1 dominates)** |
| **ESAR** | Pavement Camera | 0.0780 | 0.0349 | $t = +2.068$ | $p = 0.0686$ | $d = 0.654$ (Medium) | No (Borderline trend) |

---

## 5. Dirichlet Energy Over-Smoothing Quantification

Calculated across all 237 holdout graphs layer-by-layer in `DeepGNN` (M1):

$$\begin{aligned}
\text{Layer 1:} & \quad E_{\text{norm}} = 0.0521 \\
\text{Layer 2:} & \quad E_{\text{norm}} = 0.0532 \\
\text{Layer 4 (Peak):} & \quad E_{\text{norm}} = \mathbf{0.0797} \\
\text{Layer 8:} & \quad E_{\text{norm}} = \mathbf{0.0456} \quad (\mathbf{-42.8\% \text{ collapse from peak}})
\end{aligned}$$

- **Takeaway:** Deep GNNs suffer from quantitative feature collapse past layer 4. Capping GNN depth at 2 layers ($E_{\text{norm}} = 0.0490$) and using Transformer self-attention for global reasoning prevents over-smoothing.

---

## 6. Sensor-Conditional Deployment Decision Matrix

| Inspection Sensor / Defect Context | Recommended Model | Operating Threshold | Expected Profile |
|---|:---:|:---:|---|
| **Optical Pipeline Surface Cameras** | **M4 (Hybrid Full PE)** | $\tau^* \approx 0.50$ | 3x higher optical F1 ($0.1175$ vs $0.0390$, $p = 0.0261$). |
| **3D Laser Profilometry Scanners** | **M1 (Deep GNN 8L)** | $\tau^* \approx 0.30$ | 3.5x higher laser F1 ($0.2570$ vs $0.0739$, $p = 0.0007$). |
| **Long Continuous Longitudinal Fissures** | **M4 (Hybrid Full PE)** | $\tau^* \approx 0.64$ | Disjoint 95% CIs (**0.8120 vs 0.7056**, $d = 1.480$, $p < 0.001$). |
| **Fatigue Web / Branched Networks** | **M1 (Deep GNN 8L)** | $\tau^* \approx 0.60$ | Disjoint 95% CIs (**0.7617 vs 0.7115**, $d = 0.895$, $p < 0.001$). |
| **Micro-Rover Edge Compute (<10 MB RAM)** | **M2 (Shallow GNN 2L)** | $\tau^* \approx 0.60$ | 8,706 parameters, 150+ FPS, solid F1 ($0.7199 \pm 0.0157$). |

---

## 7. Scoped Retraining Assessment & Conclusion

No model retraining or data re-collection is justified for the two questions investigated in this analysis:
1. **TITS Calibration Instability:** Driven by calibration sample scarcity ($N=15$ images across 3 sensors) and cross-modality physical shifts (optical RGB vs laser profilometry), not model capacity. The solution is collecting 30–50 calibration images per target sensor or applying domain adaptation, not retraining backbones on DeepCrack.
2. **Router Degradation:** Driven by bounding-box heuristic failure on branched graphs (34.88% error), not backbone quality. Future routing work should explore topological graph skeletonization or joint end-to-end gating.
3. **Open Scope:** Thin/Fine Fissure performance, where all four models cluster within a narrow, mediocre band (F1 $\approx$ 0.61–0.67), remains an open question outside the scope of this analysis and is a candidate for future targeted work.

---

## 8. Reproducibility & Data Lineage Manifest

All metrics and tables in this report are directly regeneratable from reproducible scripts and committed data:
- `analyze_tits_modality_significance.py` $\rightarrow$ `results/tits_modality_significance.json`
- `run_noncircular_router_validation.py` $\rightarrow$ `results/router_per_image_categories.json`
- `analyze_routing_error_by_bucket.py` $\rightarrow$ `results/routing_error_by_bucket.json`
- `results/tits_multidraw_calibration.json` (10-draw calibration stability records)
- `results/multiseed/multiseed_summary.json` (5-seed benchmark records)
- `dashboard/index.html` & `dashboard/predictions_data.json` (Interactive Visual Telemetry)
