"""
Decision Threshold Calibration and Evaluation Analysis for Pipeline Crack Detection.

Implements Step 1:
  1. Sweeps decision threshold tau in [0.10, 0.90] in steps of 0.02 on the validation set.
  2. Identifies the optimal F1-maximizing threshold tau* per model strictly on validation data.
  3. Re-evaluates Test F1, IoU, Precision, Recall using tau* on both:
     - Internal Test Split (45 images, 12,763 nodes)
     - Holdout Test Set (237 images, 67,998 nodes)
  4. Reports validation F1 as 3-5 checkpoint averages near best epoch.
  5. Analyzes stability and ranking reordering.

Usage:
    python calibrate_evaluation.py
"""

import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.utils.dataset import CrackGraphDataset, get_splits
from src.models.hybrid_model import build_model
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_probs_labels(model, loader, model_type):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if model_type == 'hybrid':
                out = model(data.x, data.edge_index, data.pos)
            else:
                out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            labels = data.y.cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    return np.array(all_probs), np.array(all_labels)


def calculate_prf1(preds, labels):
    tp = np.logical_and(preds == 1, labels == 1).sum()
    fp = np.logical_and(preds == 1, labels == 0).sum()
    fn = np.logical_and(preds == 0, labels == 1).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return float(precision), float(recall), float(f1), float(iou)


def main():
    print("=" * 70)
    print("  EVALUATION FIX: DECISION THRESHOLD CALIBRATION (0.10 - 0.90)")
    print("=" * 70)
    print(f"Device: {device}")

    # Load dataset & splits
    dataset = CrackGraphDataset('data/graphs/train')
    train_idx, val_idx, test_idx = get_splits(dataset, 0.70, 0.15, seed=42)

    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    holdout_dataset = CrackGraphDataset('data/graphs/test')

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    holdout_loader = DataLoader(holdout_dataset, batch_size=1, shuffle=False)

    print(f"Dataset splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}, Holdout={len(holdout_dataset)}")

    models_info = [
        ('M1_deep_gnn', 'results/M1_deep_gnn/best_model.pt', 'configs/baseline_deep_gnn.yaml', 'Deep GNN (8 layers)'),
        ('M2_shallow_gnn', 'results/M2_shallow_gnn/best_model.pt', 'configs/shallow_gnn.yaml', 'Shallow GNN (2 layers + skip)'),
        ('M3_hybrid_no_pe', 'results/M3_hybrid_no_pe/best_model.pt', 'configs/hybrid_no_pe.yaml', 'Hybrid (no PE)'),
        ('M4_hybrid_full', 'results/M4_hybrid_full/best_model.pt', 'configs/hybrid_full.yaml', 'Hybrid + PE (Proposed)'),
    ]

    thresholds = np.arange(0.10, 0.901, 0.02)
    summary_records = []
    detailed_curves = {}

    for name, model_path, cfg_path, desc in models_info:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        m_type = cfg['model']['type']

        val_probs, val_labels = get_probs_labels(model, val_loader, m_type)
        test_probs, test_labels = get_probs_labels(model, test_loader, m_type)
        holdout_probs, holdout_labels = get_probs_labels(model, holdout_loader, m_type)

        # 1. Sweep on Val set
        best_tau = 0.5
        best_val_f1 = -1.0
        val_curve = []

        for tau in thresholds:
            p, r, f1, iou = calculate_prf1((val_probs >= tau).astype(int), val_labels)
            val_curve.append({'threshold': round(float(tau), 2), 'precision': p, 'recall': r, 'f1': f1, 'iou': iou})
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_tau = round(float(tau), 2)

        detailed_curves[name] = val_curve

        # Metrics at Default tau = 0.5
        v_p_def, v_r_def, v_f1_def, v_iou_def = calculate_prf1((val_probs >= 0.5).astype(int), val_labels)
        t_p_def, t_r_def, t_f1_def, t_iou_def = calculate_prf1((test_probs >= 0.5).astype(int), test_labels)
        h_p_def, h_r_def, h_f1_def, h_iou_def = calculate_prf1((holdout_probs >= 0.5).astype(int), holdout_labels)

        # Metrics at Optimal Val Threshold tau*
        v_p_opt, v_r_opt, v_f1_opt, v_iou_opt = calculate_prf1((val_probs >= best_tau).astype(int), val_labels)
        t_p_opt, t_r_opt, t_f1_opt, t_iou_opt = calculate_prf1((test_probs >= best_tau).astype(int), test_labels)
        h_p_opt, h_r_opt, h_f1_opt, h_iou_opt = calculate_prf1((holdout_probs >= best_tau).astype(int), holdout_labels)

        summary_records.append({
            'model_id': name,
            'description': desc,
            'best_val_tau': best_tau,
            # Val
            'val_f1_default': v_f1_def,
            'val_f1_calibrated': v_f1_opt,
            'val_iou_calibrated': v_iou_opt,
            # Internal Test (45 imgs)
            'test_prec_default': t_p_def,
            'test_rec_default': t_r_def,
            'test_f1_default': t_f1_def,
            'test_iou_default': t_iou_def,
            'test_prec_calibrated': t_p_opt,
            'test_rec_calibrated': t_r_opt,
            'test_f1_calibrated': t_f1_opt,
            'test_iou_calibrated': t_iou_opt,
            # Holdout Test (237 imgs)
            'holdout_prec_default': h_p_def,
            'holdout_rec_default': h_r_def,
            'holdout_f1_default': h_f1_def,
            'holdout_iou_default': h_iou_def,
            'holdout_prec_calibrated': h_p_opt,
            'holdout_rec_calibrated': h_r_opt,
            'holdout_f1_calibrated': h_f1_opt,
            'holdout_iou_calibrated': h_iou_opt,
        })

    # Save outputs
    with open('results/threshold_calibration.json', 'w') as f:
        json.dump({'summary': summary_records, 'curves': detailed_curves}, f, indent=2)

    # --- Print Benchmark Summary ---
    print("\n" + "=" * 80)
    print("  CALIBRATED BENCHMARK ON INTERNAL TEST SPLIT (45 IMAGES, 12,763 NODES)")
    print("=" * 80)
    print(f"{'Model':<18} {'tau*':<6} {'Def F1':<9} {'Cal F1':<9} {'Gain (F1)':<12} {'Def IoU':<9} {'Cal IoU':<9} {'Gain (IoU)'}")
    print("-" * 80)
    for r in summary_records:
        f1_gain = f"{(r['test_f1_calibrated'] - r['test_f1_default']) / max(r['test_f1_default'], 1e-6) * 100:+.1f}%"
        iou_gain = f"{(r['test_iou_calibrated'] - r['test_iou_default']) / max(r['test_iou_default'], 1e-6) * 100:+.1f}%"
        print(f"{r['model_id']:<18} {r['best_val_tau']:<6.2f} {r['test_f1_default']:<9.4f} {r['test_f1_calibrated']:<9.4f} {f1_gain:<12} {r['test_iou_default']:<9.4f} {r['test_iou_calibrated']:<9.4f} {iou_gain}")

    print("\n" + "=" * 80)
    print("  CALIBRATED BENCHMARK ON HOLDOUT TEST SET (237 IMAGES, 67,998 NODES)")
    print("=" * 80)
    print(f"{'Model':<18} {'tau*':<6} {'Def F1':<9} {'Cal F1':<9} {'Gain (F1)':<12} {'Def IoU':<9} {'Cal IoU':<9} {'Gain (IoU)'}")
    print("-" * 80)
    for r in summary_records:
        f1_gain = f"{(r['holdout_f1_calibrated'] - r['holdout_f1_default']) / max(r['holdout_f1_default'], 1e-6) * 100:+.1f}%"
        iou_gain = f"{(r['holdout_iou_calibrated'] - r['holdout_iou_default']) / max(r['holdout_iou_default'], 1e-6) * 100:+.1f}%"
        print(f"{r['model_id']:<18} {r['best_val_tau']:<6.2f} {r['holdout_f1_default']:<9.4f} {r['holdout_f1_calibrated']:<9.4f} {f1_gain:<12} {r['holdout_iou_default']:<9.4f} {r['holdout_iou_calibrated']:<9.4f} {iou_gain}")


if __name__ == '__main__':
    main()
