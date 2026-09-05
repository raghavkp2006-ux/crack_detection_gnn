"""
Comprehensive Evaluation, Step 5 Bucketed Analysis, and Step 6 Dirichlet Energy Script.

Evaluates all models:
  - M1: Deep GNN (8 layers)
  - M2: Shallow GNN (Focal)
  - M3: Hybrid No PE (Focal)
  - M4: Hybrid Full (Focal)
  - Step 3: M4 + Learned Residual Gate
  - Step 4: M4 + Relative Positional Bias + Residual Gate
"""

import os
import sys
import glob
import json
import time
import cv2
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset, get_splits
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc, dirichlet_energy
from src.train import load_config, evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

MODELS = [
    {
        'id': 'M1_deep_gnn',
        'name': 'M1 (Deep GNN 8L, WCE)',
        'config': os.path.join(BASE_DIR, 'configs', 'baseline_deep_gnn.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'M1_deep_gnn', 'best_model.pt'),
    },
    {
        'id': 'M2_shallow_gnn',
        'name': 'M2 (Shallow GNN 2L, Focal)',
        'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'shallow_gnn_focal.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M2_shallow_gnn', 'best_model.pt'),
    },
    {
        'id': 'M3_hybrid_no_pe',
        'name': 'M3 (Hybrid No PE, Focal)',
        'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_no_pe_focal.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M3_hybrid_no_pe', 'best_model.pt'),
    },
    {
        'id': 'M4_hybrid_full',
        'name': 'M4 (Hybrid Full PE, Focal)',
        'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_full_focal.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M4_hybrid_full', 'best_model.pt'),
    },
    {
        'id': 'M4_hybrid_gate',
        'name': 'Step 3: M4 + Gate (Focal)',
        'config': os.path.join(BASE_DIR, 'configs', 'step3_m4_gate.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step3_gate', 'M4_hybrid_gate', 'best_model.pt'),
    },
    {
        'id': 'M4_hybrid_relpos',
        'name': 'Step 4: M4 + RelPos + Gate (Focal)',
        'config': os.path.join(BASE_DIR, 'configs', 'step4_m4_relpos.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step4_relpos', 'M4_hybrid_relpos', 'best_model.pt'),
    },
]


def load_model_and_config(info, device):
    config = load_config(info['config'])
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(info['checkpoint'], map_location=device))
    model.eval()
    return model, config


def run_threshold_sweep(model, loader, device, config):
    _, _, val_labels, val_probs = evaluate(model, loader, device, config)
    best_tau, best_f1 = 0.50, 0.0
    for tau in np.arange(0.10, 0.92, 0.02):
        preds = (val_probs >= tau).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return float(best_tau), float(best_f1)


def get_predictions_per_graph(model, loader, device, config, tau):
    model_type = config['model']['type']
    graph_preds, graph_labels, graph_probs = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if model_type == 'hybrid':
                out = model(data.x, data.edge_index, data.pos)
            else:
                out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= tau).astype(int)
            labels = data.y.cpu().numpy()
            graph_preds.append(preds)
            graph_labels.append(labels)
            graph_probs.append(probs)
    return graph_preds, graph_labels, graph_probs


def classify_crack_geometry(mask_path):
    """
    Classifies crack morphology into:
      - Thin_Fine_Fissure: Area fraction < 0.025 (hardest to detect, narrow fissures)
      - Long_Elongated: Aspect ratio >= 2.0 (long continuous diagonal/longitudinal cracks)
      - Branched_Complex: Rest (interconnected multi-directional crack networks)
    """
    if not os.path.exists(mask_path):
        return 'Unknown'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 'Unknown'
    pts = np.argwhere(mask > 127)
    if len(pts) == 0:
        return 'Clean_Negative'
    
    H, W = mask.shape
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    area_frac = len(pts) / (H * W)
    
    if area_frac < 0.025:
        return 'Thin_Fine_Fissure'
    elif aspect_ratio >= 2.0:
        return 'Long_Elongated'
    else:
        return 'Branched_Complex'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on: {device}")
    
    # 1. Datasets
    train_dataset = CrackGraphDataset('data/graphs/train')
    train_idx, val_idx, test_idx = get_splits(train_dataset, 0.7, 0.15, 42)
    val_subset = Subset(train_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
    
    holdout_dataset = CrackGraphDataset('data/graphs/test')
    holdout_loader = DataLoader(holdout_dataset, batch_size=1, shuffle=False)
    
    # Map holdout graph files to masks in ../DeepCrack/test_lab
    test_files = sorted(glob.glob('data/graphs/test/*.pt'))
    test_mask_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'DeepCrack', 'test_lab'))
    
    crack_buckets = []
    for f in test_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        mask_candidate = os.path.join(test_mask_dir, f"{basename}.png")
        if not os.path.exists(mask_candidate):
            mask_candidate = os.path.join(test_mask_dir, f"{basename}.jpg")
        bucket = classify_crack_geometry(mask_candidate)
        crack_buckets.append(bucket)
        
    bucket_counts = {b: crack_buckets.count(b) for b in set(crack_buckets)}
    print(f"Holdout Benchmark Bucket Distribution (237 images): {bucket_counts}")
    
    # 2. Evaluate all models
    all_results = {}
    model_predictions = {}
    
    for info in MODELS:
        m_id = info['id']
        m_name = info['name']
        print(f"\nProcessing {m_name}...")
        model, config = load_model_and_config(info, device)
        
        # Calibrate tau on validation set
        opt_tau, val_f1 = run_threshold_sweep(model, val_loader, device, config)
        
        # Holdout inference at 0.50 and opt_tau
        preds_05, labels_all, probs_all = get_predictions_per_graph(model, holdout_loader, device, config, 0.50)
        preds_opt, _, _ = get_predictions_per_graph(model, holdout_loader, device, config, opt_tau)
        
        flat_labels = np.concatenate(labels_all)
        flat_probs = np.concatenate(probs_all)
        flat_preds_05 = np.concatenate(preds_05)
        flat_preds_opt = np.concatenate(preds_opt)
        
        f1_05 = f1_score(flat_labels, flat_preds_05, zero_division=0)
        iou_05 = compute_iou(flat_preds_05, flat_labels)
        rec_05 = recall_score(flat_labels, flat_preds_05, zero_division=0)
        prec_05 = precision_score(flat_labels, flat_preds_05, zero_division=0)
        
        f1_opt = f1_score(flat_labels, flat_preds_opt, zero_division=0)
        iou_opt = compute_iou(flat_preds_opt, flat_labels)
        rec_opt = recall_score(flat_labels, flat_preds_opt, zero_division=0)
        prec_opt = precision_score(flat_labels, flat_preds_opt, zero_division=0)
        auc = roc_auc_score(flat_labels, flat_probs)
        
        model_predictions[m_id] = {
            'preds_opt': preds_opt,
            'labels': labels_all,
            'probs': probs_all
        }
        
        all_results[m_id] = {
            'name': m_name,
            'optimal_tau': opt_tau,
            'val_f1': val_f1,
            'holdout_05': {'f1': float(f1_05), 'iou': float(iou_05), 'recall': float(rec_05), 'precision': float(prec_05)},
            'holdout_opt': {'f1': float(f1_opt), 'iou': float(iou_opt), 'recall': float(rec_opt), 'precision': float(prec_opt), 'roc_auc': float(auc)},
            'gate_transformer': float(model.get_gate_weight()) if hasattr(model, 'get_gate_weight') and model.get_gate_weight() is not None else None,
            'temperatures': [float(t) for t in model.get_temperatures()] if hasattr(model, 'get_temperatures') and model.get_temperatures() else None,
        }
        print(f"  Optimal Tau: {opt_tau:.2f} | Holdout @ 0.50: F1={f1_05:.4f}, IoU={iou_05:.4f} | Holdout @ {opt_tau:.2f}: F1={f1_opt:.4f}, IoU={iou_opt:.4f}")

    # =========================================================================
    # STEP 5: BUCKETED ANALYSIS (Thin/Fine, Long/Elongated, Branched/Complex)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: BUCKETED ANALYSIS ON HOLDOUT TEST SET")
    print("=" * 70)
    
    buckets_to_evaluate = ['Thin_Fine_Fissure', 'Long_Elongated', 'Branched_Complex']
    bucket_results = {b: {} for b in buckets_to_evaluate}
    
    for b in buckets_to_evaluate:
        b_indices = [idx for idx, b_name in enumerate(crack_buckets) if b_name == b]
        print(f"\n--- Bucket: {b} (N={len(b_indices)} images) ---")
        if len(b_indices) == 0:
            continue
            
        for info in MODELS:
            m_id = info['id']
            preds_opt = model_predictions[m_id]['preds_opt']
            labels_all = model_predictions[m_id]['labels']
            
            b_preds = np.concatenate([preds_opt[i] for i in b_indices])
            b_labels = np.concatenate([labels_all[i] for i in b_indices])
            
            f1 = f1_score(b_labels, b_preds, zero_division=0)
            iou = compute_iou(b_preds, b_labels)
            rec = recall_score(b_labels, b_preds, zero_division=0)
            prec = precision_score(b_labels, b_preds, zero_division=0)
            
            bucket_results[b][m_id] = {
                'f1': float(f1), 'iou': float(iou),
                'recall': float(rec), 'precision': float(prec)
            }
            print(f"  {info['name']:<36}: F1 = {f1:.4f} | IoU = {iou:.4f} | Recall = {rec:.4f}")
            
    # Save Bucketed Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(buckets_to_evaluate))
    width = 0.13
    
    palette = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c']
    for idx, info in enumerate(MODELS):
        m_id = info['id']
        f1_vals = [bucket_results[b].get(m_id, {}).get('f1', 0.0) for b in buckets_to_evaluate]
        ax.bar(x + (idx - 2.5) * width, f1_vals, width, label=info['name'], color=palette[idx], alpha=0.9)
        
    ax.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Step 5: Bucketed Performance Across Crack Morphologies (Holdout Test N=237)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('_', ' ')}\n(N={bucket_counts.get(b,0)})" for b in buckets_to_evaluate], fontsize=11, fontweight='bold')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    bucket_plot_path = os.path.join(PLOTS_DIR, 'bucketed_analysis.png')
    plt.savefig(bucket_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved bucketed plot to {bucket_plot_path}")

    # =========================================================================
    # STEP 6: DIRICHLET ENERGY OVER-SMOOTHING ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: DIRICHLET ENERGY OVER-SMOOTHING ANALYSIS")
    print("=" * 70)
    
    # Analyze Dirichlet Energy across layers for M1 (8 layers), M2 (2 layers), M4 (2 GNN layers)
    # E(H) = (1/|E|) sum_{(i,j)} ||h_i - h_j||^2 / (||h_i|| * ||h_j||)
    
    models_to_test = [
        ('M1 (8-Layer Deep GNN)', MODELS[0]),
        ('M2 (2-Layer Shallow GNN)', MODELS[1]),
        ('M4 (Hybrid GNN+Trans)', MODELS[3]),
    ]
    
    dirichlet_summary = {}
    
    for m_label, info in models_to_test:
        model, config = load_model_and_config(info, device)
        raw_energies_per_layer = []
        norm_energies_per_layer = []
        
        with torch.no_grad():
            for data in holdout_loader:
                data = data.to(device)
                if config['model']['type'] == 'hybrid':
                    _ = model(data.x, data.edge_index, data.pos)
                    layer_outputs = getattr(model, 'gnn_layer_outputs', [])
                else:
                    _ = model(data.x, data.edge_index)
                    layer_outputs = getattr(model, 'layer_outputs', [])
                    
                if not layer_outputs:
                    continue
                    
                src, dst = data.edge_index
                if src.numel() == 0:
                    continue
                    
                graph_raw = []
                graph_norm = []
                for h in layer_outputs:
                    # Raw Dirichlet Energy
                    diff = h[src] - h[dst]
                    raw_e = (diff.norm(dim=1) ** 2).mean().item()
                    graph_raw.append(raw_e)
                    
                    # Normalized Dirichlet Energy
                    h_norm = torch.nn.functional.normalize(h, p=2, dim=1)
                    diff_norm = h_norm[src] - h_norm[dst]
                    norm_e = (diff_norm.norm(dim=1) ** 2).mean().item()
                    graph_norm.append(norm_e)
                    
                raw_energies_per_layer.append(graph_raw)
                norm_energies_per_layer.append(graph_norm)
                
        avg_raw = np.mean(raw_energies_per_layer, axis=0).tolist()
        avg_norm = np.mean(norm_energies_per_layer, axis=0).tolist()
        dirichlet_summary[m_label] = {
            'raw_energy': avg_raw,
            'normalized_energy': avg_norm
        }
        print(f"{m_label}: Normalized Dirichlet Energy per layer = {[round(x, 4) for x in avg_norm]}")
        
    # Plot Dirichlet Energy Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Normalized Dirichlet Energy
    for m_label, data in dirichlet_summary.items():
        norm_e = data['normalized_energy']
        layers = list(range(1, len(norm_e) + 1))
        marker = 'o' if 'Deep' in m_label else 's'
        color = '#e74c3c' if 'Deep' in m_label else ('#3498db' if 'Shallow' in m_label else '#2ecc71')
        ax1.plot(layers, norm_e, marker=marker, linewidth=2.5, markersize=8, label=m_label, color=color)
        
    ax1.set_xlabel('GNN Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Dirichlet Energy E(H)', fontsize=12, fontweight='bold')
    ax1.set_title('Normalized Dirichlet Energy (Over-Smoothing Test)', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(1, 9))
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Raw Dirichlet Energy
    for m_label, data in dirichlet_summary.items():
        raw_e = data['raw_energy']
        layers = list(range(1, len(raw_e) + 1))
        marker = 'o' if 'Deep' in m_label else 's'
        color = '#e74c3c' if 'Deep' in m_label else ('#3498db' if 'Shallow' in m_label else '#2ecc71')
        ax2.plot(layers, raw_e, marker=marker, linewidth=2.5, markersize=8, label=m_label, color=color)
        
    ax2.set_xlabel('GNN Layer Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Raw Dirichlet Energy', fontsize=12, fontweight='bold')
    ax2.set_title('Raw Dirichlet Energy Across Layers', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, 9))
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    dirichlet_plot_path = os.path.join(PLOTS_DIR, 'dirichlet_energy_curves.png')
    plt.savefig(dirichlet_plot_path, dpi=300)
    plt.close()
    print(f"Saved Dirichlet energy curves to {dirichlet_plot_path}")

    # =========================================================================
    # OVERALL BENCHMARK COMPARISON PLOT
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [info['name'] for info in MODELS]
    f1_05_list = [all_results[info['id']]['holdout_05']['f1'] for info in MODELS]
    f1_opt_list = [all_results[info['id']]['holdout_opt']['f1'] for info in MODELS]
    iou_opt_list = [all_results[info['id']]['holdout_opt']['iou'] for info in MODELS]
    
    indices = np.arange(len(MODELS))
    w = 0.28
    
    b1 = ax.bar(indices - w, f1_05_list, w, label='Holdout F1 @ tau=0.50', color='#95a5a6', alpha=0.85)
    b2 = ax.bar(indices, f1_opt_list, w, label='Holdout F1 @ Calibrated tau*', color='#2980b9', alpha=0.95)
    b3 = ax.bar(indices + w, iou_opt_list, w, label='Holdout IoU @ Calibrated tau*', color='#27ae60', alpha=0.95)
    
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
                    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Holdout Benchmark: Steps 1-4 Empirical Comparison (N=237)', fontsize=14, fontweight='bold')
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    bench_plot_path = os.path.join(PLOTS_DIR, 'step4_complete_benchmark.png')
    plt.savefig(bench_plot_path, dpi=300)
    plt.close()
    print(f"Saved complete benchmark plot to {bench_plot_path}")
    
    # Save complete JSON
    final_output = {
        'all_results': all_results,
        'bucket_results': bucket_results,
        'dirichlet_summary': dirichlet_summary,
        'bucket_counts': bucket_counts
    }
    summary_path = os.path.join(BASE_DIR, 'results', 'full_evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"\nAll results saved to {summary_path}!")

if __name__ == '__main__':
    main()

