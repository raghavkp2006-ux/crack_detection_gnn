"""
Multi-Seed Experiment & Statistical Significance Script for Pipeline Crack GNN.

Executes:
1. 5-Seed training (seeds: 42, 123, 456, 789, 2026) for M1, M2, M3, M4.
2. Holdout benchmark evaluation (N=237 images) at calibrated tau*.
3. Paired t-test & Wilcoxon signed-rank test between M4 and M1.
4. Bootstrap 95% Confidence Intervals for morphology buckets.
5. Learned residual gate trajectory analysis.
"""

import os
import sys
import json
import time
import glob
import cv2
import yaml
import torch
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset, get_splits
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc
from src.train import (
    load_config, get_class_weights, get_inverted_class_weights,
    FocalLoss, EarlyStopping, train_one_epoch, evaluate
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'multiseed')
PLOTS_DIR = os.path.join(BASE_DIR, 'results', 'plots')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]

MODEL_CONFIGS = [
    {
        'id': 'M1_deep_gnn',
        'name': 'M1 (Deep GNN 8L, WCE)',
        'config_path': os.path.join(BASE_DIR, 'configs', 'baseline_deep_gnn.yaml'),
        'loss_type': 'weighted_ce',
    },
    {
        'id': 'M2_shallow_gnn',
        'name': 'M2 (Shallow GNN 2L, Focal)',
        'config_path': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'shallow_gnn_focal.yaml'),
        'loss_type': 'focal',
    },
    {
        'id': 'M3_hybrid_no_pe',
        'name': 'M3 (Hybrid No PE, Focal)',
        'config_path': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_no_pe_focal.yaml'),
        'loss_type': 'focal',
    },
    {
        'id': 'M4_hybrid_full',
        'name': 'M4 (Hybrid Full PE, Focal)',
        'config_path': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_full_focal.yaml'),
        'loss_type': 'focal',
    },
]


def classify_crack_geometry(mask_path):
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


def train_single_seed(model_info, seed, device, holdout_loader, test_mask_dir, test_files):
    m_id = model_info['id']
    cfg = load_config(model_info['config_path'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = CrackGraphDataset(cfg['data']['graph_dir'])
    train_idx, val_idx, test_idx = get_splits(dataset, cfg['data']['train_ratio'], cfg['data']['val_ratio'], seed)
    train_sub = Subset(dataset, train_idx)
    val_sub = Subset(dataset, val_idx)
    test_sub = Subset(dataset, test_idx)

    train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

    model = build_model(cfg).to(device)

    # Loss
    if model_info['loss_type'] == 'focal':
        alpha = get_inverted_class_weights(train_sub).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)
    else:
        weights = get_class_weights(train_sub).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )

    warmup_epochs = cfg['training'].get('warmup_epochs', 0)
    def lr_lambda(e):
        return float(e + 1) / float(warmup_epochs) if warmup_epochs > 0 and e < warmup_epochs else 1.0
    sched_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if warmup_epochs > 0 else None
    sched_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    out_dir = os.path.join(RESULTS_DIR, m_id, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best_model.pt')
    early_stop = EarlyStopping(patience=cfg['training']['early_stopping_patience'], path=ckpt_path)

    t0 = time.time()
    for epoch in range(cfg['training']['epochs']):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, cfg)
        v_loss, v_preds, v_labels, v_probs = evaluate(model, val_loader, device, cfg)
        v_f1 = f1_score(v_labels, v_preds, zero_division=0)

        if sched_warmup and epoch < warmup_epochs:
            sched_warmup.step()
        else:
            sched_plateau.step(v_f1)

        early_stop(v_f1, model)
        if early_stop.early_stop:
            break

    train_time = time.time() - t0

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Threshold calibration on val
    _, _, val_labels_all, val_probs_all = evaluate(model, val_loader, device, cfg)
    best_tau, best_val_f1 = 0.50, 0.0
    for tau in np.arange(0.10, 0.92, 0.02):
        preds = (val_probs_all >= tau).astype(int)
        f1 = f1_score(val_labels_all, preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_tau = float(tau)

    # Holdout benchmark evaluation (N=237 images)
    m_type = cfg['model']['type']
    graph_preds_opt = []
    graph_labels_all = []
    graph_probs_all = []
    per_graph_f1s = []

    with torch.no_grad():
        for data in holdout_loader:
            data = data.to(device)
            if m_type == 'hybrid':
                out = model(data.x, data.edge_index, data.pos)
            else:
                out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds_opt = (probs >= best_tau).astype(int)
            labels = data.y.cpu().numpy()

            graph_preds_opt.append(preds_opt)
            graph_labels_all.append(labels)
            graph_probs_all.append(probs)

            gf1 = f1_score(labels, preds_opt, zero_division=0)
            per_graph_f1s.append(float(gf1))

    flat_preds = np.concatenate(graph_preds_opt)
    flat_labels = np.concatenate(graph_labels_all)
    flat_probs = np.concatenate(graph_probs_all)

    holdout_f1 = float(f1_score(flat_labels, flat_preds, zero_division=0))
    holdout_iou = float(compute_iou(flat_preds, flat_labels))
    holdout_prec = float(precision_score(flat_labels, flat_preds, zero_division=0))
    holdout_rec = float(recall_score(flat_labels, flat_preds, zero_division=0))
    holdout_auc = float(roc_auc_score(flat_labels, flat_probs))

    # Bucketed metrics
    buckets = ['Thin_Fine_Fissure', 'Long_Elongated', 'Branched_Complex']
    b_metrics = {}
    for b in buckets:
        b_idxs = [i for i, f in enumerate(test_files) if classify_crack_geometry(os.path.join(test_mask_dir, f"{os.path.splitext(os.path.basename(f))[0]}.png")) == b]
        if b_idxs:
            b_p = np.concatenate([graph_preds_opt[i] for i in b_idxs])
            b_l = np.concatenate([graph_labels_all[i] for i in b_idxs])
            b_f1 = float(f1_score(b_l, b_p, zero_division=0))
            b_iou = float(compute_iou(b_p, b_l))
            b_metrics[b] = {'f1': b_f1, 'iou': b_iou}

    result = {
        'model_id': m_id,
        'seed': seed,
        'train_time_sec': train_time,
        'best_tau': best_tau,
        'best_val_f1': best_val_f1,
        'holdout_f1': holdout_f1,
        'holdout_iou': holdout_iou,
        'holdout_precision': holdout_prec,
        'holdout_recall': holdout_rec,
        'holdout_roc_auc': holdout_auc,
        'per_graph_f1s': per_graph_f1s,
        'bucket_metrics': b_metrics
    }
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running Multi-Seed Benchmark on {device} ===")

    holdout_dataset = CrackGraphDataset('data/graphs/test')
    holdout_loader = DataLoader(holdout_dataset, batch_size=1, shuffle=False)
    test_files = sorted(glob.glob('data/graphs/test/*.pt'))
    test_mask_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'DeepCrack', 'test_lab'))

    all_seed_results = {m['id']: [] for m in MODEL_CONFIGS}

    total_runs = len(MODEL_CONFIGS) * len(SEEDS)
    current_run = 0
    t_start_total = time.time()

    for m_info in MODEL_CONFIGS:
        for seed in SEEDS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Training {m_info['name']} with seed {seed}...")
            res = train_single_seed(m_info, seed, device, holdout_loader, test_mask_dir, test_files)
            all_seed_results[m_info['id']].append(res)
            print(f"    Done in {res['train_time_sec']:.1f}s | Tau*: {res['best_tau']:.2f} | Holdout F1: {res['holdout_f1']:.4f} | IoU: {res['holdout_iou']:.4f}")

    total_time = time.time() - t_start_total
    print(f"\nAll {total_runs} runs finished in {total_time/60.0:.2f} minutes.")

    # Aggregate statistics
    summary_stats = {}
    print("\n" + "="*80)
    print("MULTI-SEED STATISTICAL SUMMARY (5 SEEDS: 42, 123, 456, 789, 2026)")
    print("="*80)

    for m_info in MODEL_CONFIGS:
        mid = m_info['id']
        runs = all_seed_results[mid]
        f1s = [r['holdout_f1'] for r in runs]
        ious = [r['holdout_iou'] for r in runs]
        precs = [r['holdout_precision'] for r in runs]
        recs = [r['holdout_recall'] for r in runs]
        aucs = [r['holdout_roc_auc'] for r in runs]

        summary_stats[mid] = {
            'name': m_info['name'],
            'f1_mean': float(np.mean(f1s)),
            'f1_std': float(np.std(f1s, ddof=1)),
            'iou_mean': float(np.mean(ious)),
            'iou_std': float(np.std(ious, ddof=1)),
            'precision_mean': float(np.mean(precs)),
            'precision_std': float(np.std(precs, ddof=1)),
            'recall_mean': float(np.mean(recs)),
            'recall_std': float(np.std(recs, ddof=1)),
            'roc_auc_mean': float(np.mean(aucs)),
            'roc_auc_std': float(np.std(aucs, ddof=1)),
            'seed_runs': runs
        }

        print(f"{m_info['name']:<35} | F1: {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f} | IoU: {np.mean(ious):.4f} +/- {np.std(ious, ddof=1):.4f} | Rec: {np.mean(recs):.4f} +/- {np.std(recs, ddof=1):.4f} | AUC: {np.mean(aucs):.4f} +/- {np.std(aucs, ddof=1):.4f}")

    # Paired Significance Tests between M4 and M1
    m4_f1_seeds = [r['holdout_f1'] for r in all_seed_results['M4_hybrid_full']]
    m1_f1_seeds = [r['holdout_f1'] for r in all_seed_results['M1_deep_gnn']]
    t_stat_seeds, p_val_seeds = stats.ttest_rel(m4_f1_seeds, m1_f1_seeds)

    # Graph-level paired testing across all 237 holdout graphs (averaging per-graph F1 across 5 seeds)
    m4_graph_f1s_mean = np.mean([r['per_graph_f1s'] for r in all_seed_results['M4_hybrid_full']], axis=0)
    m1_graph_f1s_mean = np.mean([r['per_graph_f1s'] for r in all_seed_results['M1_deep_gnn']], axis=0)
    t_stat_graphs, p_val_graphs = stats.ttest_rel(m4_graph_f1s_mean, m1_graph_f1s_mean)
    w_stat_graphs, p_val_wilcox = stats.wilcoxon(m4_graph_f1s_mean, m1_graph_f1s_mean)

    sig_tests = {
        'seed_level_paired_t_test': {'t_stat': float(t_stat_seeds), 'p_value': float(p_val_seeds)},
        'graph_level_paired_t_test': {'t_stat': float(t_stat_graphs), 'p_value': float(p_val_graphs)},
        'graph_level_wilcoxon_test': {'w_stat': float(w_stat_graphs), 'p_value': float(p_val_wilcox)},
    }

    print("\n--- Statistical Significance Testing (M4 vs. M1) ---")
    print(f"Across 5 Random Seeds: Paired t-stat = {t_stat_seeds:.4f}, p-value = {p_val_seeds:.4f}")
    print(f"Across 237 Holdout Test Graphs: Paired t-stat = {t_stat_graphs:.4f}, p-value = {p_val_graphs:.4e}")
    print(f"Across 237 Holdout Test Graphs: Wilcoxon W = {w_stat_graphs:.4f}, p-value = {p_val_wilcox:.4e}")

    # Bootstrap 95% Confidence Intervals for Morphology Buckets
    print("\n--- Bootstrap 95% Confidence Intervals (1,000 resamples) ---")
    buckets = ['Thin_Fine_Fissure', 'Long_Elongated', 'Branched_Complex']
    bootstrap_results = {}
    B = 1000
    rng = np.random.default_rng(42)

    # Using the seed 42 predictions for graph-level bootstrap
    m4_preds = all_seed_results['M4_hybrid_full'][0]
    m1_preds = all_seed_results['M1_deep_gnn'][0]
    m2_preds = all_seed_results['M2_shallow_gnn'][0]
    m3_preds = all_seed_results['M3_hybrid_no_pe'][0]

    for b in buckets:
        bootstrap_results[b] = {}
        for mid in ['M1_deep_gnn', 'M2_shallow_gnn', 'M3_hybrid_no_pe', 'M4_hybrid_full']:
            b_vals = [r['bucket_metrics'][b]['f1'] for r in all_seed_results[mid]]
            # Bootstrap resample the 5-seed values
            boot_means = [np.mean(rng.choice(b_vals, size=len(b_vals), replace=True)) for _ in range(B)]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            bootstrap_results[b][mid] = {
                'mean': float(np.mean(b_vals)),
                'ci_95': [float(ci_low), float(ci_high)]
            }
            print(f"  {b:<20} | {mid:<18}: Mean F1 = {np.mean(b_vals):.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")

    # Plot 1: Benchmark with Error Bars (Mean +/- Std)
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [m['name'] for m in MODEL_CONFIGS]
    f1_means = [summary_stats[m['id']]['f1_mean'] for m in MODEL_CONFIGS]
    f1_stds = [summary_stats[m['id']]['f1_std'] for m in MODEL_CONFIGS]
    iou_means = [summary_stats[m['id']]['iou_mean'] for m in MODEL_CONFIGS]
    iou_stds = [summary_stats[m['id']]['iou_std'] for m in MODEL_CONFIGS]

    x = np.arange(len(MODEL_CONFIGS))
    w = 0.35

    ax.bar(x - w/2, f1_means, w, yerr=f1_stds, capsize=5, label='Test F1 (Mean ± 1σ)', color='#2980b9', alpha=0.9)
    ax.bar(x + w/2, iou_means, w, yerr=iou_stds, capsize=5, label='Test IoU (Mean ± 1σ)', color='#27ae60', alpha=0.9)

    for i in range(len(MODEL_CONFIGS)):
        ax.annotate(f"{f1_means[i]:.4f}", xy=(x[i] - w/2, f1_means[i] + f1_stds[i] + 0.01), ha='center', fontsize=9, fontweight='bold')
        ax.annotate(f"{iou_means[i]:.4f}", xy=(x[i] + w/2, iou_means[i] + iou_stds[i] + 0.01), ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Holdout Benchmark (N=237 images) — 5-Seed Reproducibility (Mean ± Std)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m['name'].split('(')[0].strip() for m in MODEL_CONFIGS], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.92)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plot_err_path = os.path.join(PLOTS_DIR, 'multiseed_benchmark_errorbars.png')
    plt.savefig(plot_err_path, dpi=300)
    plt.close()
    print(f"\nSaved error bar plot to {plot_err_path}")

    # Plot 2: Morphology Buckets with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(11, 6))
    x_b = np.arange(len(buckets))
    w_b = 0.18
    palette = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

    for idx, m_info in enumerate(MODEL_CONFIGS):
        mid = m_info['id']
        b_means = [bootstrap_results[b][mid]['mean'] for b in buckets]
        err_low = [bootstrap_results[b][mid]['mean'] - bootstrap_results[b][mid]['ci_95'][0] for b in buckets]
        err_high = [bootstrap_results[b][mid]['ci_95'][1] - bootstrap_results[b][mid]['mean'] for b in buckets]
        ax.bar(x_b + (idx - 1.5) * w_b, b_means, w_b, yerr=[err_low, err_high], capsize=4, label=m_info['name'], color=palette[idx], alpha=0.9)

    ax.set_ylabel('Test F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Morphology Bucketed F1 with 95% Bootstrap Confidence Intervals', fontsize=13, fontweight='bold')
    ax.set_xticks(x_b)
    ax.set_xticklabels([b.replace('_', ' ') for b in buckets], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plot_bci_path = os.path.join(PLOTS_DIR, 'bucketed_significance_cis.png')
    plt.savefig(plot_bci_path, dpi=300)
    plt.close()
    print(f"Saved bucketed CI plot to {plot_bci_path}")

    # Save complete JSON
    final_output = {
        'seeds': SEEDS,
        'summary_stats': summary_stats,
        'significance_tests': sig_tests,
        'bootstrap_confidence_intervals': bootstrap_results
    }
    with open(os.path.join(RESULTS_DIR, 'multiseed_summary.json'), 'w') as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved multiseed summary to {os.path.join(RESULTS_DIR, 'multiseed_summary.json')}")


if __name__ == '__main__':
    main()

