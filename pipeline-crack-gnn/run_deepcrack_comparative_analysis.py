"""
Statistical Significance, Morphology Bucketing, and TITS Transfer Benchmark for DeepCrack vs M1-M4.

Implements Phase 2:
1. Multi-seed Master Leaderboard integration (M1, M2, M3, M4, DeepCrack).
2. Paired t-tests, Wilcoxon signed-rank tests, Cohen's d effect sizes, and Bonferroni correction.
3. 1,000-resample bootstrap 95% Confidence Intervals for morphology buckets.
4. TITS multi-sensor evaluation (10 stratified draws) for DeepCrack.
5. Master comparison summary export.
"""

import os
import sys
import glob
import json
import time
import cv2
import torch
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DCS_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEEPSEG_DIR = os.path.join(DCS_ROOT, "DeepSegmentor")
sys.path.insert(0, DEEPSEG_DIR)
sys.path.insert(0, BASE_DIR)

from models.deepcrack_networks import DeepCrackNet
from src.utils.metrics import compute_iou

MULTISEED_DIR = os.path.join(BASE_DIR, "results", "multiseed")
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TITS_BASE = r"C:\Users\ragha\Downloads\CrackDataset\TITS"
TITS_GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs", "tits")


def cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    n = len(diff)
    if n < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd < 1e-8:
        return 0.0
    return float(np.mean(diff) / sd)


def get_tits_sensor(fname):
    fn = os.path.basename(fname).lower()
    if 'aigle' in fn: return 'AIGLE_RN'
    if 'esar' in fn: return 'ESAR'
    if 'lcms' in fn: return 'LCMS'
    return 'Other'


def run_tits_deepcrack_eval(deepcrack_model, device, n_draws=10):
    """
    Evaluates DeepCrack on TITS multi-sensor dataset across 10 stratified calibration draws
    using the exact 78 positive sensor graphs matching M1-M4 benchmarks.
    """
    print("\n--- Running TITS Multi-Sensor Evaluation for DeepCrack (10 Draws) ---")
    sensors = ['AIGLE_RN', 'ESAR', 'LCMS']
    
    # Collect the 78 positive sensor graphs
    tits_graph_files = sorted([f for f in glob.glob(os.path.join(TITS_GRAPH_DIR, "*.pt")) if get_tits_sensor(f) in sensors])
    print(f"Found {len(tits_graph_files)} positive sensor graphs across {sensors}")
    
    all_items = []
    for gf in tits_graph_files:
        bname = os.path.splitext(os.path.basename(gf))[0]
        s = get_tits_sensor(gf)
        img_matches = glob.glob(os.path.join(TITS_BASE, "IMAGES", s, f"{bname}.*"))
        sp_path = os.path.join(BASE_DIR, "data", "superpixels", "tits", f"{bname}.npy")
        if img_matches and os.path.exists(sp_path):
            all_items.append({
                'id': bname,
                'sensor': s,
                'img_path': img_matches[0],
                'sp_path': sp_path,
                'graph_path': gf
            })
            
    print(f"Matched {len(all_items)} image-superpixel-graph items for TITS evaluation")
    
    # Compute node probabilities for all items once
    deepcrack_model.eval()
    cached_probs = {}
    cached_gts = {}
    
    with torch.no_grad():
        for item in all_items:
            img_bgr = cv2.imread(item['img_path'])
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]
            
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            _, _, _, _, _, fused = deepcrack_model(img_tensor)
            fused_up = F.interpolate(fused, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
            prob_map = torch.sigmoid(fused_up)[0, 0].cpu().numpy()
            
            # Superpixel pooling
            segments = np.load(item['sp_path'])
            graph_data = torch.load(item['graph_path'], weights_only=False)
            y_true = graph_data.y.cpu().numpy()
            num_nodes = graph_data.num_nodes
            
            node_probs = np.zeros(num_nodes, dtype=np.float32)
            for nid in range(num_nodes):
                mask_sp = (segments == nid)
                if mask_sp.any():
                    node_probs[nid] = prob_map[mask_sp].mean()
                    
            cached_probs[item['id']] = node_probs
            cached_gts[item['id']] = y_true
            
    # Run 10 stratified draws (20% calibration / 80% holdout)
    rng = np.random.default_rng(42)
    draw_results = []
    
    for draw in range(n_draws):
        calib_ids, eval_ids = [], []
        for s in sensors:
            s_pairs = [p['id'] for p in all_items if p['sensor'] == s]
            s_shuffled = rng.permutation(s_pairs).tolist()
            n_cal = max(1, int(len(s_shuffled) * 0.20))
            calib_ids.extend(s_shuffled[:n_cal])
            eval_ids.extend(s_shuffled[n_cal:])
            
        # Calibrate tau* on calibration split
        cal_probs = np.concatenate([cached_probs[pid] for pid in calib_ids])
        cal_labels = np.concatenate([cached_gts[pid] for pid in calib_ids])
        
        best_tau, best_cal_f1 = 0.50, 0.0
        for tau in np.arange(0.10, 0.92, 0.02):
            preds = (cal_probs >= tau).astype(int)
            f1 = f1_score(cal_labels, preds, zero_division=0)
            if f1 > best_cal_f1:
                best_cal_f1 = f1
                best_tau = float(tau)
                
        # Evaluate on holdout split
        eval_probs = np.concatenate([cached_probs[pid] for pid in eval_ids])
        eval_labels = np.concatenate([cached_gts[pid] for pid in eval_ids])
        eval_preds = (eval_probs >= best_tau).astype(int)
        
        micro_f1 = float(f1_score(eval_labels, eval_preds, zero_division=0))
        micro_iou = float(compute_iou(eval_preds, eval_labels))
        micro_auc = float(roc_auc_score(eval_labels, eval_probs)) if len(np.unique(eval_labels)) > 1 else 0.5
        
        # Per-sensor metrics
        sensor_f1s = {}
        for s in sensors:
            s_eval_ids = [p['id'] for p in all_items if p['sensor'] == s and p['id'] in eval_ids]
            if s_eval_ids:
                s_p = np.concatenate([cached_probs[pid] for pid in s_eval_ids])
                s_l = np.concatenate([cached_gts[pid] for pid in s_eval_ids])
                s_pred = (s_p >= best_tau).astype(int)
                sensor_f1s[s] = float(f1_score(s_l, s_pred, zero_division=0))
            else:
                sensor_f1s[s] = 0.0
                
        draw_results.append({
            'draw': draw + 1,
            'best_tau': best_tau,
            'micro_f1': micro_f1,
            'micro_iou': micro_iou,
            'micro_roc_auc': micro_auc,
            'sensor_f1s': sensor_f1s
        })
        
    tau_vals = [d['best_tau'] for d in draw_results]
    f1_vals = [d['micro_f1'] for d in draw_results]
    iou_vals = [d['micro_iou'] for d in draw_results]
    auc_vals = [d['micro_roc_auc'] for d in draw_results]
    aigle_f1s = [d['sensor_f1s']['AIGLE_RN'] for d in draw_results]
    lcms_f1s = [d['sensor_f1s']['LCMS'] for d in draw_results]
    esar_f1s = [d['sensor_f1s']['ESAR'] for d in draw_results]
    
    tits_summary = {
        'model_name': 'DeepCrack (Liu et al. 2019)',
        'tau_mean': float(np.mean(tau_vals)),
        'tau_std': float(np.std(tau_vals, ddof=1)),
        'micro_f1_mean': float(np.mean(f1_vals)),
        'micro_f1_std': float(np.std(f1_vals, ddof=1)),
        'micro_iou_mean': float(np.mean(iou_vals)),
        'micro_iou_std': float(np.std(iou_vals, ddof=1)),
        'micro_roc_auc_mean': float(np.mean(auc_vals)),
        'micro_roc_auc_std': float(np.std(auc_vals, ddof=1)),
        'aigle_rn_f1_mean': float(np.mean(aigle_f1s)),
        'aigle_rn_f1_std': float(np.std(aigle_f1s, ddof=1)),
        'lcms_f1_mean': float(np.mean(lcms_f1s)),
        'lcms_f1_std': float(np.std(lcms_f1s, ddof=1)),
        'esar_f1_mean': float(np.mean(esar_f1s)),
        'esar_f1_std': float(np.std(esar_f1s, ddof=1)),
        'draws': draw_results
    }
    print(f"DeepCrack TITS Micro F1: {np.mean(f1_vals):.4f} +/- {np.std(f1_vals, ddof=1):.4f} | AIGLE: {np.mean(aigle_f1s):.4f} | LCMS: {np.mean(lcms_f1s):.4f} | ESAR: {np.mean(esar_f1s):.4f}")
    return tits_summary


def main():
    print("="*80)
    print("PHASE 2: DEEPCRACK COMPARATIVE STATISTICAL ANALYSIS & MASTER INTEGRATION")
    print("="*80)
    
    # 1. Load GNN multiseed results
    gnn_summary_path = os.path.join(MULTISEED_DIR, "multiseed_summary.json")
    with open(gnn_summary_path, 'r') as f:
        gnn_summary = json.load(f)
        
    # 2. Load DeepCrack multiseed results
    dc_summary_path = os.path.join(MULTISEED_DIR, "deepcrack", "deepcrack_multiseed_summary.json")
    with open(dc_summary_path, 'r') as f:
        dc_summary = json.load(f)
        
    # Extract seed runs
    stats_data = gnn_summary.get('summary_stats', gnn_summary)
    m4_runs = stats_data['M4_hybrid_full']['seed_runs']
    m1_runs = stats_data['M1_deep_gnn']['seed_runs']
    m2_runs = stats_data['M2_shallow_gnn']['seed_runs']
    m3_runs = stats_data['M3_hybrid_no_pe']['seed_runs']
    dc_runs = dc_summary['seed_runs']
    
    dc_f1s = [r['holdout_f1'] for r in dc_runs]
    m4_f1s = [r['holdout_f1'] for r in m4_runs]
    m1_f1s = [r['holdout_f1'] for r in m1_runs]
    
    # Seed-level paired tests (N=5)
    t_dc_m4_seed, p_dc_m4_seed = stats.ttest_rel(dc_f1s, m4_f1s)
    d_dc_m4_seed = cohens_d(dc_f1s, m4_f1s)
    
    t_dc_m1_seed, p_dc_m1_seed = stats.ttest_rel(dc_f1s, m1_f1s)
    d_dc_m1_seed = cohens_d(dc_f1s, m1_f1s)
    
    # Graph-level paired tests (N=237 images)
    dc_graph_f1s_mean = np.mean([r['per_graph_f1s'] for r in dc_runs], axis=0)
    m4_graph_f1s_mean = np.mean([r['per_graph_f1s'] for r in m4_runs], axis=0)
    m1_graph_f1s_mean = np.mean([r['per_graph_f1s'] for r in m1_runs], axis=0)
    
    t_dc_m4_graph, p_dc_m4_graph = stats.ttest_rel(dc_graph_f1s_mean, m4_graph_f1s_mean)
    w_dc_m4_graph, p_w_dc_m4_graph = stats.wilcoxon(dc_graph_f1s_mean, m4_graph_f1s_mean)
    d_dc_m4_graph = cohens_d(dc_graph_f1s_mean, m4_graph_f1s_mean)
    
    t_dc_m1_graph, p_dc_m1_graph = stats.ttest_rel(dc_graph_f1s_mean, m1_graph_f1s_mean)
    w_dc_m1_graph, p_w_dc_m1_graph = stats.wilcoxon(dc_graph_f1s_mean, m1_graph_f1s_mean)
    d_dc_m1_graph = cohens_d(dc_graph_f1s_mean, m1_graph_f1s_mean)
    
    print("\n--- Hypothesis Testing (DeepCrack vs M4 and vs M1) ---")
    print(f"DeepCrack vs M4 (Seed-Level N=5):  t = {t_dc_m4_seed:.4f}, p = {p_dc_m4_seed:.4f} (d = {d_dc_m4_seed:.3f})")
    print(f"DeepCrack vs M1 (Seed-Level N=5):  t = {t_dc_m1_seed:.4f}, p = {p_dc_m1_seed:.4f} (d = {d_dc_m1_seed:.3f})")
    print(f"DeepCrack vs M4 (Graph-Level N=237): t = {t_dc_m4_graph:.4f}, p = {p_dc_m4_graph:.4e}, Wilcoxon p = {p_w_dc_m4_graph:.4e} (d = {d_dc_m4_graph:.3f})")
    print(f"DeepCrack vs M1 (Graph-Level N=237): t = {t_dc_m1_graph:.4f}, p = {p_dc_m1_graph:.4e}, Wilcoxon p = {p_w_dc_m1_graph:.4e} (d = {d_dc_m1_graph:.3f})")
    
    # 3. Morphology-Bucketed Comparison & 1,000-Resample Bootstrap 95% CIs
    print("\n--- Morphology Bucketed Comparison & 1,000-Resample Bootstrap CIs ---")
    buckets = ['Long_Elongated', 'Branched_Complex', 'Thin_Fine_Fissure']
    B = 1000
    rng = np.random.default_rng(42)
    bucket_comparison = {}
    
    for b in buckets:
        dc_b_vals = [r['bucket_metrics'][b]['f1'] for r in dc_runs]
        m4_b_vals = [r['bucket_metrics'][b]['f1'] for r in m4_runs]
        m1_b_vals = [r['bucket_metrics'][b]['f1'] for r in m1_runs]
        
        n_img = dc_runs[0]['bucket_metrics'][b].get('n_images', 0)
        
        # Bootstrap CIs
        dc_boot = [np.mean(rng.choice(dc_b_vals, size=len(dc_b_vals), replace=True)) for _ in range(B)]
        m4_boot = [np.mean(rng.choice(m4_b_vals, size=len(m4_b_vals), replace=True)) for _ in range(B)]
        m1_boot = [np.mean(rng.choice(m1_b_vals, size=len(m1_b_vals), replace=True)) for _ in range(B)]
        
        dc_ci = np.percentile(dc_boot, [2.5, 97.5]).tolist()
        m4_ci = np.percentile(m4_boot, [2.5, 97.5]).tolist()
        m1_ci = np.percentile(m1_boot, [2.5, 97.5]).tolist()
        
        d_vs_m4 = cohens_d(dc_b_vals, m4_b_vals)
        d_vs_m1 = cohens_d(dc_b_vals, m1_b_vals)
        
        bucket_comparison[b] = {
            'n_images': n_img,
            'deepcrack_mean': float(np.mean(dc_b_vals)),
            'deepcrack_ci_95': dc_ci,
            'm4_mean': float(np.mean(m4_b_vals)),
            'm4_ci_95': m4_ci,
            'm1_mean': float(np.mean(m1_b_vals)),
            'm1_ci_95': m1_ci,
            'gap_vs_m4': float(np.mean(dc_b_vals) - np.mean(m4_b_vals)),
            'gap_vs_m1': float(np.mean(dc_b_vals) - np.mean(m1_b_vals)),
            'cohens_d_vs_m4': d_vs_m4,
            'cohens_d_vs_m1': d_vs_m1
        }
        
        print(f"Bucket: {b:<20} (N={n_img}) | DeepCrack: {np.mean(dc_b_vals):.4f} [{dc_ci[0]:.4f}, {dc_ci[1]:.4f}] | M4: {np.mean(m4_b_vals):.4f} | M1: {np.mean(m1_b_vals):.4f}")
        
    # 4. TITS Multi-Sensor Evaluation for DeepCrack
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dc_best_model = DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch').to(device)
    dc_ckpt = os.path.join(MULTISEED_DIR, "deepcrack", "seed_42", "best_model.pt")
    dc_best_model.load_state_dict(torch.load(dc_ckpt, map_location=device))
    tits_summary = run_tits_deepcrack_eval(dc_best_model, device, n_draws=10)
    
    # 5. Master Comparison Data Object
    master_comp = {
        'leaderboard': {
            'M4_hybrid_full': stats_data['M4_hybrid_full'],
            'M3_hybrid_no_pe': stats_data['M3_hybrid_no_pe'],
            'M2_shallow_gnn': stats_data['M2_shallow_gnn'],
            'M1_deep_gnn': stats_data['M1_deep_gnn'],
            'DeepCrack_Liu2019': dc_summary
        },
        'significance_tests': {
            'deepcrack_vs_m4': {
                'seed_paired_t': {'t': t_dc_m4_seed, 'p': p_dc_m4_seed, 'cohens_d': d_dc_m4_seed},
                'graph_paired_t': {'t': t_dc_m4_graph, 'p': p_dc_m4_graph, 'cohens_d': d_dc_m4_graph},
                'graph_wilcoxon': {'w': w_dc_m4_graph, 'p': p_w_dc_m4_graph}
            },
            'deepcrack_vs_m1': {
                'seed_paired_t': {'t': t_dc_m1_seed, 'p': p_dc_m1_seed, 'cohens_d': d_dc_m1_seed},
                'graph_paired_t': {'t': t_dc_m1_graph, 'p': p_dc_m1_graph, 'cohens_d': d_dc_m1_graph},
                'graph_wilcoxon': {'w': w_dc_m1_graph, 'p': p_w_dc_m1_graph}
            },
            'bonferroni_threshold': 0.05 / 5.0
        },
        'morphology_buckets': bucket_comparison,
        'tits_transfer': tits_summary
    }
    
    out_master = os.path.join(BASE_DIR, "results", "deepcrack_comparison_master.json")
    with open(out_master, 'w') as f:
        json.dump(master_comp, f, indent=2)
    print(f"\nSaved master comparison to {out_master}")
    
    # 6. Generate Master Leaderboard Plot with Error Bars
    models_order = [
        ('M4 (Hybrid Full PE)', stats_data['M4_hybrid_full']),
        ('M3 (Hybrid No PE)', stats_data['M3_hybrid_no_pe']),
        ('M2 (Shallow GNN 2L)', stats_data['M2_shallow_gnn']),
        ('M1 (Deep GNN 8L)', stats_data['M1_deep_gnn']),
        ('DeepCrack (Liu 2019)', dc_summary)
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [m[0] for m in models_order]
    f1_m = [m[1]['f1_mean'] for m in models_order]
    f1_s = [m[1]['f1_std'] for m in models_order]
    iou_m = [m[1]['iou_mean'] for m in models_order]
    iou_s = [m[1]['iou_std'] for m in models_order]
    
    x = np.arange(len(names))
    w = 0.35
    
    ax.bar(x - w/2, f1_m, w, yerr=f1_s, capsize=5, label='Test F1 (Mean ± 1σ)', color='#2980b9', alpha=0.9)
    ax.bar(x + w/2, iou_m, w, yerr=iou_s, capsize=5, label='Test IoU (Mean ± 1σ)', color='#27ae60', alpha=0.9)
    
    for i in range(len(names)):
        ax.annotate(f"{f1_m[i]:.4f}", xy=(x[i] - w/2, f1_m[i] + f1_s[i] + 0.012), ha='center', fontsize=9, fontweight='bold')
        ax.annotate(f"{iou_m[i]:.4f}", xy=(x[i] + w/2, iou_m[i] + iou_s[i] + 0.012), ha='center', fontsize=9, fontweight='bold')
        
    ax.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')
    ax.set_title('Master Leaderboard (N=5 Seeds, 237 Holdout Images) — Including DeepCrack Baseline', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, "master_leaderboard_with_deepcrack.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved leaderboard plot to {plot_path}")
    
    # 7. Generate Morphology Buckets Plot with CIs
    fig, ax = plt.subplots(figsize=(11, 6))
    x_b = np.arange(len(buckets))
    w_b = 0.22
    
    b_m1 = [bucket_comparison[b]['m1_mean'] for b in buckets]
    b_m4 = [bucket_comparison[b]['m4_mean'] for b in buckets]
    b_dc = [bucket_comparison[b]['deepcrack_mean'] for b in buckets]
    
    # Error bars from 95% CIs
    err_m1 = [[b_m1[i] - bucket_comparison[b]['m1_ci_95'][0] for i, b in enumerate(buckets)],
              [bucket_comparison[b]['m1_ci_95'][1] - b_m1[i] for i, b in enumerate(buckets)]]
    err_m4 = [[b_m4[i] - bucket_comparison[b]['m4_ci_95'][0] for i, b in enumerate(buckets)],
              [bucket_comparison[b]['m4_ci_95'][1] - b_m4[i] for i, b in enumerate(buckets)]]
    err_dc = [[b_dc[i] - bucket_comparison[b]['deepcrack_ci_95'][0] for i, b in enumerate(buckets)],
              [bucket_comparison[b]['deepcrack_ci_95'][1] - b_dc[i] for i, b in enumerate(buckets)]]
              
    ax.bar(x_b - w_b, b_m1, w_b, yerr=err_m1, capsize=4, label='M1 (Deep GNN 8L)', color='#f39c12', alpha=0.9)
    ax.bar(x_b, b_m4, w_b, yerr=err_m4, capsize=4, label='M4 (Hybrid Full PE)', color='#2ecc71', alpha=0.9)
    ax.bar(x_b + w_b, b_dc, w_b, yerr=err_dc, capsize=4, label='DeepCrack (Pixel CNN)', color='#9b59b6', alpha=0.9)
    
    for i in range(len(buckets)):
        ax.annotate(f"{b_m1[i]:.3f}", xy=(x_b[i] - w_b, b_m1[i] + 0.03), ha='center', fontsize=8.5, fontweight='bold')
        ax.annotate(f"{b_m4[i]:.3f}", xy=(x_b[i], b_m4[i] + 0.03), ha='center', fontsize=8.5, fontweight='bold')
        ax.annotate(f"{b_dc[i]:.3f}", xy=(x_b[i] + w_b, b_dc[i] + 0.03), ha='center', fontsize=8.5, fontweight='bold')
        
    ax.set_ylabel('Holdout Test F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Morphology-Bucketed Comparison with 95% Bootstrap CIs', fontsize=13, fontweight='bold')
    ax.set_xticks(x_b)
    ax.set_xticklabels([b.replace('_', ' ') for b in buckets], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    
    bucket_plot_path = os.path.join(PLOTS_DIR, "morphology_buckets_with_deepcrack.png")
    plt.savefig(bucket_plot_path, dpi=300)
    plt.close()
    print(f"Saved morphology bucket plot to {bucket_plot_path}")


if __name__ == '__main__':
    main()
