"""
Step 2 Experiment: Add class-imbalance handling to the loss (Focal Loss gamma=2).

Retrains M2 (Shallow GNN), M3 (Hybrid no PE), and M4 (Hybrid + PE) with Focal Loss (gamma=2.0)
holding all architecture parameters, splits, and seeds constant.
Compares:
  - Step 1 (Standard Weighted CE) vs. Step 2 (Focal Loss gamma=2)
  - Evaluates both at default threshold (0.50) and calibrated threshold (tau*)
  - Quantifies how much of the F1/recall gap closes to isolate loss vs. architecture effects.

Usage:
    python run_step2_experiment.py
"""

import os
import sys
import time
import json
import yaml
import subprocess
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "configs", "step2_loss")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "step2_loss")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def banner(msg: str):
    width = 70
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def create_step2_configs():
    """Create exact duplicate configs of M2, M3, M4 with loss: focal and focal_gamma: 2.0."""
    base_configs = [
        ("M2_shallow_gnn", "shallow_gnn.yaml", "shallow_gnn_focal.yaml"),
        ("M3_hybrid_no_pe", "hybrid_no_pe.yaml", "hybrid_no_pe_focal.yaml"),
        ("M4_hybrid_full", "hybrid_full.yaml", "hybrid_full_focal.yaml"),
    ]
    created = []
    for model_id, src_cfg, tgt_cfg in base_configs:
        src_path = os.path.join(BASE_DIR, "configs", src_cfg)
        with open(src_path) as f:
            cfg = yaml.safe_load(f)
        
        # Add focal loss
        cfg['training']['loss'] = 'focal'
        cfg['training']['focal_gamma'] = 2.0
        cfg['output_dir'] = f"results/step2_loss/{model_id}"
        
        tgt_path = os.path.join(CONFIG_DIR, tgt_cfg)
        with open(tgt_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        created.append((model_id, tgt_path))
    return created


def train_model(model_id, config_path):
    banner(f"TRAINING {model_id} WITH FOCAL LOSS (gamma=2.0)")
    cmd = [sys.executable, "-m", "src.train", "--config", config_path]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"[ERROR] Training {model_id} failed with code {res.returncode}")
        return None
    print(f"Training completed successfully in {elapsed:.1f}s.")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    res_path = os.path.join(BASE_DIR, cfg['output_dir'], 'results.json')
    if os.path.exists(res_path):
        with open(res_path) as f:
            return json.load(f)
    return None


def run_threshold_sweep_and_eval(model_records):
    """Sweeps threshold on val set for newly trained models and evaluates test sets."""
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from src.utils.dataset import CrackGraphDataset, get_splits
    from src.models.hybrid_model import build_model
    from src.utils.metrics import compute_f1, compute_iou

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CrackGraphDataset('data/graphs/train')
    train_idx, val_idx, test_idx = get_splits(dataset, 0.70, 0.15, seed=42)

    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, shuffle=False)
    holdout_loader = DataLoader(CrackGraphDataset('data/graphs/test'), batch_size=1, shuffle=False)

    thresholds = np.arange(0.10, 0.901, 0.02)
    step2_eval_results = []

    def eval_loader(model, loader, m_type):
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for d in loader:
                d = d.to(device)
                out = model(d.x, d.edge_index, d.pos) if m_type == 'hybrid' else model(d.x, d.edge_index)
                p = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                probs.extend(p)
                labels.extend(d.y.cpu().numpy())
        return np.array(probs), np.array(labels)

    def calc_metrics(preds, labels):
        tp = np.logical_and(preds == 1, labels == 1).sum()
        fp = np.logical_and(preds == 1, labels == 0).sum()
        fn = np.logical_and(preds == 0, labels == 1).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return float(prec), float(rec), float(f1), float(iou)

    for m_id, cfg_path in model_records:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        weights_path = os.path.join(BASE_DIR, cfg['output_dir'], 'best_model.pt')
        if not os.path.exists(weights_path):
            continue
            
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        m_type = cfg['model']['type']

        v_p, v_y = eval_loader(model, val_loader, m_type)
        t_p, t_y = eval_loader(model, test_loader, m_type)
        h_p, h_y = eval_loader(model, holdout_loader, m_type)

        # Val threshold sweep
        best_tau = 0.50
        best_val_f1 = -1.0
        for tau in thresholds:
            _, _, f1, _ = calc_metrics((v_p >= tau).astype(int), v_y)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_tau = round(float(tau), 2)

        # Metrics at default 0.50
        t_prec_def, t_rec_def, t_f1_def, t_iou_def = calc_metrics((t_p >= 0.50).astype(int), t_y)
        h_prec_def, h_rec_def, h_f1_def, h_iou_def = calc_metrics((h_p >= 0.50).astype(int), h_y)

        # Metrics at calibrated tau*
        t_prec_cal, t_rec_cal, t_f1_cal, t_iou_cal = calc_metrics((t_p >= best_tau).astype(int), t_y)
        h_prec_cal, h_rec_cal, h_f1_cal, h_iou_cal = calc_metrics((h_p >= best_tau).astype(int), h_y)

        step2_eval_results.append({
            'model_id': m_id,
            'best_val_tau': best_tau,
            'best_val_f1': best_val_f1,
            # Test split (45 imgs)
            'test_prec_def': t_prec_def, 'test_rec_def': t_rec_def,
            'test_f1_def': t_f1_def, 'test_iou_def': t_iou_def,
            'test_prec_cal': t_prec_cal, 'test_rec_cal': t_rec_cal,
            'test_f1_cal': t_f1_cal, 'test_iou_cal': t_iou_cal,
            # Holdout (237 imgs)
            'holdout_prec_def': h_prec_def, 'holdout_rec_def': h_rec_def,
            'holdout_f1_def': h_f1_def, 'holdout_iou_def': h_iou_def,
            'holdout_prec_cal': h_prec_cal, 'holdout_rec_cal': h_rec_cal,
            'holdout_f1_cal': h_f1_cal, 'holdout_iou_cal': h_iou_cal,
        })

    return step2_eval_results


def main():
    banner("STEP 2: RETRAINING M2 / M3 / M4 WITH FOCAL LOSS (gamma=2.0)")
    model_configs = create_step2_configs()
    
    # Train M2, M3, M4
    train_results = {}
    for m_id, cfg_path in model_configs:
        res = train_model(m_id, cfg_path)
        if res:
            train_results[m_id] = res

    # Run threshold sweep & evaluation
    banner("EVALUATING FOCAL LOSS MODELS WITH THRESHOLD CALIBRATION")
    eval_results = run_threshold_sweep_and_eval(model_configs)

    # Save results
    out_file = os.path.join(RESULTS_DIR, "step2_comparison.json")
    with open(out_file, 'w') as f:
        json.dump({'train_results': train_results, 'eval_results': eval_results}, f, indent=2)
    print(f"\nStep 2 results saved to {out_file}")

    # Load Step 1 baseline for direct comparison
    step1_file = os.path.join(BASE_DIR, "results", "threshold_calibration.json")
    step1_data = {}
    if os.path.exists(step1_file):
        with open(step1_file) as f:
            step1_data = {r['model_id']: r for r in json.load(f)['summary']}

    # Print Side-by-Side Comparison
    banner("SIDE-BY-SIDE COMPARISON: STEP 1 (WEIGHTED CE) vs. STEP 2 (FOCAL LOSS)")
    print("\nHOLDOUT TEST SET (237 IMAGES, 67,998 NODES) — CALIBRATED THRESHOLD:")
    print(f"{'Model':<18} {'Loss':<12} {'tau*':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10}")
    print("-" * 76)
    for r in eval_results:
        m_id = r['model_id']
        s1 = step1_data.get(m_id, {})
        print(f"{m_id:<18} {'Weighted CE':<12} {s1.get('best_val_tau', 0):<6.2f} {s1.get('holdout_prec_calibrated', 0):<10.4f} {s1.get('holdout_rec_calibrated', 0):<10.4f} {s1.get('holdout_f1_calibrated', 0):<10.4f} {s1.get('holdout_iou_calibrated', 0):<10.4f}")
        print(f"{m_id:<18} {'Focal (g=2)':<12} {r['best_val_tau']:<6.2f} {r['holdout_prec_cal']:<10.4f} {r['holdout_rec_cal']:<10.4f} {r['holdout_f1_cal']:<10.4f} {r['holdout_iou_cal']:<10.4f}")
        diff_f1 = r['holdout_f1_cal'] - s1.get('holdout_f1_calibrated', 0)
        diff_rec = r['holdout_rec_cal'] - s1.get('holdout_rec_calibrated', 0)
        print(f"   Delta: F1={diff_f1:+.4f}, Recall={diff_rec:+.4f}\n")


if __name__ == '__main__':
    main()
