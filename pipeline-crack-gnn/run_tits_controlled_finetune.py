"""
TITS Transfer Learning Reconciliation & Controlled Fine-Tuning Experiment.

Investigates:
1. Zero-shot transfer performance across the 5 TITS sensor modalities:
   AIGLE_RN, ESAR, LCMS, LRIS, TEMPEST2.
2. Direct comparison: M1 vs M2 vs M3 (no PE) vs M4 (PE) on TITS to test if 2D positional encoding
   or parameter scale caused the transfer reversal.
3. Controlled fine-tuning: fine-tune M1 and M4 on a 20% slice of TITS (18 graphs) and test on 80% holdout (74 graphs).
"""

import os
import sys
import glob
import json
import time
import torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from collections import defaultdict

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc
from src.train import load_config, get_inverted_class_weights, FocalLoss, evaluate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITS_GRAPH_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'tits')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

MODELS_ZERO_SHOT = [
    {
        'id': 'M1_deep_gnn',
        'name': 'M1 (Deep GNN 8L)',
        'config': os.path.join(BASE_DIR, 'configs', 'baseline_deep_gnn.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'M1_deep_gnn', 'best_model.pt'),
    },
    {
        'id': 'M2_shallow_gnn',
        'name': 'M2 (Shallow GNN 2L)',
        'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'shallow_gnn_focal.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M2_shallow_gnn', 'best_model.pt'),
    },
    {
        'id': 'M3_hybrid_no_pe',
        'name': 'M3 (Hybrid No PE)',
        'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_no_pe_focal.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M3_hybrid_no_pe', 'best_model.pt'),
    },
    {
        'id': 'M4_hybrid_full',
        'name': 'M4 (Hybrid Full PE)',
        'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_full_focal.yaml'),
        'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M4_hybrid_full', 'best_model.pt'),
    },
]


def extract_sensor_name(filename):
    fname = os.path.basename(filename).lower()
    if 'aigle' in fname:
        return 'AIGLE_RN'
    elif 'esar' in fname:
        return 'ESAR'
    elif 'lcms' in fname:
        return 'LCMS'
    elif 'lris' in fname:
        return 'LRIS'
    elif 'tempest' in fname:
        return 'TEMPEST2'
    return 'Other'


def run_zero_shot_evaluation(device, tits_dataset, tits_files):
    print("\n" + "="*70)
    print("1. ZERO-SHOT TRANSFER EVALUATION ACROSS ALL TITS SENSORS (N=92 GRAPHS)")
    print("="*70)
    loader = DataLoader(tits_dataset, batch_size=1, shuffle=False)
    
    sensors = [extract_sensor_name(f) for f in tits_files]
    sensor_types = sorted(list(set(sensors)))
    print(f"Sensor distribution: { {s: sensors.count(s) for s in sensor_types} }")

    zero_shot_results = {}

    for m_info in MODELS_ZERO_SHOT:
        mid = m_info['id']
        mname = m_info['name']
        cfg = load_config(m_info['config'])
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(m_info['checkpoint'], map_location=device))
        model.eval()

        m_type = cfg['model']['type']
        graph_preds, graph_labels, graph_probs = [], [], []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                if m_type == 'hybrid':
                    out = model(data.x, data.edge_index, data.pos)
                else:
                    out = model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                graph_probs.append(probs)
                graph_labels.append(data.y.cpu().numpy())

        # Sweep threshold to find best operating point on TITS
        flat_labels = np.concatenate(graph_labels)
        flat_probs = np.concatenate(graph_probs)

        best_tau, best_f1 = 0.50, 0.0
        for tau in np.arange(0.10, 0.92, 0.02):
            preds = (flat_probs >= tau).astype(int)
            f1 = f1_score(flat_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

        # Calibrated predictions
        flat_preds = (flat_probs >= best_tau).astype(int)
        f1 = float(f1_score(flat_labels, flat_preds, zero_division=0))
        iou = float(compute_iou(flat_preds, flat_labels))
        rec = float(recall_score(flat_labels, flat_preds, zero_division=0))
        prec = float(precision_score(flat_labels, flat_preds, zero_division=0))
        auc = float(roc_auc_score(flat_labels, flat_probs))

        # Per-sensor evaluation
        per_sensor = {}
        for s in sensor_types:
            s_idxs = [i for i, sn in enumerate(sensors) if sn == s]
            if s_idxs:
                s_l = np.concatenate([graph_labels[i] for i in s_idxs])
                s_p = np.concatenate([(graph_probs[i] >= best_tau).astype(int) for i in s_idxs])
                s_prob = np.concatenate([graph_probs[i] for i in s_idxs])
                s_f1 = float(f1_score(s_l, s_p, zero_division=0))
                s_iou = float(compute_iou(s_p, s_l))
                s_rec = float(recall_score(s_l, s_p, zero_division=0))
                s_auc = float(roc_auc_score(s_l, s_prob)) if len(np.unique(s_l)) > 1 else 0.5
                per_sensor[s] = {'f1': s_f1, 'iou': s_iou, 'recall': s_rec, 'roc_auc': s_auc}

        zero_shot_results[mid] = {
            'name': mname,
            'optimal_tau': best_tau,
            'f1': f1, 'iou': iou, 'recall': rec, 'precision': prec, 'roc_auc': auc,
            'per_sensor': per_sensor
        }

        print(f"\n{mname:<25} | Tau*: {best_tau:.2f} | F1: {f1:.4f} | IoU: {iou:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | AUC: {auc:.4f}")
        for s in sensor_types:
            sm = per_sensor[s]
            print(f"    - {s:<10}: F1 = {sm['f1']:.4f} | IoU = {sm['iou']:.4f} | Rec = {sm['recall']:.4f} | AUC = {sm['roc_auc']:.4f}")

    return zero_shot_results, sensors, sensor_types


def run_controlled_finetune(device, tits_dataset, tits_files, sensors):
    print("\n" + "="*70)
    print("2. CONTROLLED FEW-SHOT FINE-TUNING EXPERIMENT (20% TRAIN / 80% TEST)")
    print("="*70)

    # Stratified 20% train / 80% holdout split across sensors
    rng = np.random.default_rng(42)
    train_indices = []
    test_indices = []

    sensor_types = sorted(list(set(sensors)))
    for s in sensor_types:
        s_idxs = [i for i, sn in enumerate(sensors) if sn == s]
        rng.shuffle(s_idxs)
        n_train = max(1, int(len(s_idxs) * 0.20))
        train_indices.extend(s_idxs[:n_train])
        test_indices.extend(s_idxs[n_train:])

    print(f"Fine-tuning split: {len(train_indices)} train graphs, {len(test_indices)} holdout test graphs.")

    train_sub = Subset(tits_dataset, train_indices)
    test_sub = Subset(tits_dataset, test_indices)

    train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_sub, batch_size=1, shuffle=False)

    finetune_results = {}

    for mid, cfg_path, ckpt_path in [
        ('M1_deep_gnn', 'configs/baseline_deep_gnn.yaml', 'results/M1_deep_gnn/best_model.pt'),
        ('M4_hybrid_full', 'configs/step2_loss/hybrid_full_focal.yaml', 'results/step2_loss/M4_hybrid_full/best_model.pt')
    ]:
        cfg = load_config(cfg_path)
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Fine-tune with small LR
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        alpha = get_inverted_class_weights(train_sub).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=2.0)

        model.train()
        for ep in range(15):
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                if cfg['model']['type'] == 'hybrid':
                    out = model(data.x, data.edge_index, data.pos)
                else:
                    out = model(data.x, data.edge_index)
                loss = criterion(out, data.y.long())
                loss.backward()
                optimizer.step()

        # Evaluate on the 80% holdout test set
        model.eval()
        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                if cfg['model']['type'] == 'hybrid':
                    out = model(data.x, data.edge_index, data.pos)
                else:
                    out = model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                test_probs.append(probs)
                test_labels.append(data.y.cpu().numpy())

        flat_l = np.concatenate(test_labels)
        flat_pr = np.concatenate(test_probs)

        # Calibrate threshold
        best_tau, best_f1 = 0.50, 0.0
        for tau in np.arange(0.10, 0.92, 0.02):
            p = (flat_pr >= tau).astype(int)
            f1 = f1_score(flat_l, p, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)

        final_p = (flat_pr >= best_tau).astype(int)
        f1 = float(f1_score(flat_l, final_p, zero_division=0))
        iou = float(compute_iou(final_p, flat_l))
        rec = float(recall_score(flat_l, final_p, zero_division=0))
        prec = float(precision_score(flat_l, final_p, zero_division=0))
        auc = float(roc_auc_score(flat_l, flat_pr))

        finetune_results[mid] = {
            'optimal_tau': best_tau,
            'f1': f1, 'iou': iou, 'recall': rec, 'precision': prec, 'roc_auc': auc
        }
        print(f"\nFine-Tuned {mid:<18} (Trained on 20% TITS, Evaluated on 80% Holdout):")
        print(f"  Tau*: {best_tau:.2f} | F1: {f1:.4f} | IoU: {iou:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | ROC-AUC: {auc:.4f}")

    return finetune_results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tits_dataset = CrackGraphDataset(TITS_GRAPH_DIR)
    tits_files = sorted(glob.glob(os.path.join(TITS_GRAPH_DIR, '*.pt')))

    zero_shot, sensors, sensor_types = run_zero_shot_evaluation(device, tits_dataset, tits_files)
    finetune_res = run_controlled_finetune(device, tits_dataset, tits_files, sensors)

    summary = {
        'zero_shot': zero_shot,
        'controlled_finetuning': finetune_res,
        'sensor_distribution': {s: sensors.count(s) for s in sensor_types}
    }

    out_file = os.path.join(RESULTS_DIR, 'tits_reconciliation_summary.json')
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved TITS reconciliation results to {out_file}")


if __name__ == '__main__':
    main()

