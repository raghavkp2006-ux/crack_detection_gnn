# Rigorious TITS Multi-Sensor Evaluation & Calibration Script
import os, sys, glob, json, torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset
from src.utils.metrics import compute_iou
from src.train import load_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITS_GRAPH_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'tits')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

MODELS = [
    {'id': 'M1_deep_gnn', 'name': 'M1 (Deep GNN 8L)', 'config': os.path.join(BASE_DIR, 'configs', 'baseline_deep_gnn.yaml'), 'checkpoint': os.path.join(BASE_DIR, 'results', 'M1_deep_gnn', 'best_model.pt')},
    {'id': 'M2_shallow_gnn', 'name': 'M2 (Shallow GNN 2L)', 'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'shallow_gnn_focal.yaml'), 'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M2_shallow_gnn', 'best_model.pt')},
    {'id': 'M3_hybrid_no_pe', 'name': 'M3 (Hybrid No PE)', 'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_no_pe_focal.yaml'), 'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M3_hybrid_no_pe', 'best_model.pt')},
    {'id': 'M4_hybrid_full', 'name': 'M4 (Hybrid Full PE)', 'config': os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_full_focal.yaml'), 'checkpoint': os.path.join(BASE_DIR, 'results', 'step2_loss', 'M4_hybrid_full', 'best_model.pt')},
]

def get_sensor(fname):
    fn = os.path.basename(fname).lower()
    if 'aigle' in fn: return 'AIGLE_RN'
    if 'esar' in fn: return 'ESAR'
    if 'lcms' in fn: return 'LCMS'
    if 'lris' in fn: return 'LRIS'
    if 'tempest' in fn: return 'TEMPEST2'
    return 'Other'

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Evaluating rigorous TITS transfer on:', device)

    tits_dataset = CrackGraphDataset(TITS_GRAPH_DIR)
    tits_files = sorted(glob.glob(os.path.join(TITS_GRAPH_DIR, '*.pt')))

    # Filter out TEMPEST2 (N=1, zero cracks)
    valid_indices, lris_indices, pos_sensor_indices = [], [], []
    for i, f in enumerate(tits_files):
        s = get_sensor(f)
        if s == 'TEMPEST2': continue
        elif s == 'LRIS': lris_indices.append(i)
        else: pos_sensor_indices.append(i); valid_indices.append(i)

    print(f'Total valid: {len(valid_indices)} positive sensors, {len(lris_indices)} LRIS negative control (TEMPEST2 dropped)')

    # Stratified 20% calibration / 80% evaluation split on positive sensors
    rng = np.random.default_rng(42)
    calib_indices, eval_indices = [], []

    pos_sensors = [get_sensor(tits_files[i]) for i in pos_sensor_indices]
    for s in ['AIGLE_RN', 'ESAR', 'LCMS']:
        s_idxs = [pos_sensor_indices[k] for k, sn in enumerate(pos_sensors) if sn == s]
        rng.shuffle(s_idxs)
        n_cal = max(1, int(len(s_idxs) * 0.20))
        calib_indices.extend(s_idxs[:n_cal])
        eval_indices.extend(s_idxs[n_cal:])

    print(f'Stratified Split: {len(calib_indices)} calibration graphs, {len(eval_indices)} holdout evaluation graphs')

    calib_sub = Subset(tits_dataset, calib_indices)
    eval_sub = Subset(tits_dataset, eval_indices)
    lris_sub = Subset(tits_dataset, lris_indices)

    calib_loader = DataLoader(calib_sub, batch_size=1, shuffle=False)
    eval_loader = DataLoader(eval_sub, batch_size=1, shuffle=False)
    lris_loader = DataLoader(lris_sub, batch_size=1, shuffle=False)

    results = {}

    for minfo in MODELS:
        mid = minfo['id']
        mname = minfo['name']
        cfg = load_config(minfo['config'])
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(minfo['checkpoint'], map_location=device))
        model.eval()

        m_type = cfg['model']['type']

        # 1. Calibrate threshold strictly on calib_loader
        cal_labels, cal_probs = [], []
        with torch.no_grad():
            for data in calib_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.pos) if m_type == 'hybrid' else model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                cal_probs.append(probs)
                cal_labels.append(data.y.cpu().numpy())

        cal_labels = np.moncatenate if hasattr(np, 'moncatenate') else np.concatenate(cal_labels)
        cal_probs = np.concatenate(cal_probs)

        best_tau, best_cal_f1 = 0.50, 0.0
        for tau in np.arange(0.10, 0.92, 0.02):
            preds = (cal_probs >= tau).astype(int)
            f1 = f1_score(cal_labels, preds, zero_division=0)
            if f1 > best_cal_f1:
                best_cal_f1 = f1
                best_tau = float(tau)

        # 2. Evaluate on 80% holdout eval_loader
        eval_labels, eval_probs, eval_sensors = [], [], []
        with torch.no_grad():
            for idx, data in zip(eval_indices, eval_loader):
                data = data.to(device)
                out = model(data.x, data.edge_index, data.pos) if m_type == 'hybrid' else model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                eval_probs.append(probs)
                eval_labels.append(data.y.cpu().numpy())
                eval_sensors.append(get_sensor(tits_files[idx]))

        flat_eval_labels = np.concatenate(eval_labels)
        flat_eval_probs = np.concatenate(eval_probs)
        flat_eval_preds = (flat_eval_probs >= best_tau).astype(int)

        micro_f1 = float(f1_score(flat_eval_labels, flat_eval_preds, zero_division=0))
        micro_iou = float(compute_iou(flat_eval_preds, flat_eval_labels))
        micro_prec = float(precision_score(flat_eval_labels, flat_eval_preds, zero_division=0))
        micro_rec = float(recall_score(flat_eval_labels, flat_eval_preds, zero_division=0))
        micro_auc = float(roc_auc_score(flat_eval_labels, flat_eval_probs))

        sensor_metrics, sensor_weights = {}, {}
        for s in ['AIGLE_RN', 'ESAR', 'LCMS']:
            s_mask = [i for i, sn in enumerate(eval_sensors) if sn == s]
            if s_mask:
                s_l = np.concatenate([eval_labels[i] for i in s_mask])
                s_pr = np.concatenate([eval_probs[i] for i in s_mask])
                s_p = (s_pr >= best_tau).astype(int)
                s_f1 = float(f1_score(s_l, s_p, zero_division=0))
                s_iou = float(compute_iou(s_p, s_l))
                s_prec = float(precision_score(s_l, s_p, zero_division=0))
                s_rec = float(recall_score(s_l, s_p, zero_division=0))
                s_auc = float(roc_auc_score(s_l, s_pr)) if len(np.unique(s_l)) > 1 else 0.5
                sensor_metrics[s] = {
                    'n_graphs': len(s_mask),
                    'n_nodes': len(s_l),
                    'n_cracks': int((s_l == 1).sum()),
                    'f1': s_f1,
                    'iou': s_iou,
                    'precision': s_prec,
                    'recall': s_rec,
                    'roc_auc': s_auc
                }
                sensor_weights[s] = len(s_mask)

        total_eval_graphs = sum(sensor_weights.values())
        macro_f1 = sum(sensor_metrics[s]['f1'] * sensor_weights[s] for s in sensor_weights) / total_eval_graphs
        macro_iou = sum(sensor_metrics[s]['iou'] * sensor_weights[s] for s in sensor_weights) / total_eval_graphs
        macro_auc = sum(sensor_metrics[s]['roc_auc'] * sensor_weights[s] for s in sensor_weights) / total_eval_graphs

        # 3. LRIS Negative Control (N=11)
        lris_preds_count, lris_nodes_count = 0, 0
        with torch.no_grad():
            for data in lris_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.pos) if m_type == 'hybrid' else model(data.x, data.edge_index)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= best_tau).astype(int)
                lris_preds_count += int(preds.sum())
                lris_nodes_count += len(preds)

        fpr_lris = lris_preds_count / lris_nodes_count if lris_nodes_count > 0 else 0.0

        results[mid] = {
            'name': mname, 'optimal_tau': best_tau, 'calib_f1': float(best_cal_f1),
            'eval_micro': {'f1': micro_f1, 'iou': micro_iou, 'precision': micro_prec, 'recall': micro_rec, 'roc_auc': micro_auc},
            'eval_macro_weighted': {'f1': macro_f1, 'iou': macro_iou, 'roc_auc': macro_auc},
            'negative_control_lris': {'n_graphs': len(lris_indices), 'n_nodes': lris_nodes_count, 'false_positives': lris_preds_count, 'fpr': float(fpr_lris), 'specificity': float(1.0 - fpr_lris)},
            'per_sensor': sensor_metrics}

        print(f'=== {mname} (Tau*={best_tau:.2f} calibrated on 20% calib split) ===')
        print(f'  Micro-Avg (Holdout 80%): F1={micro_f1:.4f} | IoU={micro_iou:.4f} | Rec={micro_rec:.4f} | Prec={micro_prec:.4f} | AUC={micro_auc:.4f}')
        print(f'  Macro-Avg (Weighted):    F1={macro_f1:.4f} | IoU={macro_iou:.4f} | AUC={macro_auc:.4f}')
        print(f'  LRIS Negative Control:   FPR={fpr_lris*100:.2f}% | Specificity={(1.0-fpr_lris)*100:.2f}% ({lris_preds_count}/{lris_nodes_count} false nodes)')
        for s, sm in sensor_metrics.items():
            print(f'    - {s:<10} (N={sm["n_graphs"]}): F1={sm["f1"]:.4f} | IoU={sm["iou"]:.4f} | Rec={sm["recall"]:.4f} | AUC={sm["roc_auc"]:.4f}')

    out_path = os.path.join(RESULTS_DIR, 'tits_rigorous_evaluation_summary.json')
    with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f'Saved rigorous TITS results to {out_path}')

if __name__ == '__main__': main()
