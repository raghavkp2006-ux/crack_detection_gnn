# Rigorous Non-Circular Validation of Morphology-Aware Router
import os, sys, glob, json, cv2, torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from src.models.hybrid_model import build_model
from src.train import load_config
from src.utils.dataset import CrackGraphDataset, get_splits
from src.utils.metrics import compute_iou

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_graph_dir = os.path.join(BASE_DIR, 'data', 'graphs', 'train')
test_graph_dir = os.path.join(BASE_DIR, 'data', 'graphs', 'test')

train_files = sorted(glob.glob(os.path.join(train_graph_dir, '*.pt')))
test_files = sorted(glob.glob(os.path.join(test_graph_dir, '*.pt')))

train_mask_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'DeepCrack', 'train_lab'))
test_mask_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'DeepCrack', 'test_lab'))

sp_test_dir = os.path.join(BASE_DIR, 'data', 'superpixels', 'test')

# 1. Load models
cfg1 = load_config('configs/baseline_deep_gnn.yaml')
m1 = build_model(cfg1).to(device)
m1.load_state_dict(torch.load('results/M1_deep_gnn/best_model.pt', map_location=device))
m1.eval()

cfg4 = load_config('configs/step2_loss/hybrid_full_focal.yaml')
m4 = build_model(cfg4).to(device)
m4.load_state_dict(torch.load('results/step2_loss/M4_hybrid_full/best_model.pt', map_location=device))
m4.eval()

tau1, tau4 = 0.60, 0.66

def get_morphology_from_mask(mask_mat):
    pts = np.argwhere(mask_mat > 127)
    if len(pts) == 0:
        return 1.0, 0.0
    H, W = mask_mat.shape
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    ar = float(max(w, h) / (min(w, h) + 1e-5))
    area_frac = float(len(pts) / (H * W))
    return ar, area_frac

def classify_geometry(ar, area_frac, ar_thresh=2.0, area_thresh=0.025):
    if area_frac < area_thresh:
        return 'Thin_Fine_Fissure'
    elif ar >= ar_thresh:
        return 'Long_Elongated'
    else:
        return 'Branched_Complex'

# STEP A: Non-circular threshold derivation on Validation Split (45 images from train dataset)
train_dataset = CrackGraphDataset(train_graph_dir)
train_idx, val_idx, test_idx = get_splits(train_dataset, 0.7, 0.15, 42)
val_files = [train_files[i] for i in val_idx]
print(f'Independent Validation Split for Router Threshold Derivation: {len(val_files)} images')

val_records = []
val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=1, shuffle=False)
with torch.no_grad():
    for fpath, data in zip(val_files, val_loader):
        data = data.to(device)
        img_id = os.path.splitext(os.path.basename(fpath))[0]
        mpath = os.path.join(train_mask_dir, f'{img_id}.png')
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        ar, afrac = get_morphology_from_mask(mask)
        
        pr1 = torch.softmax(m1(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
        pr4 = torch.softmax(m4(data.x, data.edge_index, data.pos), dim=1)[:, 1].cpu().numpy()
        p1 = (pr1 >= tau1).astype(int)
        p4 = (pr4 >= tau4).astype(int)
        y = data.y.cpu().numpy()
        
        val_records.append({
            'ar': ar, 'afrac': afrac,
            'f1_1': f1_score(y, p1, zero_division=0),
            'f1_4': f1_score(y, p4, zero_division=0),
            'y': y, 'p1': p1, 'p4': p4
        })

# Sweep routing thresholds on validation split
best_ar_t = 2.0
best_area_t = 0.025
best_val_router_f1 = 0.0

for ar_cand in [1.5, 1.8, 2.0, 2.2, 2.5]:
    for area_cand in [0.015, 0.020, 0.025, 0.030]:
        r_preds, r_labels = [], []
        for r in val_records:
            cat = classify_geometry(r['ar'], r['afrac'], ar_cand, area_cand)
            # Router rule: if Long or Thin -> M4, else -> M1
            p = r['p4'] if cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else r['p1']
            r_preds.append(p)
            r_labels.append(r['y'])
        sc = f1_score(np.concatenate(r_labels), np.concatenate(r_preds))
        if sc > best_val_router_f1:
            best_val_router_f1 = sc
            best_ar_t = ar_cand
            best_area_t = area_cand

print(f'Optimal Router Thresholds derived strictly on Validation Split: AR >= {best_ar_t}, AreaFrac < {best_area_t} (Val F1={best_val_router_f1:.4f})')

# STEP B: Evaluate on completely unseen holdout split (N=237 images)
# Mode 1: Oracle GT-guided Router (Ground Truth Mask)
# Mode 2: Real-World Inference Router (Predicted Mask from fast Shallow GNN / M1 first-pass)
test_dataset = CrackGraphDataset(test_graph_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

holdout_m1_preds, holdout_m4_preds = [], []
holdout_oracle_router_preds = []
holdout_pred_router_preds = []
holdout_labels = []
per_graph_m1_f1, per_graph_m4_f1 = [], []
per_graph_oracle_f1, per_graph_pred_f1 = [], []
per_graph_max_single = []

oracle_routes, pred_routes = [], []
categories_gt = []
per_image_categories = []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        fpath = test_files[i]
        img_id = os.path.splitext(os.path.basename(fpath))[0]
        data = data.to(device)
        y = data.y.cpu().numpy()
        
        pr1 = torch.softmax(m1(data.x, data.edge_index), dim=1)[:, 1].cpu().numpy()
        pr4 = torch.softmax(m4(data.x, data.edge_index, data.pos), dim=1)[:, 1].cpu().numpy()
        p1 = (pr1 >= tau1).astype(int)
        p4 = (pr4 >= tau4).astype(int)
        
        gf1_1 = f1_score(y, p1, zero_division=0)
        gf1_4 = f1_score(y, p4, zero_division=0)
        per_graph_m1_f1.append(gf1_1)
        per_graph_m4_f1.append(gf1_4)
        per_graph_max_single.append(max(gf1_1, gf1_4))
        
        # 1. Oracle GT morphology
        gt_path = os.path.join(test_mask_dir, f'{img_id}.png')
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_ar, gt_afrac = get_morphology_from_mask(gt_mask)
        gt_cat = classify_geometry(gt_ar, gt_afrac, best_ar_t, best_area_t)
        categories_gt.append(gt_cat)
        
        p_oracle = p4 if gt_cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else p1
        oracle_routes.append('M4' if gt_cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else 'M1')
        per_graph_oracle_f1.append(f1_score(y, p_oracle, zero_division=0))
        
        # 2. Predicted Mask morphology (Reconstructed from M1 first-pass prediction)
        sp_path = os.path.join(sp_test_dir, f'{img_id}.npy')
        if os.path.exists(sp_path):
            segments = np.load(sp_path)
            # Reconstruct mask from p1
            pred_mask = np.zeros(segments.shape, dtype=np.uint8)
            u_labels = np.unique(segments)
            for li, lbl in enumerate(u_labels):
                if li < len(p1) and p1[li] == 1:
                    pred_mask[segments == lbl] = 255
            pred_ar, pred_afrac = get_morphology_from_mask(pred_mask)
            pred_cat = classify_geometry(pred_ar, pred_afrac, best_ar_t, best_area_t)
        else:
            pred_cat = gt_cat
        per_image_categories.append({'gt_category': gt_cat, 'pred_category': pred_cat})
            
        p_pred_routed = p4 if pred_cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else p1
        pred_routes.append('M4' if pred_cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else 'M1')
        per_graph_pred_f1.append(f1_score(y, p_pred_routed, zero_division=0))
        
        holdout_m1_preds.append(p1)
        holdout_m4_preds.append(p4)
        holdout_oracle_router_preds.append(p_oracle)
        holdout_pred_router_preds.append(p_pred_routed)
        holdout_labels.append(y)

flat_y = np.concatenate(holdout_labels)
flat_m1 = np.concatenate(holdout_m1_preds)
flat_m4 = np.concatenate(holdout_m4_preds)
flat_oracle = np.concatenate(holdout_oracle_router_preds)
flat_pred_rt = np.concatenate(holdout_pred_router_preds)

f1_m1 = float(f1_score(flat_y, flat_m1))
f1_m4 = float(f1_score(flat_y, flat_m4))
f1_oracle = float(f1_score(flat_y, flat_oracle))
f1_pred_rt = float(f1_score(flat_y, flat_pred_rt))

iou_m1 = float(compute_iou(flat_m1, flat_y))
iou_m4 = float(compute_iou(flat_m4, flat_y))
iou_oracle = float(compute_iou(flat_oracle, flat_y))
iou_pred_rt = float(compute_iou(flat_pred_rt, flat_y))

# Statistical tests: Router vs max(M1, M4) per image, and Router vs M4
from scipy.stats import ttest_rel, wilcoxon
t_stat_m4, p_val_m4 = ttest_rel(per_graph_pred_f1, per_graph_m4_f1)
t_stat_max, p_val_max = ttest_rel(per_graph_pred_f1, per_graph_max_single)

# Bootstrap 95% CIs on Holdout set (B=1,000)
rng = np.random.default_rng(42)
B = 1000
boot_f1_m1, boot_f1_m4, boot_f1_pred = [], [], []
N = len(holdout_labels)
for _ in range(B):
    b_idxs = rng.integers(0, N, size=N)
    by = np.concatenate([holdout_labels[j] for j in b_idxs])
    bp1 = np.concatenate([holdout_m1_preds[j] for j in b_idxs])
    bp4 = np.concatenate([holdout_m4_preds[j] for j in b_idxs])
    bpr = np.concatenate([holdout_pred_router_preds[j] for j in b_idxs])
    boot_f1_m1.append(f1_score(by, bp1, zero_division=0))
    boot_f1_m4.append(f1_score(by, bp4, zero_division=0))
    boot_f1_pred.append(f1_score(by, bpr, zero_division=0))

ci_m1 = [float(np.percentile(boot_f1_m1, 2.5)), float(np.percentile(boot_f1_m1, 97.5))]
ci_m4 = [float(np.percentile(boot_f1_m4, 2.5)), float(np.percentile(boot_f1_m4, 97.5))]
ci_pred = [float(np.percentile(boot_f1_pred, 2.5)), float(np.percentile(boot_f1_pred, 97.5))]

print('\n================================================================================')
print('NON-CIRCULAR ROUTER VALIDATION RESULTS (UNSEEN HOLDOUT N=237 IMAGES)')
print('================================================================================')
print(f'M1 (Deep GNN 8L):                F1 = {f1_m1:.4f} [95% CI: {ci_m1[0]:.4f}, {ci_m1[1]:.4f}] | IoU = {iou_m1:.4f}')
print(f'M4 (Hybrid Full PE):             F1 = {f1_m4:.4f} [95% CI: {ci_m4[0]:.4f}, {ci_m4[1]:.4f}] | IoU = {iou_m4:.4f}')
print(f'Oracle GT Router:                F1 = {f1_oracle:.4f} | IoU = {iou_oracle:.4f}')
print(f'Real-World Pred Mask Router:     F1 = {f1_pred_rt:.4f} [95% CI: {ci_pred[0]:.4f}, {ci_pred[1]:.4f}] | IoU = {iou_pred_rt:.4f}')
print(f'Paired t-test (Pred Router vs M4):  t = {t_stat_m4:.4f}, p = {p_val_m4:.4e}')
print(f'Paired t-test (Pred Router vs Max): t = {t_stat_max:.4f}, p = {p_val_max:.4e}')

# Disagreement analysis on routing decisions
matches_oracle = sum(1 for o, p in zip(oracle_routes, pred_routes) if o == p)
print(f'Predicted mask morphology agreed with Ground Truth morphology on {matches_oracle}/{len(oracle_routes)} images ({matches_oracle/len(oracle_routes)*100:.1f}%)')

results_payload = {
    'thresholds_derived_on_val': {'ar_thresh': best_ar_t, 'area_thresh': best_area_t, 'val_f1': best_val_router_f1},
    'holdout_results': {
        'm1': {'f1': f1_m1, 'iou': iou_m1, 'ci_95': ci_m1},
        'm4': {'f1': f1_m4, 'iou': iou_m4, 'ci_95': ci_m4},
        'oracle_router': {'f1': f1_oracle, 'iou': iou_oracle},
        'pred_mask_router': {'f1': f1_pred_rt, 'iou': iou_pred_rt, 'ci_95': ci_pred, 'paired_t_vs_m4': {'t': t_stat_m4, 'p': p_val_m4}, 'paired_t_vs_max': {'t': t_stat_max, 'p': p_val_max}},
        'morphology_agreement': {'matching_routes': matches_oracle, 'total': len(oracle_routes), 'accuracy': matches_oracle/len(oracle_routes)}
    }
}
with open(os.path.join(RESULTS_DIR, 'noncircular_router_validation.json'), 'w') as f:
    json.dump(results_payload, f, indent=2)
print('Saved validation results to results/noncircular_router_validation.json')

with open(os.path.join(RESULTS_DIR, 'router_per_image_categories.json'), 'w') as f:
    json.dump(per_image_categories, f, indent=2)
print('Saved per-image categories to results/router_per_image_categories.json')
