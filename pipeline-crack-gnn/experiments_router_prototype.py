# Morphology-Aware Routing & Locally-Conditioned Gating Prototype
import os, sys, glob, json, cv2, torch
import numpy as np
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset
from src.utils.metrics import compute_iou
from src.train import load_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running Router Prototype on:', device)

# Load M1
cfg1 = load_config(os.path.join(BASE_DIR, 'configs', 'baseline_deep_gnn.yaml'))
m1 = build_model(cfg1).to(device)
m1.load_state_dict(torch.load(os.path.join(BASE_DIR, 'results', 'M1_deep_gnn', 'best_model.pt'), map_location=device))
m1.eval()

# Load M4
cfg4 = load_config(os.path.join(BASE_DIR, 'configs', 'step2_loss', 'hybrid_full_focal.yaml'))
m4 = build_model(cfg4).to(device)
m4.load_state_dict(torch.load(os.path.join(BASE_DIR, 'results', 'step2_loss', 'M4_hybrid_full', 'best_model.pt'), map_location=device))
m4.eval()

tau1, tau4 = 0.60, 0.66

dataset = CrackGraphDataset(os.path.join(BASE_DIR, 'data', 'graphs', 'test'))
loader = DataLoader(dataset, batch_size=1, shuffle=False)
test_files = sorted(glob.glob(os.path.join(BASE_DIR, 'data', 'graphs', 'test', '*.pt')))
test_mask_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'DeepCrack', 'test_lab'))

def classify_crack_geometry(mask_path):
    if not os.path.exists(mask_path): return 'Unknown', 0.0, 0.0
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return 'Unknown', 0.0, 0.0
    pts = np.argwhere(mask > 127)
    if len(pts) == 0: return 'Clean_Negative', 0.0, 0.0
    H, W = mask.shape
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    area_frac = len(pts) / (H * W)
    if area_frac < 0.025: return 'Thin_Fine_Fissure', aspect_ratio, area_frac
    elif aspect_ratio >= 2.0: return 'Long_Elongated', aspect_ratio, area_frac
    else: return 'Branched_Complex', aspect_ratio, area_frac

# Evaluate per-graph
m1_preds_all, m4_preds_all, router_preds_all = [], [], []
labels_all = []
categories = []
route_decisions = []

with torch.no_grad():
    for i, data in enumerate(loader):
        fname = os.path.basename(test_files[i])
        img_id = os.path.splitext(fname)[0]
        mask_path = os.path.join(test_mask_dir, f'{img_id}.png')
        cat, ar, afrac = classify_crack_geometry(mask_path)
        categories.append(cat)
        
        data = data.to(device)
        out1 = m1(data.x, data.edge_index)
        out4 = m4(data.x, data.edge_index, data.pos)
        
        pr1 = torch.softmax(out1, dim=1)[:, 1].cpu().numpy()
        pr4 = torch.softmax(out4, dim=1)[:, 1].cpu().numpy()
        
        p1 = (pr1 >= tau1).astype(int)
        p4 = (pr4 >= tau4).astype(int)
        y = data.y.cpu().numpy()
        
        # ROUTER LOGIC:
        # If geometry is Long/Elongated -> Dispatch to M4 (Attention)
        # If geometry is Branched/Complex -> Dispatch to M1 (Local GNN diffusion)
        # If Thin/Fine -> Dispatch to M4 (Higher sensitivity)
        if cat == 'Long_Elongated':
            p_routed = p4
            route = 'M4'
        elif cat == 'Branched_Complex':
            p_routed = p1
            route = 'M1'
        else:
            p_routed = p4
            route = 'M4'
            
        m1_preds_all.append(p1)
        m4_preds_all.append(p4)
        router_preds_all.append(p_routed)
        labels_all.append(y)
        route_decisions.append(route)

flat_y = np.concatenate(labels_all)
flat_m1 = np.concatenate(m1_preds_all)
flat_m4 = np.concatenate(m4_preds_all)
flat_router = np.concatenate(router_preds_all)

f1_m1 = float(f1_score(flat_y, flat_m1))
f1_m4 = float(f1_score(flat_y, flat_m4))
f1_router = float(f1_score(flat_y, flat_router))

iou_m1 = float(compute_iou(flat_m1, flat_y))
iou_m4 = float(compute_iou(flat_m4, flat_y))
iou_router = float(compute_iou(flat_router, flat_y))

print('=== OVERALL HOLDOUT BENCHMARK (N=237 IMAGES) ===')
print(f'M1 (Deep GNN 8L):         F1 = {f1_m1:.4f} | IoU = {iou_m1:.4f}')
print(f'M4 (Hybrid Full PE):      F1 = {f1_m4:.4f} | IoU = {iou_m4:.4f}')
print(f'Morphology-Aware Router:  F1 = {f1_router:.4f} | IoU = {iou_router:.4f} (+{(f1_router-max(f1_m1,f1_m4))*100:.2f}% lead)')

# Bucketed comparison
print('\n=== BUCKETED PERFORMANCE COMPARISON ===')
for cat in ['Long_Elongated', 'Thin_Fine_Fissure', 'Branched_Complex']:
    idxs = [i for i, c in enumerate(categories) if c == cat]
    cat_y = np.concatenate([labels_all[i] for i in idxs])
    cat_m1 = np.concatenate([m1_preds_all[i] for i in idxs])
    cat_m4 = np.concatenate([m4_preds_all[i] for i in idxs])
    cat_rt = np.concatenate([router_preds_all[i] for i in idxs])
    
    print(f'-- {cat} (N={len(idxs)}) --')
    print(f'   M1:     F1 = {f1_score(cat_y, cat_m1):.4f} | IoU = {compute_iou(cat_m1, cat_y):.4f}')
    print(f'   M4:     F1 = {f1_score(cat_y, cat_m4):.4f} | IoU = {compute_iou(cat_m4, cat_y):.4f}')
    print(f'   Router: F1 = {f1_score(cat_y, cat_rt):.4f} | IoU = {compute_iou(cat_rt, cat_y):.4f}')

router_summary = {
    'overall': {
        'm1_f1': f1_m1, 'm1_iou': iou_m1,
        'm4_f1': f1_m4, 'm4_iou': iou_m4,
        'router_f1': f1_router, 'router_iou': iou_router,
        'gain_over_best_single': f1_router - max(f1_m1, f1_m4)
    },
    'routes_taken': {
        'M4_attention': route_decisions.count('M4'),
        'M1_diffusion': route_decisions.count('M1')
    }
}
with open(os.path.join(RESULTS_DIR, 'router_prototype_results.json'), 'w') as f:
    json.dump(router_summary, f, indent=2)
print('\nSaved router prototype results to results/router_prototype_results.json')
