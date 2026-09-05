import os, sys, glob, json, cv2, torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from src.models.hybrid_model import build_model
from src.train import load_config
from src.utils.dataset import CrackGraphDataset
from torch_geometric.loader import DataLoader

BASE_DIR = r'c:\antigravity_projects\dcsnew\pipeline-crack-gnn'
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')
os.makedirs(DASHBOARD_DIR, exist_ok=True)

test_mask_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'DeepCrack', 'test_lab'))
test_files = sorted(glob.glob('data/graphs/test/*.pt'))
print('Total holdout test graphs:', len(test_files))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load M1
cfg1 = load_config('configs/baseline_deep_gnn.yaml')
m1 = build_model(cfg1).to(device)
m1.load_state_dict(torch.load('results/M1_deep_gnn/best_model.pt', map_location=device))
m1.eval()

# Load M4
cfg4 = load_config('configs/step2_loss/hybrid_full_focal.yaml')
m4 = build_model(cfg4).to(device)
m4.load_state_dict(torch.load('results/step2_loss/M4_hybrid_full/best_model.pt', map_location=device))
m4.eval()

tau1, tau4 = 0.60, 0.66

def classify_crack(mask_path):
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
    ar = float(max(w, h) / (min(w, h) + 1e-5))
    area_frac = float(len(pts) / (H * W))
    if area_frac < 0.025: return 'Thin_Fine_Fissure', ar, area_frac
    elif ar >= 2.0: return 'Long_Elongated', ar, area_frac
    else: return 'Branched_Complex', ar, area_frac

dataset = CrackGraphDataset('data/graphs/test')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

items = []
m1_disagrees = 0

with torch.no_grad():
    for i, data in enumerate(loader):
        fname = os.path.basename(test_files[i])
        img_id = os.path.splitext(fname)[0]
        mask_path = os.path.join(test_mask_dir, f'{img_id}.png')
        cat, ar, area_frac = classify_crack(mask_path)
        
        data = data.to(device)
        out1 = m1(data.x, data.edge_index)
        out4 = m4(data.x, data.edge_index, data.pos)
        
        pr1 = torch.softmax(out1, dim=1)[:, 1].cpu().numpy()
        pr4 = torch.softmax(out4, dim=1)[:, 1].cpu().numpy()
        
        p1 = (pr1 >= tau1).astype(int)
        p4 = (pr4 >= tau4).astype(int)
        y = data.y.cpu().numpy()
        
        f1_1 = float(f1_score(y, p1, zero_division=0))
        rec_1 = float(recall_score(y, p1, zero_division=0))
        prec_1 = float(precision_score(y, p1, zero_division=0))
        
        f1_4 = float(f1_score(y, p4, zero_division=0))
        rec_4 = float(recall_score(y, p4, zero_division=0))
        prec_4 = float(precision_score(y, p4, zero_division=0))
        
        diff = abs(f1_4 - f1_1)
        disagree = diff >= 0.15
        if disagree: m1_disagrees += 1
        
        items.append({
            'index': i,
            'id': img_id,
            'category': cat,
            'aspect_ratio': round(ar, 2),
            'area_fraction': round(area_frac, 4),
            'n_nodes': len(y),
            'n_cracks': int((y == 1).sum()),
            'm1': {'f1': round(f1_1, 4), 'recall': round(rec_1, 4), 'precision': round(prec_1, 4)},
            'm4': {'f1': round(f1_4, 4), 'recall': round(rec_4, 4), 'precision': round(prec_4, 4)},
            'f1_diff': round(f1_4 - f1_1, 4),
            'abs_diff': round(diff, 4),
            'disagree': disagree,
            'winner': 'M4' if f1_4 > f1_1 + 0.05 else ('M1' if f1_1 > f1_4 + 0.05 else 'Tie')
        })

out_json = os.path.join(DASHBOARD_DIR, 'predictions_data.json')
with open(out_json, 'w') as f:
    json.dump(items, f, indent=2)

print(f'Saved {len(items)} image comparisons to {out_json}.')
print(f'Models disagree (|F1_M4 - F1_M1| >= 0.15) on {m1_disagrees} images ({m1_disagrees/len(items)*100:.1f}%).')
