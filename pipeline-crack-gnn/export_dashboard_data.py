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

# Load M2
cfg2 = load_config('configs/shallow_gnn.yaml')
m2 = build_model(cfg2).to(device)
m2.load_state_dict(torch.load('results/M2_shallow_gnn/best_model.pt', map_location=device))
m2.eval()

# Load M3
cfg3 = load_config('configs/hybrid_no_pe.yaml')
m3 = build_model(cfg3).to(device)
m3.load_state_dict(torch.load('results/M3_hybrid_no_pe/best_model.pt', map_location=device))
m3.eval()

# Load M4
cfg4 = load_config('configs/step2_loss/hybrid_full_focal.yaml')
m4 = build_model(cfg4).to(device)
m4.load_state_dict(torch.load('results/step2_loss/M4_hybrid_full/best_model.pt', map_location=device))
m4.eval()

# Thresholds per model
tau1 = 0.60
tau2 = 0.70
tau3 = 0.74
tau4 = 0.66

# Load DeepCrack per-image predictions if available
dc_result_path = os.path.join(BASE_DIR, 'results', 'multiseed', 'deepcrack', 'seed_42', 'result.json')
dc_per_image = {}
if os.path.exists(dc_result_path):
    with open(dc_result_path, 'r') as f:
        dc_data = json.load(f)
        dc_per_image = dc_data.get('per_image_results', {})
    print(f'Loaded DeepCrack predictions for {len(dc_per_image)} test images.')
else:
    print('Warning: DeepCrack seed_42 result.json not found, using defaults.')

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
m1_m4_disagrees = 0

with torch.no_grad():
    for i, data in enumerate(loader):
        fname = os.path.basename(test_files[i])
        img_id = os.path.splitext(fname)[0]
        mask_path = os.path.join(test_mask_dir, f'{img_id}.png')
        cat, ar, area_frac = classify_crack(mask_path)
        
        data = data.to(device)
        out1 = m1(data.x, data.edge_index)
        out2 = m2(data.x, data.edge_index)
        out3 = m3(data.x, data.edge_index, data.pos)
        out4 = m4(data.x, data.edge_index, data.pos)
        
        pr1 = torch.softmax(out1, dim=1)[:, 1].cpu().numpy()
        pr2 = torch.softmax(out2, dim=1)[:, 1].cpu().numpy()
        pr3 = torch.softmax(out3, dim=1)[:, 1].cpu().numpy()
        pr4 = torch.softmax(out4, dim=1)[:, 1].cpu().numpy()
        
        p1 = (pr1 >= tau1).astype(int)
        p2 = (pr2 >= tau2).astype(int)
        p3 = (pr3 >= tau3).astype(int)
        p4 = (pr4 >= tau4).astype(int)
        y = data.y.cpu().numpy()
        
        def calc_metrics(pred):
            f1 = float(f1_score(y, pred, zero_division=0))
            rec = float(recall_score(y, pred, zero_division=0))
            prec = float(precision_score(y, pred, zero_division=0))
            return {'f1': round(f1, 4), 'recall': round(rec, 4), 'precision': round(prec, 4)}
            
        m1_m = calc_metrics(p1)
        m2_m = calc_metrics(p2)
        m3_m = calc_metrics(p3)
        m4_m = calc_metrics(p4)
        
        # DeepCrack metric for this image
        dc_info = dc_per_image.get(img_id, {})
        dc_m = {
            'f1': dc_info.get('f1', 0.0),
            'recall': dc_info.get('recall', 0.0),
            'precision': dc_info.get('precision', 0.0)
        }
        
        router_model = 'M4' if cat in ['Long_Elongated', 'Thin_Fine_Fissure'] else 'M1'
        router_m = m4_m if router_model == 'M4' else m1_m
        
        diff = abs(m4_m['f1'] - m1_m['f1'])
        disagree = diff >= 0.15
        if disagree: m1_m4_disagrees += 1
        
        all_f1s = {'M1': m1_m['f1'], 'M2': m2_m['f1'], 'M3': m3_m['f1'], 'M4': m4_m['f1'], 'DeepCrack': dc_m['f1']}
        best_overall = max(all_f1s, key=all_f1s.get)
        
        items.append({
            'index': i,
            'id': img_id,
            'category': cat,
            'aspect_ratio': round(ar, 2),
            'area_fraction': round(area_frac, 4),
            'n_nodes': len(y),
            'n_cracks': int((y == 1).sum()),
            'm1': m1_m,
            'm2': m2_m,
            'm3': m3_m,
            'm4': m4_m,
            'deepcrack': dc_m,
            'router': {
                'recommended': router_model,
                'metrics': router_m
            },
            'best_model': best_overall,
            'best_f1': all_f1s[best_overall],
            'f1_diff': round(m4_m['f1'] - m1_m['f1'], 4),
            'abs_diff': round(diff, 4),
            'disagree': disagree,
            'winner': 'M4' if m4_m['f1'] > m1_m['f1'] + 0.05 else ('M1' if m1_m['f1'] > m4_m['f1'] + 0.05 else 'Tie')
        })

out_json = os.path.join(DASHBOARD_DIR, 'predictions_data.json')
with open(out_json, 'w') as f:
    json.dump(items, f, indent=2)

print(f'Successfully saved {len(items)} multi-model image comparisons (with DeepCrack) to {out_json}.')
print(f'M1 vs M4 disagree on {m1_m4_disagrees} images ({m1_m4_disagrees/len(items)*100:.1f}%).')
