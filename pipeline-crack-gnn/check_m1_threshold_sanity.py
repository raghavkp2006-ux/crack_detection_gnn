# Sanity check M1 AIGLE_RN threshold across multiple random 20/80 splits
import os, glob, torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset
from src.train import load_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITS_GRAPH_DIR = os.path.join(BASE_DIR, 'data', 'graphs', 'tits')
tits_dataset = CrackGraphDataset(TITS_GRAPH_DIR)
tits_files = sorted(glob.glob(os.path.join(TITS_GRAPH_DIR, '*.pt')))

def get_sensor(fname):
    fn = os.path.basename(fname).lower()
    if 'aigle' in fn: return 'AIGLE_RN'
    if 'esar' in fn: return 'ESAR'
    if 'lcms' in fn: return 'LCMS'
    return 'Other'

pos_indices = [i for i, f in enumerate(tits_files) if get_sensor(f) in ['AIGLE_RN', 'ESAR', 'LCMS']]
pos_sensors = [get_sensor(tits_files[i]) for i in pos_indices]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg1 = load_config('configs/baseline_deep_gnn.yaml')
m1 = build_model(cfg1).to(device)
m1.load_state_dict(torch.load('results/M1_deep_gnn/best_model.pt', map_location=device))
m1.eval()

cfg4 = load_config('configs/step2_loss/hybrid_full_focal.yaml')
m4 = build_model(cfg4).to(device)
m4.load_state_dict(torch.load('results/step2_loss/M4_hybrid_full/best_model.pt', map_location=device))
m4.eval()

# Check threshold response on AIGLE_RN alone across sweeps
aigle_idxs = [pos_indices[i] for i, s in enumerate(pos_sensors) if s == 'AIGLE_RN']
aigle_sub = Subset(tits_dataset, aigle_idxs)
aigle_loader = DataLoader(aigle_sub, batch_size=1, shuffle=False)

labels, m1_probs, m4_probs = [], [], []
with torch.no_grad():
    for d in aigle_loader:
        d = d.to(device)
        labels.append(d.y.cpu().numpy())
        m1_probs.append(torch.softmax(m1(d.x, d.edge_index), dim=1)[:, 1].cpu().numpy())
        m4_probs.append(torch.softmax(m4(d.x, d.edge_index, d.pos), dim=1)[:, 1].cpu().numpy())

flat_y = np.concatenate(labels)
flat_m1 = np.concatenate(m1_probs)
flat_m4 = np.concatenate(m4_probs)

print(f'AIGLE_RN (Total N={len(aigle_idxs)} graphs, {len(flat_y)} nodes, {flat_y.sum()} crack nodes):')
print(f'M1 Probability Range on AIGLE: min={flat_m1.min():.4f}, max={flat_m1.max():.4f}, mean={flat_m1.mean():.4f}, p95={np.percentile(flat_m1, 95):.4f}')
print(f'M4 Probability Range on AIGLE: min={flat_m4.min():.4f}, max={flat_m4.max():.4f}, mean={flat_m4.mean():.4f}, p95={np.percentile(flat_m4, 95):.4f}')

print('\nSweep Tau for M1 on AIGLE_RN:')
for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.85]:
    p = (flat_m1 >= tau).astype(int)
    f1 = f1_score(flat_y, p, zero_division=0)
    rec = recall_score(flat_y, p, zero_division=0)
    prec = precision_score(flat_y, p, zero_division=0)
    print(f'  Tau={tau:.2f} | F1={f1:.4f} | Rec={rec:.4f} | Prec={prec:.4f} | PredPos={p.sum()}')

print('\nSweep Tau for M4 on AIGLE_RN:')
for tau in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.82, 0.85]:
    p = (flat_m4 >= tau).astype(int)
    f1 = f1_score(flat_y, p, zero_division=0)
    rec = recall_score(flat_y, p, zero_division=0)
    prec = precision_score(flat_y, p, zero_division=0)
    print(f'  Tau={tau:.2f} | F1={f1:.4f} | Rec={rec:.4f} | Prec={prec:.4f} | PredPos={p.sum()}')

print('\nTesting 3 different random 20/80 calibration splits for M1 across all positive sensors:')
for seed in [42, 100, 2026]:
    rng = np.random.default_rng(seed)
    cal_idxs, ev_idxs = [], []
    for s in ['AIGLE_RN', 'ESAR', 'LCMS']:
        s_idxs = [pos_indices[k] for k, sn in enumerate(pos_sensors) if sn == s]
        rng.shuffle(s_idxs)
        n_cal = max(1, int(len(s_idxs) * 0.20))
        cal_idxs.extend(s_idxs[:n_cal])
        ev_idxs.extend(s_idxs[n_cal:])
    
    # Evaluate calibration threshold
    cal_loader = DataLoader(Subset(tits_dataset, cal_idxs), batch_size=1, shuffle=False)
    ev_loader = DataLoader(Subset(tits_dataset, ev_idxs), batch_size=1, shuffle=False)
    
    cal_y, cal_pr = [], []
    with torch.no_grad():
        for d in cal_loader:
            d = d.to(device)
            cal_y.append(d.y.cpu().numpy())
            cal_pr.append(torch.softmax(m1(d.x, d.edge_index), dim=1)[:, 1].cpu().numpy())
    cal_y = np.concatenate(cal_y)
    cal_pr = np.concatenate(cal_pr)
    
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.1, 0.92, 0.02):
        sc = f1_score(cal_y, (cal_pr >= t).astype(int), zero_division=0)
        if sc > best_f: best_f, best_t = sc, float(t)
        
    # Evaluate on eval_loader for AIGLE only
    ev_aigle_idxs = [i for i in ev_idxs if get_sensor(tits_files[i]) == 'AIGLE_RN']
    ev_aigle_loader = DataLoader(Subset(tits_dataset, ev_aigle_idxs), batch_size=1, shuffle=False)
    ey, epr = [], []
    with torch.no_grad():
        for d in ev_aigle_loader:
            d = d.to(device)
            ey.append(d.y.cpu().numpy())
            epr.append(torch.softmax(m1(d.x, d.edge_index), dim=1)[:, 1].cpu().numpy())
    ey = np.concatenate(ey)
    epr = np.concatenate(epr)
    f1_aigle = f1_score(ey, (epr >= best_t).astype(int), zero_division=0)
    print(f'Seed {seed}: Calib Tau*={best_t:.2f} (Calib F1={best_f:.4f}) -> Holdout AIGLE_RN F1 = {f1_aigle:.4f}')
