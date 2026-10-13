import os, sys, glob
import numpy as np
import skimage.io

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # pipeline-crack-gnn
sys.path.insert(0, BASE)
OUT = os.path.join(BASE, 'results', 'paper_v2')
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 456, 789, 2026]
DC = os.path.abspath(os.path.join(BASE, '..', 'DeepCrack'))   # raw DeepCrack folder (gitignored)

def rel(p): return os.path.join(BASE, p)

# key -> (config, loss_type, run_root). Existing runs are reused (checkpoints verified in predict_all.py).
GRAPH_CELLS = {
 'M1_wce':   ('configs/baseline_deep_gnn.yaml',               'weighted_ce', 'results/multiseed/M1_deep_gnn'),
 'M1_focal': ('configs/baseline_deep_gnn.yaml',               'focal',       'results/paper_v2/runs/M1_focal'),
 'M2_wce':   ('configs/shallow_gnn.yaml',                     'weighted_ce', 'results/paper_v2/runs/M2_wce'),
 'M2_focal': ('configs/step2_loss/shallow_gnn_focal.yaml',    'focal',       'results/multiseed/M2_shallow_gnn'),
 'M3_wce':   ('configs/hybrid_no_pe.yaml',                    'weighted_ce', 'results/paper_v2/runs/M3_wce'),
 'M3_focal': ('configs/step2_loss/hybrid_no_pe_focal.yaml',   'focal',       'results/multiseed/M3_hybrid_no_pe'),
 'M4_wce':   ('configs/hybrid_full.yaml',                     'weighted_ce', 'results/paper_v2/runs/M4_wce'),
 'M4_focal': ('configs/step2_loss/hybrid_full_focal.yaml',    'focal',       'results/multiseed/M4_hybrid_full'),
}

def stem(p): return os.path.splitext(os.path.basename(p))[0]

def find(dir_, s):
    for e in ('.jpg', '.png', '.jpeg', '.bmp'):
        p = os.path.join(dir_, s + e)
        if os.path.exists(p): return p
    raise FileNotFoundError(os.path.join(dir_, s))

def read_rgb(p):
    im = skimage.io.imread(p)
    if im.ndim == 2: im = np.stack([im] * 3, -1)
    return im[..., :3]

def read_mask(p):
    m = skimage.io.imread(p)
    if m.ndim == 3: m = m[..., 0]
    return (m > 127).astype(np.float32)          # DeepCrack polarity: crack = white

_SP = None
def get_segments(s, split):
    """Cached SLIC segments if present, else recompute with the paper settings (deterministic)."""
    global _SP
    if _SP is None:
        # Other datasets (and test_single) may reuse DeepCrack stems.
        _SP = {(part, stem(p)): p for part in ('train', 'test')
               for p in glob.glob(os.path.join(BASE, 'data', 'superpixels', part, '*.npy'))}
    if (split, s) in _SP: return np.load(_SP[split, s])
    from src.preprocessing.superpixel import compute_superpixels
    return compute_superpixels(read_rgb(find(os.path.join(DC, f'{split}_img'), s)), 300, 10.0, 1.0)

def bucket_meta(stems):
    """Same rule as run_multiseed_experiment.classify_crack_geometry: thin -> long -> branched."""
    out = []
    for s in stems:
        m = read_mask(find(os.path.join(DC, 'test_lab'), s)) > 0.5
        pts = np.argwhere(m)
        if len(pts) == 0: out.append(('Clean', 0.0, 0.0)); continue
        h = pts[:, 0].max() - pts[:, 0].min() + 1; w = pts[:, 1].max() - pts[:, 1].min() + 1
        ar = max(h, w) / (min(h, w) + 1e-5); af = len(pts) / m.size
        b = 'Thin' if af < 0.025 else ('Long' if ar >= 2.0 else 'Branched')
        out.append((b, float(ar), float(af)))
    return out
