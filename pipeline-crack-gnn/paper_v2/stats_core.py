import numpy as np
from common import OUT, SEEDS

def load_labels():
    with np.load(f'{OUT}/test_labels.npz') as z:
        return z['y'], z['offsets'], [str(s) for s in z['stems']]
def load_probs(key, seed):
    with np.load(f'{OUT}/probs/{key}/seed_{seed}.npz') as z:
        p, tau = z['probs'].astype(np.float32), float(z['tau'])
    y, _, _ = load_labels()
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or not np.isfinite(tau):
        raise ValueError(f'Invalid probabilities/threshold for {key}/{seed}')
    return p, tau
def decide(p, tau): return p >= np.float32(tau)
def seg_sum(a, off): return np.add.reduceat(a, off[:-1])
def counts(key):
    y, off, _ = load_labels(); T, Fp, Fn = [], [], []
    for s in SEEDS:
        p, t = load_probs(key, s); pr = decide(p, t)
        T.append(seg_sum((pr & (y == 1)).astype(np.int64), off)); Fp.append(seg_sum((pr & (y == 0)).astype(np.int64), off))
        Fn.append(seg_sum((~pr & (y == 1)).astype(np.int64), off))
    return dict(tp=np.array(T), fp=np.array(Fp), fn=np.array(Fn))          # each (n_seeds, n_images)
def f1_from(tp, fp, fn): return 2 * tp / np.maximum(2 * tp + fp + fn, 1)
def iou_from(tp, fp, fn): return tp / np.maximum(tp + fp + fn, 1)
def holm(p):
    p = np.asarray(p, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)): raise ValueError('Holm requires finite p-values in [0, 1]')
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

def complete_probs(key):
    from os.path import isfile
    return all(isfile(f'{OUT}/probs/{key}/seed_{s}.npz') for s in SEEDS)
