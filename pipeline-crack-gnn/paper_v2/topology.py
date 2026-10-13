import json, numpy as np, os
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from common import *; from stats_core import *

KEYS = [k for k in ['M1_focal', 'M2_focal', 'M3_focal', 'M4_focal', 'DeepCrack', 'M1_wce'] if complete_probs(k)]
if not KEYS: raise SystemExit('No complete five-seed predictions; run predict_all.py first.')
y, off, stems = load_labels(); I = len(stems); S8 = np.ones((3, 3), int)
DATA = {k: [load_probs(k, s) for s in SEEDS] for k in KEYS}

def cldice(pr, gt, sg):
    if sg.sum() == 0: return np.nan
    sp = skeletonize(pr); tp_ = (sp & gt).sum() / sp.sum() if sp.sum() > 0 else 0.0; ts = (sg & pr).sum() / sg.sum()
    return 0.0 if tp_ + ts == 0 else 2 * tp_ * ts / (tp_ + ts)

def conn(pr, gl, objs, min_area=20, thr=0.5):
    pl, _ = ndi.label(pr, structure=S8); ok = tot = 0
    for c, sl in enumerate(objs, 1):
        comp = gl[sl] == c; a = comp.sum()
        if a < min_area: continue
        tot += 1; cnt = np.bincount(pl[sl][comp]); cnt[0] = 0
        if cnt.max() / a >= thr: ok += 1
    return ok, tot

R = {k: dict(cl=np.full((len(SEEDS), I), np.nan), ok=np.zeros((len(SEEDS), I)), tot=np.zeros((len(SEEDS), I))) for k in KEYS}
for i, s in enumerate(stems):
    seg = get_segments(s, 'test'); lab, inv = np.unique(seg, return_inverse=True); inv = inv.reshape(seg.shape)
    assert len(lab) == off[i + 1] - off[i], s
    gt = read_mask(find(os.path.join(DC, 'test_lab'), s)) > 0.5
    assert gt.shape == seg.shape, f'mask/segments shape mismatch: {s}'
    sg = skeletonize(gt)
    gl, _ = ndi.label(gt, structure=S8); objs = ndi.find_objects(gl)
    for k in KEYS:
        for si, (p, t) in enumerate(DATA[k]):
            pr = decide(p[off[i]:off[i + 1]], t)[inv]
            R[k]['cl'][si, i] = cldice(pr, gt, sg); R[k]['ok'][si, i], R[k]['tot'][si, i] = conn(pr, gl, objs)
    if i % 20 == 0: print(i, '/', I, flush=True)

valid = ~np.isnan(R[KEYS[0]]['cl'][0]); rng = np.random.default_rng(0); B = 1000
ix = np.where(valid)[0]; cols = ix[rng.integers(0, len(ix), (B, len(ix)))]
out = {}
for k in KEYS:
    cl = R[k]['cl'][:, valid]; ok, tot = R[k]['ok'], R[k]['tot']
    per_img = np.nanmean(R[k]['cl'], 0); bs = np.nanmean(per_img[cols], 1)
    macro = np.where(tot > 0, ok / np.maximum(tot, 1), np.nan)
    out[k] = dict(cldice_seed_mean=float(cl.mean(1).mean()), cldice_seed_sd=float(cl.mean(1).std(ddof=1)), cldice_img_ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                  conn_micro_seed_mean=float((ok.sum(1) / tot.sum(1)).mean()), conn_micro_seed_sd=float((ok.sum(1) / tot.sum(1)).std(ddof=1)),
                  conn_macro_seed_mean=float(np.nanmean(macro, 1).mean()), conn_macro_seed_sd=float(np.nanmean(macro, 1).std(ddof=1)))
diffs = {}
for a, b in [('M4_focal', 'M2_focal'), ('M3_focal', 'M2_focal'), ('DeepCrack', 'M4_focal')]:
    if a in R and b in R:
        pa, pb = np.nanmean(R[a]['cl'], 0), np.nanmean(R[b]['cl'], 0); d = np.nanmean(pa[cols], 1) - np.nanmean(pb[cols], 1)
        diffs[f'{a}-{b}'] = dict(mean=float(d.mean()), ci95=np.percentile(d, [2.5, 97.5]).tolist())
json.dump(dict(models=out, diffs=diffs, n_images_cldice=int(valid.sum()), n_gt_components=float(R[KEYS[0]]['tot'][0].sum())),
          open(f'{OUT}/topology.json', 'w'), indent=2)
