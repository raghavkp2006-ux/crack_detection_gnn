import os, json, numpy as np, scipy.stats as st
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import namedtuple
from common import *; from stats_core import *

GRAPH = [f'M{i}_{l}' for i in range(1, 5) for l in ('wce', 'focal')]
KEYS = [k for k in GRAPH + ['DeepCrack'] if complete_probs(k)]
PAIRS = [('M4_focal', 'M1_focal'), ('M4_focal', 'M2_focal'), ('M4_focal', 'M3_focal'), ('M3_focal', 'M2_focal'), ('DeepCrack', 'M4_focal')]
y, off, stems = load_labels(); C = {k: counts(k) for k in KEYS}; R = {}
def ms(v): v = np.asarray(v); return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), per_seed=v.tolist())

def paired_t(a, b):
    if np.all(np.asarray(a) == np.asarray(b)):
        return namedtuple('PairedTest', 'statistic pvalue')(0.0, 1.0)
    return st.ttest_rel(a, b)

# 1 overall
R['overall'] = {}
for k, c in C.items():
    tp, fp, fn = c['tp'].sum(1), c['fp'].sum(1), c['fn'].sum(1)
    auc, ap = [], []
    for s in SEEDS:
        p, _ = load_probs(k, s); auc.append(roc_auc_score(y, p)); ap.append(average_precision_score(y, p))
    R['overall'][k] = dict(f1=ms(f1_from(tp, fp, fn)), iou=ms(iou_from(tp, fp, fn)), precision=ms(tp / np.maximum(tp + fp, 1)),
                           recall=ms(tp / np.maximum(tp + fn, 1)), roc_auc=ms(auc), pr_auc=ms(ap))

# 2 loss grid (paired by seed)
R['loss_grid'] = {}
for m in 'M1 M2 M3 M4'.split():
    if f'{m}_wce' in C and f'{m}_focal' in C:
        a = f1_from(C[f'{m}_focal']['tp'].sum(1), C[f'{m}_focal']['fp'].sum(1), C[f'{m}_focal']['fn'].sum(1))
        b = f1_from(C[f'{m}_wce']['tp'].sum(1), C[f'{m}_wce']['fp'].sum(1), C[f'{m}_wce']['fn'].sum(1))
        t, p = paired_t(a, b)
        R['loss_grid'][m] = dict(focal=ms(a), wce=ms(b), diff_mean=float((a - b).mean()), t=float(t), p=float(p))
ps = [R['loss_grid'][m]['p'] for m in R['loss_grid']]
for m, pa in zip(R['loss_grid'], holm(ps + [1.0] * (4 - len(ps)))): R['loss_grid'][m]['p_holm'] = float(pa)

# 3 pairwise family (image level, seed-averaged per-image F1) with Holm
img_f1 = {k: f1_from(c['tp'], c['fp'], c['fn']).mean(0) for k, c in C.items()}
rows = []
for a, b in PAIRS:
    if a not in C or b not in C: continue
    d = img_f1[a] - img_f1[b]; tt = paired_t(img_f1[a], img_f1[b])
    try: w = st.wilcoxon(img_f1[a], img_f1[b]).pvalue if np.any(d) else 1.0
    except ValueError: w = 1.0
    sa = f1_from(C[a]['tp'].sum(1), C[a]['fp'].sum(1), C[a]['fn'].sum(1)); sb = f1_from(C[b]['tp'].sum(1), C[b]['fp'].sum(1), C[b]['fn'].sum(1))
    rows.append(dict(a=a, b=b, mean_diff_img=float(d.mean()), cohen_dz=float(d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else (0.0 if d.mean() == 0 else np.copysign(np.inf, d.mean()))), t_img=float(tt.statistic),
                     p_t_img=float(tt.pvalue), p_wilcoxon_img=float(w), p_seed_t=float(paired_t(sa, sb).pvalue)))
for r, pa in zip(rows, holm([r['p_wilcoxon_img'] for r in rows] + [1.0] * (len(PAIRS) - len(rows)))): r['p_wilcoxon_holm'] = float(pa)
R['pairs'] = rows

# 4 morphology buckets + image-level bootstrap
meta = bucket_meta(stems); bk = np.array([m[0] for m in meta])
json.dump(dict(stems=stems, bucket=bk.tolist(), ar=[m[1] for m in meta], area_frac=[m[2] for m in meta]), open(f'{OUT}/test_buckets.json', 'w'))
rng = np.random.default_rng(0); B = 1000
def boot(c, cols):
    tp, fp, fn = (c[n][:, cols].sum(-1) for n in ('tp', 'fp', 'fn')); return f1_from(tp, fp, fn).mean(0)   # (B,)
R['buckets'] = {}
for b in ('Long', 'Branched', 'Thin'):
    ix = np.where(bk == b)[0]
    if not len(ix):
        R['buckets'][b] = dict(n=0, models={}, diffs={}); continue
    cols = ix[rng.integers(0, len(ix), (B, len(ix)))]
    ent = dict(n=int(len(ix)), models={}, diffs={})
    bs = {k: boot(c, cols) for k, c in C.items()}
    for k, c in C.items():
        pt = f1_from(c['tp'][:, ix].sum(1), c['fp'][:, ix].sum(1), c['fn'][:, ix].sum(1)).mean()
        ent['models'][k] = dict(f1=float(pt), ci95=np.percentile(bs[k], [2.5, 97.5]).tolist())
    for a, bb in PAIRS:
        if a in bs and bb in bs:
            d = bs[a] - bs[bb]
            ent['diffs'][f'{a}-{bb}'] = dict(mean=float(d.mean()), ci95=np.percentile(d, [2.5, 97.5]).tolist(),
                                              p_boot=float(min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B))))
    R['buckets'][b] = ent
json.dump(R, open(f'{OUT}/analysis_main.json', 'w'), indent=2)
print(json.dumps({k: {m: round(v['f1']['mean'], 4) for m, v in R['overall'].items()} for k in ['f1']}, indent=1))
