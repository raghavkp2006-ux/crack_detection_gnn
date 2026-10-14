"""Scene-clustered inference using immutable primary prediction artifacts."""
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage.morphology import skeletonize

from common import BASE, OUT, DC, SEEDS, get_segments, read_mask, find
from stats_core import load_labels, load_probs, counts, f1_from, iou_from, holm, decide

KEYS = ['M1_focal','M2_focal','M3_focal','M4_focal','DeepCrack','M1_wce']
PAIRS = [('M4_focal','M1_focal'),('M4_focal','M2_focal'),('M4_focal','M3_focal'),('M3_focal','M2_focal'),('DeepCrack','M4_focal')]
TPAIRS = [('M4_focal','M2_focal'),('M3_focal','M2_focal'),('DeepCrack','M4_focal')]
B = 2000
source = lambda s: re.sub(r'-\d+$', '', s)


def ci(v): return np.percentile(v, [2.5,97.5]).tolist()
def boot_p(v): return float(min(1.0,max(1/B,2*min(np.mean(v<=0),np.mean(v>=0)))))
def ms(v): return dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),per_seed=np.asarray(v).tolist())
def scene_groups(stems, ix):
    names=sorted({source(stems[i]) for i in ix})
    return names,[np.array([i for i in ix if source(stems[i])==s],dtype=int) for s in names]


def pooled(c,ix): return f1_from(*(c[n][:,ix].sum(1) for n in ['tp','fp','fn']))

def bootstrap_counts(c, groups, sampled):
    sums={n:np.stack([c[n][:,g].sum(1) for g in groups],axis=1) for n in ['tp','fp','fn']}
    return f1_from(*(sums[n][:,sampled].sum(-1) for n in ['tp','fp','fn'])).mean(0)


def pair_tests(image_f1, groups):
    rows=[]
    for a,b in PAIRS:
        d=np.array([(image_f1[a][g]-image_f1[b][g]).mean() for g in groups])
        sd=d.std(ddof=1)
        t=stats.ttest_1samp(d,0) if np.any(d) else None
        w=float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0
        rows.append(dict(a=a,b=b,n_scenes=len(groups),mean_scene_difference=float(d.mean()),
                         cohen_dz=float(d.mean()/sd) if sd>0 else 0.0,
                         t_p=float(t.pvalue) if t is not None else 1.0,wilcoxon_p=w,
                         scene_mean_differences=d.tolist()))
    for r,p in zip(rows,holm([r['wilcoxon_p'] for r in rows])): r['wilcoxon_holm_p']=float(p)
    return rows


# Identical empty-GT and empty-prediction definitions to topology.py.
def cldice(pr, gt, sg):
    if sg.sum() == 0: return np.nan
    sp = skeletonize(pr); tp_ = (sp & gt).sum() / sp.sum() if sp.sum() > 0 else 0.0; ts = (sg & pr).sum() / sg.sum()
    return 0.0 if tp_ + ts == 0 else 2 * tp_ * ts / (tp_ + ts)


def table(lines,headers,rows):
    lines += ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
    lines += ['| '+' | '.join(str(x) for x in row)+' |' for row in rows]
    lines += ['']


def markdown(r, old, oldtop):
    lines=['# Scene-clustered statistics','',f'Images: {r["n_images"]}; source scenes: {r["n_scenes"]}; cluster-bootstrap resamples: {B}.',
           'Bootstrap RNG: numpy.default_rng(0). Seeds are fixed; scenes are resampled with replacement and all their images retained.',
           'Overall estimates pool node counts within each seed and average F1 across seeds. Scene-test differences first average image F1 over seeds, then equally average within scenes.',
           'PR-AUC means average precision. SD uses ddof=1. Tests are two-sided; significance comparisons use 0.05.', '', '## 1. Overall','']
    table(lines,['Model','Pooled node F1, seed mean','Scene-bootstrap 95% CI'],[[k,v['f1_mean'],v['ci95']] for k,v in r['overall'].items()])
    lines += ['## 2. Pairwise family','', 'Scene-test means and pooled bootstrap differences are different estimands; both are reported explicitly.','']
    table(lines,['A - B','Scene mean difference','d_z','t p','Wilcoxon p','Holm p','Pooled F1 difference','Pooled scene-bootstrap 95% CI','Bootstrap p'],
          [[v['a']+' - '+v['b'],v['mean_scene_difference'],v['cohen_dz'],v['t_p'],v['wilcoxon_p'],v['wilcoxon_holm_p'],v['pooled_f1_difference'],v['pooled_ci95'],v['bootstrap_p']] for v in r['pairs']])
    lines += ['## 3. Morphology','']
    table(lines,['Bucket','Images','Scenes','Model','F1','Scene-bootstrap 95% CI'],
          [[b,v['n_images'],v['n_scenes'],k,m['f1'],m['ci95']] for b,v in r['buckets'].items() for k,m in v['models'].items()])
    table(lines,['Bucket','A - B','Pooled F1 difference','Scene-bootstrap 95% CI','Bootstrap p'],
          [[b,k,d['difference'],d['ci95'],d['p_boot']] for b,v in r['buckets'].items() for k,d in v['diffs'].items()])
    lines += ['## 4. Topology','', 'clDice is averaged over valid images and seeds, using the original topology.py mask/skeleton definitions. Connectivity definitions are unchanged.','']
    table(lines,['Model or difference','clDice estimate','Scene-bootstrap 95% CI'],
          [[k,v['mean'],v['ci95']] for k,v in r['topology']['models'].items()]+[[k,v['difference'],v['ci95']] for k,v in r['topology']['diffs'].items()])
    leak=r['leakage_sensitivity']; lines += ['## 5. Leakage sensitivity','',f'Retained images: {leak["n_images"]}; retained scenes: {leak["n_scenes"]}.',
            'Excluded stems: '+', '.join(leak['excluded_stems'])+'.','', 'Metrics are seed mean +/- seed SD; all primary thresholds are reused.','']
    metrics=['f1','iou','precision','recall','roc_auc','pr_auc']
    table(lines,['Model']+metrics,[[k]+[str(v[m]['mean'])+' +/- '+str(v[m]['sd']) for m in metrics] for k,v in leak['overall'].items()])
    table(lines,['A - B','Full scene mean difference','Excluded scene mean difference','Full Holm p','Excluded Holm p','Sign changed','Significance changed'],
          [[v['a']+' - '+v['b'],v['full_mean_scene_difference'],v['mean_scene_difference'],v['full_wilcoxon_holm_p'],v['wilcoxon_holm_p'],v['sign_changed'],v['significance_changed']] for v in leak['pairs']])
    lines += ['## Ten-line summary','']
    oldpairs={(v['a'],v['b']):v for v in old['pairs']}
    for i,v in enumerate(r['pairs'],1):
        p0=oldpairs[v['a'],v['b']]['p_wilcoxon_holm']; p1=v['wilcoxon_holm_p']
        state='remains significant' if p0<.05 and p1<.05 else 'loses significance' if p0<.05 else 'becomes significant' if p1<.05 else 'remains non-significant'
        lines += [f'{i}. {v["a"]} - {v["b"]}: {state}; image Holm p={p0}; scene Holm p={p1}.']
    for i,(a,b) in enumerate(TPAIRS,6):
        key=a+'-'+b; oldci=oldtop['diffs'][key]['ci95']; newci=r['topology']['diffs'][key]['ci95']
        excludes=lambda c:c[0]>0 or c[1]<0
        state='remains significant' if excludes(oldci) and excludes(newci) else 'loses significance' if excludes(oldci) else 'becomes significant' if excludes(newci) else 'remains non-significant'
        lines += [f'{i}. clDice {key}: {state}; image CI={oldci}; scene CI={newci}.']
    changed=[v['a']+'-'+v['b'] for v in leak['pairs'] if v['sign_changed'] or v['significance_changed']]
    lines += [f'9. Leakage exclusion sign/significance changes relative to full scene tests: {", ".join(changed) if changed else "none"}.',
              '10. Stored model seeds, predictions, thresholds, splits, grids and existing result files are unchanged.']
    return '\n'.join(lines)+'\n'


def main():
    started=time.perf_counter(); out=Path(OUT)
    for name in ['analysis_cluster.json','CLUSTER_NUMBERS.md']:
        if (out/name).exists(): raise FileExistsError(out/name)
    read=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
    old=read('analysis_main.json'); oldtop=read('topology.json'); buckets=read('test_buckets.json')
    y,off,stems=load_labels(); assert buckets['stems']==stems
    all_ix=np.arange(len(stems)); scenes,groups=scene_groups(stems,all_ix)
    assert len(stems)==237 and len(scenes)==60
    rng=np.random.default_rng(0); sampled=rng.integers(0,len(groups),(B,len(groups)))
    C={k:counts(k) for k in KEYS}; image_f1={k:f1_from(c['tp'],c['fp'],c['fn']).mean(0) for k,c in C.items()}
    boot={k:bootstrap_counts(c,groups,sampled) for k,c in C.items()}
    r=dict(n_images=len(stems),n_scenes=len(scenes),n_bootstrap=B,bootstrap_rng_seed=0,seeds=SEEDS,source_ids=scenes,overall={},pairs=pair_tests(image_f1,groups),buckets={})
    for k,c in C.items(): r['overall'][k]=dict(f1_mean=float(pooled(c,all_ix).mean()),ci95=ci(boot[k]))
    for row in r['pairs']:
        a,b=row['a'],row['b']; d=boot[a]-boot[b]
        row.update(pooled_f1_difference=r['overall'][a]['f1_mean']-r['overall'][b]['f1_mean'],pooled_ci95=ci(d),bootstrap_p=boot_p(d))
    bk=np.array(buckets['bucket'])
    for b in ['Long','Branched','Thin']:
        ix=np.flatnonzero(bk==b); names,gs=scene_groups(stems,ix)
        sb=rng.integers(0,len(gs),(B,len(gs))); bb={k:bootstrap_counts(c,gs,sb) for k,c in C.items()}
        v=dict(n_images=len(ix),n_scenes=len(gs),models={},diffs={})
        for k,c in C.items(): v['models'][k]=dict(f1=float(pooled(c,ix).mean()),ci95=ci(bb[k]))
        for a,z in PAIRS:
            d=bb[a]-bb[z]; v['diffs'][a+'-'+z]=dict(difference=v['models'][a]['f1']-v['models'][z]['f1'],ci95=ci(d),p_boot=boot_p(d))
        r['buckets'][b]=v
    # Aggregate files did not save per-image clDice, so recompute with identical definitions.
    P={k:[load_probs(k,s) for s in SEEDS] for k in KEYS}
    CL={k:np.full((len(SEEDS),len(stems)),np.nan) for k in KEYS}
    for i,s in enumerate(stems):
        seg=get_segments(s,'test'); labels,inv=np.unique(seg,return_inverse=True); inv=inv.reshape(seg.shape)
        gt=read_mask(find(str(Path(DC,'test_lab')),s))>.5
        assert seg.shape==gt.shape and len(labels)==off[i+1]-off[i],s
        sg=skeletonize(gt)
        for k in KEYS:
            for si,(p,t) in enumerate(P[k]): CL[k][si,i]=cldice(decide(p[off[i]:off[i+1]],t)[inv],gt,sg)
        if i%40==0:print(f'clDice {i}/{len(stems)}',flush=True)
    r['topology']=dict(models={},diffs={}); cb={}
    valid=~np.isnan(CL[KEYS[0]][0]); denominators=np.array([valid[g].sum() for g in groups])
    for k,cl in CL.items():
        per_image=np.zeros(len(stems));per_image[valid]=cl[:,valid].mean(0)
        sums=np.array([per_image[g].sum() for g in groups])
        cb[k]=sums[sampled].sum(1)/denominators[sampled].sum(1)
        r['topology']['models'][k]=dict(mean=float(cl[:,valid].mean()),ci95=ci(cb[k]))
    for a,b in TPAIRS:r['topology']['diffs'][a+'-'+b]=dict(difference=r['topology']['models'][a]['mean']-r['topology']['models'][b]['mean'],ci95=ci(cb[a]-cb[b]))
    train_sources={source(p.stem) for p in Path(BASE,'data/graphs/train').glob('*.pt')}
    excluded=[s for s in stems if source(s) in train_sources];assert len(excluded)==19
    assert set(excluded)==set(read('source_overlap.json')['overlapping_test_stems'])
    ix=np.array([i for i,s in enumerate(stems) if s not in set(excluded)]); sn,sgs=scene_groups(stems,ix)
    node_mask=np.repeat(np.isin(all_ix,ix),np.diff(off)); yy=y[node_mask]
    leak=dict(excluded_stems=excluded,n_images=len(ix),n_scenes=len(sn),overall={},pairs=pair_tests(image_f1,sgs))
    for k,c in C.items():
        tp,fp,fn=(c[n][:,ix].sum(1) for n in ['tp','fp','fn'])
        auc=[];ap=[]
        for p,_ in P[k]: auc.append(roc_auc_score(yy,p[node_mask]));ap.append(average_precision_score(yy,p[node_mask]))
        leak['overall'][k]=dict(f1=ms(f1_from(tp,fp,fn)),iou=ms(iou_from(tp,fp,fn)),precision=ms(tp/np.maximum(tp+fp,1)),recall=ms(tp/np.maximum(tp+fn,1)),roc_auc=ms(auc),pr_auc=ms(ap))
    for row,full in zip(leak['pairs'],r['pairs']):
        row.update(full_mean_scene_difference=full['mean_scene_difference'],full_wilcoxon_holm_p=full['wilcoxon_holm_p'],
                   sign_changed=bool(np.sign(row['mean_scene_difference'])!=np.sign(full['mean_scene_difference'])),
                   significance_changed=bool((row['wilcoxon_holm_p']<.05)!=(full['wilcoxon_holm_p']<.05)))
    r['leakage_sensitivity']=leak;r['wall_seconds']=time.perf_counter()-started
    with (out/'analysis_cluster.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False)
    with (out/'CLUSTER_NUMBERS.md').open('x',encoding='utf-8') as f:f.write(markdown(r,old,oldtop))
    print(f'Saved scene-clustered results: {out}',flush=True)


if __name__=='__main__': main()
