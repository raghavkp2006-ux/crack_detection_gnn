"""Eight requested IEEE-size figures from immutable saved results and images."""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

from common import OUT, DC, SEEDS, find, get_segments
from stats_core import load_labels, load_probs, counts, f1_from, decide

PRIMARY=['M1_focal','M2_focal','M3_focal','M4_focal','DeepCrack']
COLORS={'M1':'#0072B2','M2':'#E69F00','M3':'#009E73','M4':'#D55E00','DeepCrack':'#CC79A7'}
NAMES={k:k.split('_')[0] for k in PRIMARY}
NAMES['M1_wce']='M1 WCE'
color=lambda k:COLORS[k.split('_')[0]]
FIGURES=['fig04_threshold_calibration','fig05_multiseed_benchmark','fig06_loss_comparison',
         'fig07_accuracy_cost_tradeoff','fig09_topology_metrics','fig10_morphology_stratified_f1',
         'fig14_efficiency_stages','fig11_qualitative_results']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.titlesize':8.5,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})


def single(height=3.0):
    fig,ax=plt.subplots(figsize=(3.5,height));fig.subplots_adjust(left=.18,right=.98,bottom=.21,top=.96)
    return fig,ax


def save(fig,name,directory):
    fig.savefig(directory/(name+'.png'),dpi=300,facecolor='white')
    fig.savefig(directory/(name+'.pdf'),facecolor='white')
    plt.close(fig); print('Saved '+name,flush=True)


def interval(ax,x,point,bounds,horizontal=False):
    lo,hi=bounds
    if horizontal:
        ax.hlines(x,lo,hi,color='#222222',lw=.8);ax.vlines([lo,hi],x-.05,x+.05,color='#222222',lw=.8)
    else:
        ax.vlines(x,lo,hi,color='#222222',lw=.8);ax.hlines([lo,hi],x-.025,x+.025,color='#222222',lw=.8)


def main():
    started=time.perf_counter();out=Path(OUT);directory=out/'figures'
    targets=[directory/(n+e) for n in FIGURES for e in ['.png','.pdf']]+[directory/'captions_notes.md',directory/'qualitative_selection.json']
    for p in targets:
        if p.exists():raise FileExistsError(p)
    read=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
    A=read('analysis_main.json');C=read('analysis_cluster.json');T=read('topology.json');E=read('efficiency.json');S=read('sensitivity_tau.json');BK=read('test_buckets.json')
    directory.mkdir(exist_ok=True)
    notes=['# Figure captions and sources','', 'All graph-model primary plots use focal loss. Model colors are fixed across figures. Fonts are at least 8 pt; widths are 3.5 in or 7.16 in. PNGs are 300 dpi; PDFs retain vector plotting elements.','']
    # 04: shared threshold grid, original validation curves, seed range.
    fig,ax=single(3.0);fig.subplots_adjust(bottom=.33)
    grid=np.array(S['threshold_grid'])
    for k in PRIMARY:
        rows=sorted([r for r in S['runs'] if r['model']==k],key=lambda r:SEEDS.index(r['seed']))
        v=np.array([r['validation_f1_by_tau'] for r in rows])
        ax.fill_between(grid,v.min(0),v.max(0),color=color(k),alpha=.09,lw=0)
        ax.plot(grid,v.mean(0),color=color(k),lw=1.3,label=NAMES[k])
    ax.axvline(.90,color='#555555',ls=':',lw=1,label='0.90 edge')
    ax.set(xlabel='Validation threshold',ylabel='Validation node F1',xlim=(.10,.98),ylim=(0,1))
    ax.set_xticks([.1,.3,.5,.7,.9]);ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.32),ncol=3,frameon=False,columnspacing=.85,handlelength=1.35)
    save(fig,FIGURES[0],directory)
    notes += ['## fig04_threshold_calibration','Mean validation F1 across the five fixed seeds; shading is the seed minimum–maximum, not a confidence interval. Dotted vertical line marks the primary grid edge 0.90. Source: sensitivity_tau.json.','']
    # 05: metric hatching, model color, deterministic seed-point offsets.
    fig,ax=single(3.0);x=np.arange(5);w=.34;jitter=np.linspace(-.09,.09,5)
    for i,k in enumerate(PRIMARY):
        for j,m in enumerate(['f1','iou']):
            v=A['overall'][k][m];xx=i+(j-.5)*w
            ax.bar(xx,v['mean'],width=w,color=color(k),alpha=1 if j==0 else .45,hatch='' if j==0 else '//',edgecolor=color(k),linewidth=.6)
            ax.errorbar(xx,v['mean'],yerr=v['sd'],color='#222222',fmt='none',capsize=2,lw=.75)
            ax.scatter(xx+jitter*.7,v['per_seed'],s=8,facecolor='white',edgecolor='#222222',linewidth=.5,zorder=4)
    ax.set(ylabel='Test node score',ylim=(0,1.03));ax.set_xticks(x,['M1','M2','M3','M4','Deep\nCrack']);ax.grid(axis='y',alpha=.15)
    ax.legend(handles=[Patch(facecolor='#777777',label='F1'),Patch(facecolor='#bbbbbb',hatch='//',label='IoU')],frameon=False,loc='upper left')
    save(fig,FIGURES[1],directory)
    notes += ['## fig05_multiseed_benchmark','Bars are seed mean; error bars are ±1 sample SD (ddof=1) over five seeds. White points are all five seed values at deterministic horizontal offsets. Sources: analysis_main.json, overall.*.{f1,iou}.','']
    # 06: full stored loss grid.
    fig,ax=single(3.0)
    for i in range(4):
        for j,loss in enumerate(['wce','focal']):
            k=f'M{i+1}_{loss}';v=A['overall'][k]['f1'];xx=i+(j-.5)*w
            ax.bar(xx,v['mean'],width=w,color=color(k),alpha=.45 if j==0 else 1,hatch='//' if j==0 else '',edgecolor=color(k),linewidth=.6)
            ax.errorbar(xx,v['mean'],yerr=v['sd'],color='#222222',fmt='none',capsize=2,lw=.75)
            ax.scatter(xx+jitter*.7,v['per_seed'],s=9,facecolor='white',edgecolor='#222222',linewidth=.5,zorder=4)
    ax.set(ylabel='Test node F1',ylim=(0,1.03));ax.set_xticks(np.arange(4),['M1','M2','M3','M4']);ax.grid(axis='y',alpha=.15)
    ax.legend(handles=[Patch(facecolor='#bbbbbb',hatch='//',label='Weighted CE'),Patch(facecolor='#777777',label='Focal')],frameon=False,loc='upper left')
    save(fig,FIGURES[2],directory)
    notes += ['## fig06_loss_comparison','Bars are mean F1 ±1 sample SD across the same five seeds for each loss/model cell. White points show every seed; hatched bars are weighted CE. Source: analysis_main.json, overall.*.f1.','']
    # 07: exact proportional marker areas; crosshairs locate tiny graph circles.
    fig,ax=single(3.0)
    positions={'M1_focal':(.19,.07),'M2_focal':(.03,.43),'M3_focal':(.43,.25),'M4_focal':(.48,.57),'DeepCrack':(.72,.88)}
    for k in PRIMARY:
        e=E['models'][k];g=e.get('gmacs_analytic_mean_graph',e['gmacs_analytic']);y=A['overall'][k]['f1']['mean']
        ax.scatter(e['params'],y,s=18*g,color=color(k),linewidth=0,zorder=3)
        if k!='DeepCrack':ax.plot(e['params'],y,marker='+',ms=4,mew=.7,color=color(k),ls='none')
        ax.annotate(NAMES[k],xy=(e['params'],y),xytext=positions[k],textcoords='axes fraction',fontsize=8.5,
                    color=color(k),arrowprops=dict(arrowstyle='-',lw=.6,color='#555555'),bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.set_xscale('log');ax.set(xlabel='Parameter count (log scale)',ylabel='Mean test node F1',xlim=(5e3,5e7),ylim=(.665,.91))
    ax.text(.02,.96,'Circle area ∝ analytic GMACs',transform=ax.transAxes,fontsize=8,va='top');ax.grid(alpha=.15)
    save(fig,FIGURES[3],directory)
    notes += ['## fig07_accuracy_cost_tradeoff','No error bars. F1 is the five-seed mean. Circle area is exactly 18 times analytic GMACs in points squared; thin crosshairs locate the very small graph-model circles, without adding an area offset. Graph sizes use gmacs_analytic_mean_graph; DeepCrack uses gmacs_analytic. Sources: analysis_main.json and efficiency.json.','']
    # 09: cluster CI only for clDice; connectivity point estimates unchanged.
    fig,ax=single(3.65);fig.subplots_adjust(left=.28,bottom=.25)
    keys=['M1_focal','M2_focal','M3_focal','M4_focal','M1_wce','DeepCrack']
    for i,k in enumerate(keys):
        t=T['models'][k];vals=[C['topology']['models'][k]['mean'],t['conn_micro_seed_mean'],t['conn_macro_seed_mean']]
        for j,v in enumerate(vals):
            yy=i+(j-1)*.22;ax.barh(yy,v,height=.19,color=color(k),alpha=[1,.65,.35][j],hatch=['','//','..'][j],edgecolor=color(k),lw=.4)
            if j==0:interval(ax,yy,v,C['topology']['models'][k]['ci95'],horizontal=True)
    ax.set_yticks(np.arange(6),['M1 focal','M2 focal','M3 focal','M4 focal','M1 WCE','DeepCrack']);ax.invert_yaxis()
    ax.set(xlabel='Topology score',xlim=(0,1));ax.grid(axis='x',alpha=.15)
    fig.legend(handles=[Patch(facecolor='#777777',label='clDice'),Patch(facecolor='#aaaaaa',hatch='//',label='Conn. micro'),Patch(facecolor='#cccccc',hatch='..',label='Conn. macro')],loc='lower center',ncol=2,frameon=False)
    save(fig,FIGURES[4],directory)
    notes += ['## fig09_topology_metrics','clDice intervals are 95% percentile scene-cluster-bootstrap CIs with 2000 resamples. Connectivity bars show seed means without uncertainty bars. Sources: analysis_cluster.json, topology.models; topology.json, models.*.conn_micro_seed_mean and conn_macro_seed_mean. M1 WCE is the secondary comparison.','']
    # 10: no callouts, annotations, or winner markers.
    fig,ax=plt.subplots(figsize=(7.16,2.8));fig.subplots_adjust(left=.09,right=.99,bottom=.25,top=.97)
    for bi,b in enumerate(['Long','Branched','Thin']):
        for mi,k in enumerate(PRIMARY):
            v=C['buckets'][b]['models'][k];xx=bi+(mi-2)*.14
            ax.bar(xx,v['f1'],width=.13,color=color(k),label=NAMES[k] if bi==0 else None)
            interval(ax,xx,v['f1'],v['ci95'])
    ax.set_xticks(range(3),['Long','Branched','Thin']);ax.set(ylabel='Test node F1',ylim=(0,1.02));ax.grid(axis='y',alpha=.15)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False)
    save(fig,FIGURES[5],directory)
    notes += ['## fig10_morphology_stratified_f1','Bars show pooled node F1 averaged over seeds. Error bars are 95% percentile cluster-bootstrap CIs, resampling only source scenes containing images of each bucket and retaining that bucket’s images from sampled scenes. Source: analysis_cluster.json, buckets.','']
    # 14: model colors + stage hatching; explicit different statistic for CNN.
    fig,ax=single(3.1);fig.subplots_adjust(left=.26,bottom=.36)
    stages=['slic','features','graph','h2d_forward_d2h','reconstruct'];hatches=['','//','..','xx','\\\\']
    labels=['SLIC','Features','Graph','Transfer + forward','Reconstruct']
    for i,k in enumerate(PRIMARY[:4]):
        left=0
        for j,s in enumerate(stages):
            v=E['models'][k]['stage_mean_ms'][s]
            ax.barh(i,v,left=left,height=.60,color=color(k),alpha=1-j*.13,hatch=hatches[j],edgecolor='#333333',lw=.25);left+=v
    ax.barh(4,E['models']['DeepCrack']['e2e_ms']['median'],height=.60,color=color('DeepCrack'),edgecolor='#333333',ls='--')
    ax.set_yticks(range(5),['M1','M2','M3','M4','DeepCrack\n(median)']);ax.invert_yaxis();ax.set_xlabel('End-to-end time (ms)');ax.grid(axis='x',alpha=.15)
    fig.legend(handles=[Patch(facecolor='#aaaaaa',hatch=hatches[j],label=labels[j]) for j in range(5)],loc='lower center',ncol=2,frameon=False,columnspacing=.8)
    save(fig,FIGURES[6],directory)
    notes += ['## fig14_efficiency_stages','DeepCrack bar is a median; graph-model bars are means. No error bars. Stage hatching encodes SLIC, features, graph construction, H2D/forward/D2H, and reconstruction; the transfer/forward stage is not model-only GPU latency. Model color remains fixed. Source: efficiency.json, models.*.stage_mean_ms and models.DeepCrack.e2e_ms.median.','']
    # 11: fixed deterministic image selection and native aspect ratio.
    qstart=time.perf_counter();y,off,stems=load_labels();assert BK['stems']==stems
    ct={k:counts(k) for k in PRIMARY};imf={k:f1_from(v['tp'],v['fp'],v['fn']) for k,v in ct.items()}
    means=imf['M4_focal'].mean(0);bk=np.array(BK['bucket']);selected=[]
    for b in ['Long','Branched','Thin']:
        ix=np.flatnonzero(bk==b);median=float(np.median(means[ix]));i=int(ix[np.argmin(np.abs(means[ix]-median))])
        selected.append(dict(row=b,index=i,stem=stems[i],bucket_median_m4_f1=median))
    i=int(np.argmin(means));selected.append(dict(row='Failure',index=i,stem=stems[i]))
    P={k:load_probs(k,42) for k in ['DeepCrack','M1_focal','M4_focal']}
    fig,axs=plt.subplots(4,5,figsize=(7.16,6.0));fig.subplots_adjust(left=.045,right=.995,top=.95,bottom=.02,wspace=.03,hspace=.10)
    for ri,row in enumerate(selected):
        i=row['index'];s=row['stem'];image=mpimg.imread(find(str(Path(DC,'test_img')),s));mask=mpimg.imread(find(str(Path(DC,'test_lab')),s))
        if mask.ndim==3:mask=mask[...,0]
        gt=mask>(.5 if np.issubdtype(mask.dtype,np.floating) else 127)
        seg=get_segments(s,'test');labs,inv=np.unique(seg,return_inverse=True);inv=inv.reshape(seg.shape)
        assert image.shape[:2]==gt.shape==seg.shape and len(labs)==off[i+1]-off[i],s
        maps=[]
        for k,(p,t) in P.items():maps.append(decide(p[off[i]:off[i+1]],t)[inv])
        for j,img in enumerate([image,gt]+maps):
            axs[ri,j].imshow(img,cmap='gray' if j else None,aspect='equal',vmin=0 if j else None,vmax=1 if j else None,interpolation='nearest' if j else 'antialiased')
            axs[ri,j].axis('off')
        axs[ri,0].text(-.10,.5,row['row'],rotation=90,transform=axs[ri,0].transAxes,ha='center',va='center',fontsize=8.5)
        row['m4_seed_averaged_f1']=float(means[i])
        row['per_model_f1_seed42']={k:float(imf[k][0,i]) for k in PRIMARY}
        row['per_model_seed_averaged_f1']={k:float(imf[k][:,i].mean()) for k in PRIMARY}
        row['native_shape']=list(seg.shape)
    for ax,title in zip(axs[0],['Input','Ground truth','DeepCrack','M1 focal','M4 focal']):ax.set_title(title,fontsize=8.5)
    save(fig,FIGURES[7],directory)
    with (directory/'qualitative_selection.json').open('x',encoding='utf-8') as f:json.dump(dict(seed=42,rule='Within each bucket, first image in stored sorted order closest to median seed-averaged M4 F1; failure is first global minimum. No deduplication or hand selection.',images=selected,wall_seconds=time.perf_counter()-qstart),f,indent=2)
    notes += ['## fig11_qualitative_results','No uncertainty bars or overlays. Columns show native RGB, white-crack GT, and node predictions reconstructed from seed 42 probabilities. Every image retains its native aspect ratio; blank space is intentional. Bucket rows use the closest-to-median seed-averaged M4 image, and failure uses the global minimum. Ties use the first stored sorted stem; duplicates are not replaced. Sources: test_labels.npz, test_buckets.json, probs/*/seed_42.npz, cached test superpixels, native DeepCrack test images/masks; qualitative_selection.json records stems and all primary-model F1 values.','',f'Figure-generation wall seconds: {time.perf_counter()-started}','']
    with (directory/'captions_notes.md').open('x',encoding='utf-8') as f:f.write('\n'.join(notes))
    print(json.dumps(selected,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
