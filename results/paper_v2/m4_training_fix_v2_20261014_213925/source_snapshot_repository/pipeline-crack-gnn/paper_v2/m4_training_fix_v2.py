"""Frozen-split M4 stabilization study. No consumed DeepCrack test access.

Run from any cwd with Python 3.11. TITS is read-only and scoring requires an
immutable configuration lock plus an exclusive, non-retryable consumption marker.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from improve_m2_m4 import (Experiment, BASE, ROOT, MetricSet, Recipe, GRIDS, build_model,
    FocalLoss, compute_superpixels, extract_node_features, build_pirm_graph,
    symmetrize_graph, build_pyg_data, assign_node_labels, sha, read, save, utc,
    cv2, np, torch, F, DeepCrackNet, DataLoader, projected_run_seconds)

OLD = BASE / 'results/paper_v2/improve_m2_m4_20261014_1112_02'
TITS = Path('C:/Users/ragha/Downloads/CrackDataset/TITS')
SEEDS = [42, 1, 2]
KEYS = ['strict_cr', 'cldice', 'pixel_f1']
EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def mean_sd(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values)>1 else None, n=len(values))


def summary(rows):
    return {key: mean_sd([row[key] for row in rows]) for key in KEYS}


def augmented_graph(task):
    """Fixed eight-view bank shared by C/D/seeds; image transforms precede SLIC."""
    stem, variant, directory = task
    path = Path(directory) / f'{stem}_v{variant}.pt'
    if path.exists():
        return str(path), sha(path)
    image = cv2.cvtColor(cv2.imread(str(ROOT / f'DeepCrack/train_img/{stem}.jpg')), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(ROOT / f'DeepCrack/train_lab/{stem}.png'), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f'Missing training image or mask: {stem}')
    image, mask = np.rot90(image, variant % 4), np.rot90(mask, variant % 4)
    if variant // 4:
        image, mask = np.fliplr(image), np.fliplr(mask)
    seed = int(hashlib.sha256(f'augmentation-42-{stem}-{variant}'.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    contrast, brightness = float(rng.uniform(.8, 1.2)), float(rng.uniform(-.2, .2))
    image = np.clip(image.astype(np.float32)*contrast + 255*brightness, 0, 255).astype(np.uint8)
    image, mask = np.ascontiguousarray(image), np.ascontiguousarray(mask)
    seg = compute_superpixels(image, 600, 10., 1., enforce_connectivity=False)
    features, positions = extract_node_features(image, seg)
    ei, ew = symmetrize_graph(*build_pirm_graph(features, 8, .1))
    labels = assign_node_labels(seg, mask, .3)
    graph = build_pyg_data(features, ei, ew, positions, labels)
    with path.open('xb') as stream:
        torch.save(graph, stream)
    return str(path), sha(path)


class Study(Experiment):
    def __init__(self, output):
        self.out = Path(output).resolve()
        if not self.out.is_relative_to(ROOT / 'results/paper_v2') or not self.out.name.startswith('m4_training_fix_v2_'):
            raise ValueError('Use a fresh repository-root results/paper_v2/m4_training_fix_v2_<timestamp> folder')
        if not (self.out/'claim.md').exists():
            raise ValueError('Write the preregistration before execution')
        self.step_name = 'validation'
        self.deadline = datetime.fromisoformat(read(self.out/'session.json')['started_at_utc'].replace('Z','+00:00')).timestamp()+9*3600
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_cache, self.items_cache = {}, {}
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.old_lock = read(OLD/'locked_configs.json')
        self.val_items = None
        self.native_train = None
        self.native_val = None
        self.val_metrics = None

    def check(self, estimate=0):
        if time.time()+estimate >= self.deadline:
            raise RuntimeError(f'Nine-hour budget cannot fit next mandatory operation ({estimate:.1f}s)')

    def paths(self, split, stem, labels=False):
        if split != 'train':
            raise RuntimeError('Consumed DeepCrack test access forbidden in this study')
        return super().paths(split, stem, labels)

    def verify_sources(self):
        for path, digest in self.protocol['sources'].items():
            if sha(Path(path)) != digest:
                raise ValueError(f'Source changed since preflight: {path}')

    def preflight(self):
        if (self.out/'protocol.json').exists():
            self.verify_sources()
            return
        start = time.perf_counter()
        original = read(OLD/'protocol.json')
        split = {key: original['split'][key] for key in ['train', 'val', 'unused']}
        assert [len(split[k]) for k in ['train','val','unused']] == [210,45,45]
        assert len(set(sum(split.values(), []))) == 300
        assert self.old_lock['grids'] == GRIDS
        if not (TITS/'IMAGES').is_dir() or not (TITS/'GROUND_TRUTH').is_dir():
            raise FileNotFoundError('TITS RGB/source images or native GT folder is missing')
        inventory = {str(p.relative_to(TITS)): [p.stat().st_size,p.stat().st_mtime_ns]
                     for p in TITS.rglob('*') if p.is_file()}
        pairs, groups = [], []
        for folder in sorted((TITS/'GROUND_TRUTH').iterdir()):
            if not folder.is_dir():
                continue
            images = sorted(p for p in (TITS/'IMAGES'/folder.name).iterdir() if p.suffix.lower() in EXTS)
            masks = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
            groups.append(dict(sensor=folder.name, source_images=len(images), masks=len(masks)))
            for mask in masks:
                core = mask.stem.removeprefix('GT_')
                matches = [p for p in images if p.stem.startswith('Im_GT_'+core)]
                if len(matches) != 1:
                    raise ValueError(f'Unpaired/ambiguous TITS mask: {mask}; {matches}')
                image = matches[0]
                im, gt = cv2.imread(str(image), cv2.IMREAD_UNCHANGED), cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
                if im is None or gt is None or im.size==0 or gt.size==0:
                    raise ValueError(f'Unreadable image or native GT: {image}')
                unique = np.unique(gt).tolist()
                # TITS includes one valid all-white empty GT and one antialiased
                # grayscale mask. The dataset's fixed dark-crack threshold is 127.
                if float((gt<127).mean()) >= .5 or gt.max() < 127:
                    raise ValueError(f'TITS mask polarity unexpected: {mask}')
                pairs.append(dict(stem=folder.name+'/'+core, image=str(image), mask=str(mask),
                    image_shape=list(im.shape), mask_shape=list(gt.shape),
                    image_sha256=sha(image), mask_sha256=sha(mask), mask_values=unique,
                    empty_mask=not bool((gt<127).any()), mask_polarity='black crack / white background; threshold <127'))
        if len(pairs)!=66 or len({p['image'] for p in pairs})!=66:
            raise ValueError(f'Expected 66 distinct annotated pairs; found {len(pairs)}')
        dc_hashes = {sha(p) for p in (ROOT/'DeepCrack/train_img').iterdir() if p.suffix.lower() in EXTS}
        overlap = [p['stem'] for p in pairs if p['image_sha256'] in dc_hashes]
        if overlap:
            raise ValueError(f'TITS/DeepCrack training-source byte duplicate: {overlap}')
        save(self.out/'tits_manifest.json',dict(root=str(TITS), read_only=True, sensors=groups, pairs=pairs,
            annotated_pairs=len(pairs), source_images=sum(g['source_images'] for g in groups),
            official_layout='IMAGES/<sensor>, GROUND_TRUTH/<sensor>, RESULTS/<sensor>; no train/test directories in supplied data',
            cohort='all 66 uniquely annotated pairs; unannotated/noGT images excluded, no synthetic negative masks',
            channels='grayscale inputs promoted to three identical RGB channels; RGB inputs converted BGR->RGB',
            shape_policy='historical 256x256 input; probability projected directly to native GT dimensions, including eight unequal source/mask shapes',
            differing_shape_pairs=[p['stem'] for p in pairs if p['image_shape'][:2]!=p['mask_shape']],
            dataset_distinctness='Five TITS sensor groups and different source files; no byte overlap with the 300 DeepCrack training-source images',
            no_deepcrack_training_image_duplicates=True, external_file_metadata=inventory,
            preflight_wall_seconds=time.perf_counter()-start))
        sources=[Path(__file__), HERE/'improve_m2_m4.py', HERE/'improvement_core.py',
                 BASE/'src/models/hybrid_model.py', BASE/'src/train.py',
                 BASE/'src/preprocessing/features.py', BASE/'src/preprocessing/superpixel.py',
                 BASE/'src/preprocessing/build_dataset.py', BASE/'src/preprocessing/pirm_graph.py',
                 ROOT/'crack_metrics.py', ROOT/'DeepSegmentor/models/deepcrack_networks.py',
                 OLD/'locked_configs.json', OLD/'protocol.json', self.out/'claim.md']
        save(self.out/'protocol.json',dict(created=utc(),split_seed=42,split=split,seeds=SEEDS,grids=GRIDS,
            sources={str(p):sha(p) for p in sources},
            baseline_config=next(c['descriptor']['config'] for c in self.old_lock['cases'] if c['model']=='M4' and c['seed']==42),
            per_epoch_raw_threshold=.5, checkpoint_rule='trailing 5 complete epochs mean of (clDice + strict CR)/2; latest epoch weights, earliest tie',
            replacement_rule='sequential A/B/C/D vs current incumbent; three-seed processed validation mean CR >= incumbent+0.010 and clDice >= incumbent',
            augmentation=dict(views=8,rotations=[0,90,180,270],horizontal_flip=[False,True],
                brightness=[-.2,.2],contrast=[.8,1.2],cache_seed=42,scope='train only, native image, before SLIC; choose one cached view per source per epoch'),
            ema=dict(decay=.99,update='after each optimizer step',parameters='EMA floating weights',buffers='copy current batchnorm buffers'),
            budget_hours=9,per_run_max_projected_seconds=2400,
            environment=dict(python=sys.version,torch=torch.__version__,device=str(self.dev),gpu=torch.cuda.get_device_name(0) if self.dev.type=='cuda' else None),
            source_overlap_caveat=original['existing_source_overlap_audit']))
        self.event('preflight_complete',wall_seconds=time.perf_counter()-start,pairs=len(pairs))
        print(f'PREFLIGHT: {len(pairs)} annotated TITS pairs; grayscale promoted to RGB; eight native mask/source shape differences',flush=True)

    def prepare(self):
        if self.val_items is not None:
            return
        started=time.perf_counter()
        self.val_items=self.items(self.protocol['split']['val'])
        self.val_metrics=MetricSet([i['gt'] for i in self.val_items])
        self.native_train=[]; self.native_val=[]
        for split,target in [('train',self.native_train),('val',self.native_val)]:
            for stem in self.protocol['split'][split]:
                path=OLD/f'graphs/segments_600/{stem}.pt'
                if not path.exists():
                    raise FileNotFoundError(f'Historical focal600 native graph missing: {path}')
                target.append(torch.load(path,weights_only=False).to(self.dev))
        # Match training validation node-label space independently of native pixel scoring.
        self.val_inference=[(self.graph_input(i,600)[0].to(self.dev),self.graph_input(i,600)[1]) for i in self.val_items]
        save(self.out/'training_inputs.json',dict(native_graphs=[dict(stem=s,sha256=sha(OLD/f'graphs/segments_600/{s}.pt'))
            for s in self.protocol['split']['train']+self.protocol['split']['val']],
            historical_cache_reused_read_only=True,wall_seconds=time.perf_counter()-started))
        self.event('prepare_complete',wall_seconds=time.perf_counter()-started)

    def augment(self):
        directory=self.out/'augmented_graphs'
        if (directory/'manifest.json').exists():
            return
        self.check(estimate=900)
        directory.mkdir(exist_ok=True)
        tasks=[(stem,view,str(directory)) for stem in self.protocol['split']['train'] for view in range(8)]
        start=time.perf_counter()
        print(f'AUGMENT: caching {len(tasks)} native augmented graphs',flush=True)
        with ProcessPoolExecutor(max_workers=4) as pool:
            records=[]
            for index,(path,digest) in enumerate(pool.map(augmented_graph,tasks,chunksize=4)):
                records.append(dict(path=path,sha256=digest))
                if (index+1)%100==0:
                    print(f'AUGMENT: {index+1}/{len(tasks)} {time.perf_counter()-start:.1f}s',flush=True)
                    self.check()
        save(directory/'manifest.json',dict(records=records,wall_seconds=time.perf_counter()-start,settings=self.protocol['augmentation']))
        self.event('augmentation_complete',wall_seconds=time.perf_counter()-start)

    @torch.no_grad()
    def evaluate_epoch(self, model):
        model.eval(); node_tp=node_fp=node_fn=0
        for graph in self.native_val:
            pred=model(graph.x,graph.edge_index,graph.pos).argmax(1)
            y=graph.y.long()
            node_tp+=int(((pred==1)&(y==1)).sum().item());node_fp+=int(((pred==1)&(y==0)).sum().item());node_fn+=int(((pred==0)&(y==1)).sum().item())
        probs=[]
        for graph,inverse in self.val_inference:
            probs.append(torch.softmax(model(graph.x,graph.edge_index,graph.pos),1)[:,1].cpu().numpy()[inverse])
        metrics=self.val_metrics.score(p>=np.float32(.5) for p in probs)
        metrics['node_f1']=2*node_tp/max(2*node_tp+node_fp+node_fn,1)
        return metrics

    def train(self, candidate,seed):
        directory=self.out/f'runs/{candidate}/seed_{seed}'
        if (directory/'training.json').exists():
            return read(directory/'training.json')
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'started.json').exists():
            raise RuntimeError(f'Interrupted run cannot silently restart/overwrite: {directory}')
        self.check(estimate=600)
        self.verify_sources()
        save(directory/'started.json',dict(time=utc(),candidate=candidate,seed=seed))
        cfg=copy.deepcopy(self.protocol['baseline_config']); cfg['training'].update(seed=seed,epochs=100,early_stopping_patience=None)
        settings=dict(candidate=candidate,seed=seed,config=cfg,accumulation=1 if candidate=='A' else 8,
            cosine=candidate!='A',augmentation=candidate in ['C','D'],ema=candidate=='D',raw_tau=.5,
            split_seed=42,checkpoint_rule=self.protocol['checkpoint_rule'],protocol_sha256=sha(self.out/'protocol.json'))
        save(directory/'config.json',settings)
        command=[sys.executable,str(Path(__file__).resolve()),'train-one','--output',str(self.out),'--candidate',candidate,'--seed',str(seed)]
        save(directory/'command.json',dict(argv=command,command=subprocess.list2cmdline(command),cwd=str(ROOT)))
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=build_model(cfg).to(self.dev)
        labels=torch.cat([g.y for g in self.native_train]);counts=torch.bincount(labels.long(),minlength=2).float()
        criterion=FocalLoss((counts.flip(0)/counts.sum()).to(self.dev),gamma=2.)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=cfg['training']['weight_decay'])
        if settings['cosine']:
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=99,eta_min=1e-5)
        else:
            warm=cfg['training'].get('warmup_epochs',5)
            warmup=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:(e+1)/warm if e<warm else 1.)
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5)
        ema=copy.deepcopy(model).eval() if settings['ema'] else None
        if ema is not None:
            for p in ema.parameters():p.requires_grad_(False)
        augmented=None
        if settings['augmentation']:
            augmented=[[torch.load(self.out/f'augmented_graphs/{stem}_v{view}.pt',weights_only=False).to(self.dev)
                for view in range(8)] for stem in self.protocol['split']['train']]
        history=[]; best_score=-math.inf; selected=None;run_start=time.perf_counter()
        # Historical DataLoader shuffle controls RNG order; accumulation does not batch graphs together.
        for epoch in range(1,101):
            self.check(); epstart=time.perf_counter();model.train();loss_sum=0.;n_nodes=0;optimizer_steps=0
            if augmented is not None:
                chosen=[views[np.random.randint(8)] for views in augmented]
            else:chosen=self.native_train
            loader=DataLoader(chosen,batch_size=1,shuffle=True)
            optimizer.zero_grad(set_to_none=True)
            for index,g in enumerate(loader):
                logits=model(g.x,g.edge_index,g.pos);loss=criterion(logits,g.y.long())
                if not bool(torch.isfinite(loss)):raise ValueError('Non-finite training loss')
                accumulation=settings['accumulation'];group_start=(index//accumulation)*accumulation
                divisor=min(accumulation,len(chosen)-group_start)
                (loss/divisor).backward();loss_sum+=float(loss.item())*g.num_nodes;n_nodes+=g.num_nodes
                if (index+1)%accumulation==0 or index+1==len(chosen):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training'].get('grad_clip_norm',1.))
                    optimizer.step();optimizer.zero_grad(set_to_none=True);optimizer_steps+=1
                    if ema is not None:
                        with torch.no_grad():
                            for averaged,current in zip(ema.parameters(),model.parameters()):averaged.mul_(.99).add_(current,alpha=.01)
                            for averaged,current in zip(ema.buffers(),model.buffers()):averaged.copy_(current)
            evaluated=ema if ema is not None else model
            metrics=self.evaluate_epoch(evaluated)
            smooth=None
            if len(history)>=4:
                smooth=float(np.mean([(row['cldice']+row['strict_cr'])/2 for row in history[-4:]]+[(metrics['cldice']+metrics['strict_cr'])/2]))
            improved=smooth is not None and smooth>best_score+1e-12
            checkpoint=None
            if improved or epoch%10==0 or epoch==100:
                checkpoint=directory/f'checkpoint_epoch_{epoch:03d}.pt'
                with checkpoint.open('xb') as stream:torch.save(evaluated.state_dict(),stream)
            if improved:best_score=smooth;selected=str(checkpoint)
            current_lr=float(optimizer.param_groups[0]['lr'])
            if settings['cosine']:
                if epoch<100:scheduler.step()
            elif epoch<=warm:warmup.step()
            else:scheduler.step(metrics['node_f1'])
            if self.dev.type=='cuda':torch.cuda.synchronize()
            row=dict(epoch=epoch,train_loss=loss_sum/n_nodes,**metrics,
                smoothed_connectivity_score=smooth,lr=current_lr,optimizer_steps=optimizer_steps,
                wall_seconds=time.perf_counter()-epstart,saved_checkpoint=str(checkpoint) if checkpoint else None,
                evaluated_weights='EMA' if ema is not None else 'current')
            history.append(row);append(directory/'epochs.jsonl',row)
            if epoch==3:
                projection=projected_run_seconds([h['wall_seconds'] for h in history],100)
                save(directory/'timing_gate.json',dict(first_three_epoch_seconds=[h['wall_seconds'] for h in history],projected_seconds=projection,limit_seconds=2400,passed=projection<=2400))
                if projection>2400:raise RuntimeError(f'Candidate {candidate} seed{seed} timing gate exceeded: {projection:.1f}s')
            if epoch%10==0 or epoch<=3:
                print(f'{candidate} seed{seed} epoch{epoch}: nodeF1={metrics["node_f1"]:.4f} F1={metrics["pixel_f1"]:.4f} clDice={metrics["cldice"]:.4f} CR={metrics["strict_cr"]:.4f} {row["wall_seconds"]:.2f}s',flush=True)
        result=dict(candidate=candidate,seed=seed,epochs=100,selected_checkpoint=selected,
            selected_checkpoint_sha256=sha(selected),best_smoothed_connectivity=best_score,
            selected_epoch=int(Path(selected).stem.split('_')[-1]),final_checkpoint=str(directory/'checkpoint_epoch_100.pt'),
            wall_seconds=time.perf_counter()-run_start,training_class_counts=counts.cpu().tolist(),
            descriptor=dict(name=f'M4_{candidate}',seed=seed,config=cfg,checkpoint=selected,
                checkpoint_sha256=sha(selected),segments=600,raw_tau=.5,origin='m4_training_fix_v2'))
        save(directory/'training.json',result)
        self.event('run_completed',candidate=candidate,seed=seed,wall_seconds=result['wall_seconds'])
        del model,ema,augmented;torch.cuda.empty_cache()
        return result

    def calibrate(self,candidate,seed):
        path=self.out/f'validation/{candidate}_seed{seed}.json'
        if path.exists():return read(path)
        training=read(self.out/f'runs/{candidate}/seed_{seed}/training.json')
        descriptor=training['descriptor']
        probs=self.probabilities(descriptor,self.val_items,f'{candidate}_seed{seed}')
        result=self.search_recipe(probs,self.val_metrics,Recipe(.5,.5),f'{candidate}_seed{seed}')
        # Include size scores for report but never use held-out data for selection.
        result['raw_detailed']=self.val_metrics.score((p>=np.float32(.5) for p in probs),True)
        result['processed_detailed']=self.val_metrics.score((__import__('improvement_core').process(p,result['recipe']) for p in probs),True)
        save(self.out/f'validation/{candidate}_seed{seed}_detailed.json',result)
        return result

    def train_all(self):
        self.preflight();self.prepare()
        for candidate in ['A','B','C','D']:
            if candidate in ['C','D']:self.augment()
            for seed in SEEDS:
                self.train(candidate,seed)
                self.calibrate(candidate,seed)
        self.select_and_lock()

    def select_and_lock(self):
        if (self.out/'locked_configs.json').exists():return
        self.verify_sources()
        incumbent=[c for c in self.old_lock['cases'] if c['model']=='M4']
        old_mean={key:float(np.mean([c['validation'][key] for c in incumbent])) for key in KEYS}
        current=dict(old_mean);winner='historical';records=[]
        for candidate in ['A','B','C','D']:
            results=[read(self.out/f'validation/{candidate}_seed{s}_detailed.json') for s in SEEDS]
            processed=summary([r['metrics'] for r in results]);raw=summary([r['baseline'] for r in results])
            means={k:processed[k]['mean'] for k in KEYS}
            qualifies=means['strict_cr']>=current['strict_cr']+.01-1e-12 and means['cldice']>=current['cldice']-1e-12
            records.append(dict(candidate=candidate,raw=raw,processed=processed,incumbent_before=winner,
                reference_mean=current.copy(),accepted=qualifies))
            if qualifies:winner=candidate;current=means
        save(self.out/'selection.json',dict(winner=winner,historical_mean=old_mean,winning_mean=current,trials=records))
        cases=[]
        for old in incumbent:
            descriptor=old['descriptor'];seed=old['seed']
            for mode,recipe in [('raw',Recipe(descriptor['raw_tau'],descriptor['raw_tau']).json()),('processed',old['recipe'])]:
                cases.append(dict(model='Old M4',seed=seed,mode=mode,descriptor=descriptor,recipe=recipe,
                    validation=old['validation'] if mode=='processed' else None))
        for candidate_name in ['New M4']:
            for seed in SEEDS:
                old=next(c for c in incumbent if c['seed']==seed)
                if winner=='historical':
                    descriptor=old['descriptor'];recipe=old['recipe'];processed=old['validation'];raw=None
                else:
                    descriptor=read(self.out/f'runs/{winner}/seed_{seed}/training.json')['descriptor']
                    result=read(self.out/f'validation/{winner}_seed{seed}_detailed.json')
                    recipe=result['recipe'];processed=result['processed_detailed'];raw=result['raw_detailed']
                for mode,r,m in [('raw',Recipe(descriptor['raw_tau'],descriptor['raw_tau']).json(),raw),('processed',recipe,processed)]:
                    cases.append(dict(model=candidate_name,seed=seed,mode=mode,descriptor=descriptor,recipe=r,validation=m))
        for old in self.old_lock['cases']:
            if old['model']=='M2':
                desc=old['descriptor']
                for mode,recipe in [('raw',Recipe(desc['raw_tau'],desc['raw_tau']).json()),('processed',old['recipe'])]:
                    cases.append(dict(model='M2',seed=old['seed'],mode=mode,descriptor=desc,recipe=recipe,validation=old['validation'] if mode=='processed' else None))
            elif old['model'] in ['DeepCrack raw','DeepCrack processed']:
                cases.append(dict(model='DeepCrack',seed=old['seed'],mode='raw' if old['model'].endswith('raw') else 'processed',
                    descriptor=old['descriptor'],recipe=old['recipe'],validation=old['validation']))
        # Finish all comparator validation and curves before lock and held-out pass.
        cache={}
        for case in cases:
            descriptor=case['descriptor'];key=descriptor['checkpoint_sha256']
            if key not in cache:
                cache[key]=self.probabilities(descriptor,self.val_items,f'lock_{descriptor["name"]}_seed{case["seed"]}')
            probs=cache[key]
            score=self.val_metrics.score((__import__('improvement_core').process(p,case['recipe']) for p in probs),True)
            if case['validation'] is not None:
                for k in KEYS:
                    if abs(score[k]-case['validation'][k])>1e-8:raise ValueError(f'Comparator validation replay mismatch: {case["model"]}, {case["seed"]}, {k}')
            case['validation']=score
        self.tradeoffs(cases,cache)
        complexity={}
        for case in cases:
            key=case['model']
            if key not in complexity:
                complexity[key]=self.complexity(case['descriptor'],self.val_items[0])
                complexity[key]['input']='first DeepCrack validation image, 256-square inference'
        save(self.out/'complexity.json',complexity)
        save(self.out/'locked_configs.json',dict(created=utc(),winner=winner,cases=cases,
            protocol_sha256=sha(self.out/'protocol.json'),tits_manifest_sha256=sha(self.out/'tits_manifest.json'),
            source_hashes=self.protocol['sources'],comparator_mode='three historical training seeds; M2/DeepCrack prior locked recipes retained; no TITS-finetuned checkpoints',
            optional_M2_retraining='skipped; optional and not part of mandatory candidate study',
            held_out_cohort='all 66 annotated TITS pairs',selection=read(self.out/'selection.json'),
            raw_threshold_note='new candidates fixed 0.5; historical M4/M2/DeepCrack retain previous locked fixed thresholds; processed uses locked validation recipe'))
        save(self.out/'locked_configs.sha256.json',dict(sha256=sha(self.out/'locked_configs.json')))
        self.event('locked',winner=winner,cases=len(cases))
        print(f'LOCKED winner={winner}, {len(cases)} raw/processed cases',flush=True)

    def tradeoffs(self,cases,cache):
        rows=[]
        for case in cases:
            if case['mode']!='processed' or case['model']=='Old M4':continue
            p=cache[case['descriptor']['checkpoint_sha256']]
            for low in GRIDS['thresholds']:
                if low>case['recipe']['high']:continue
                recipe=dict(case['recipe'],low=low)
                metrics=self.val_metrics.score((__import__('improvement_core').process(prob,recipe) for prob in p))
                rows.append(dict(model=case['model'],seed=case['seed'],recipe=recipe,metrics=metrics))
        save(self.out/'validation_tradeoffs.json',rows)

    def confirm(self):
        self.check(estimate=2*3600)
        self.verify_sources()
        lock_path=self.out/'locked_configs.json'
        if sha(lock_path)!=read(self.out/'locked_configs.sha256.json')['sha256']:raise ValueError('Lock changed')
        lock=read(lock_path)
        if sha(self.out/'tits_manifest.json')!=lock['tits_manifest_sha256']:raise ValueError('TITS manifest changed')
        save(self.out/'tits_consumed.json',dict(time=utc(),lock_sha256=sha(lock_path),retry_allowed=False))
        self.step_name='held_out_TITS';start=time.perf_counter();manifest=read(self.out/'tits_manifest.json')
        items=[]
        for pair in manifest['pairs']:
            self.check()
            if sha(pair['image'])!=pair['image_sha256'] or sha(pair['mask'])!=pair['mask_sha256']:raise ValueError('TITS data changed since preflight')
            image=cv2.cvtColor(cv2.imread(pair['image'],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
            gt=cv2.imread(pair['mask'],cv2.IMREAD_GRAYSCALE)<127
            items.append(dict(stem=pair['stem'],image=image,gt=gt,split='TITS'))
        metrics=MetricSet([item['gt'] for item in items]);cache={};results=[]
        for index,case in enumerate(lock['cases']):
            self.check();descriptor=case['descriptor'];key=descriptor['checkpoint_sha256']
            if key not in cache:
                # TITS stems contain a slash; inherited probability filenames use numeric IDs.
                cache[key]=self.probabilities(descriptor,items,f'{case["model"].replace(" ","_")}_seed{case["seed"]}','TITS')
            score=metrics.score((__import__('improvement_core').process(p,case['recipe']) for p in cache[key]),True)
            result=dict(model=case['model'],seed=case['seed'],mode=case['mode'],recipe=case['recipe'],held_out=score,
                validation=case['validation'],val_minus_held_out_pp={k:100*(case['validation'][k]-score[k]) for k in KEYS})
            save(self.out/f'held_out/case_{index:02d}.json',result);results.append(result)
            print(f'TITS {case["model"]} {case["mode"]} seed{case["seed"]}: CR={score["strict_cr"]:.4f} clDice={score["cldice"]:.4f} F1={score["pixel_f1"]:.4f}',flush=True)
        after={str(p.relative_to(TITS)):[p.stat().st_size,p.stat().st_mtime_ns] for p in TITS.rglob('*') if p.is_file()}
        if after!=manifest['external_file_metadata']:raise ValueError('External TITS file metadata changed during read-only study')
        save(self.out/'held_out_results.json',dict(results=results,wall_seconds=time.perf_counter()-start,
            external_tits_unchanged=True,lock_sha256=sha(lock_path)))
        self.event('held_out_complete',wall_seconds=time.perf_counter()-start)

    def plot(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plots=self.out/'plots';plots.mkdir(exist_ok=True)
        peaks=[]
        for candidate in ['A','B','C','D']:
            fig,axes=plt.subplots(3,1,figsize=(12,12),constrained_layout=True)
            for ax,seed in zip(axes,SEEDS):
                path=self.out/f'runs/{candidate}/seed_{seed}/epochs.jsonl'
                if not path.exists():continue
                rows=[json.loads(line) for line in path.read_text().splitlines()]
                x=np.array([r['epoch'] for r in rows])
                record=dict(candidate=candidate,seed=seed)
                for key,color in [('node_f1','#7260a9'),('pixel_f1','#2563ab'),('cldice','#15966f'),('strict_cr','#d26920')]:
                    y=np.array([r[key] for r in rows]);ax.plot(x,y,color=color,alpha=.2,linewidth=.7)
                    if len(y)>=5:
                        smooth=np.convolve(y,np.ones(5)/5,mode='valid');ax.plot(x[4:],smooth,color=color,label=key)
                        peak=int(x[4:][int(np.argmax(smooth))]);record[key+'_smoothed_peak_epoch']=peak
                        record[key+'_smoothed_peak']=float(np.max(smooth))
                record['connectivity_peak_after_node_peak']={k:record.get(k+'_smoothed_peak_epoch',0)>record.get('node_f1_smoothed_peak_epoch',0) for k in ['cldice','strict_cr']}
                peaks.append(record);ax.set(title=f'Candidate {candidate}, seed {seed}',xlabel='Epoch',ylabel='Validation metric',ylim=(0,1));ax.grid(alpha=.2);ax.legend(ncol=4)
            fig.suptitle('Fixed raw threshold 0.5; faint per epoch, solid trailing 5-epoch average')
            fig.savefig(plots/f'{candidate}_epochs.png',dpi=140);plt.close(fig)
        if not (self.out/'smoothed_metric_peaks.json').exists():save(self.out/'smoothed_metric_peaks.json',peaks)
        if (self.out/'validation_tradeoffs.json').exists():
            rows=read(self.out/'validation_tradeoffs.json');fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
            for model,color in [('M2','#d26920'),('New M4','#2563ab'),('DeepCrack','#15966f')]:
                for seed in SEEDS:
                    subset=[r for r in rows if r['model']==model and r['seed']==seed]
                    for ax,key in zip(axes,['strict_cr','cldice']):
                        ax.plot([r['metrics']['pixel_f1'] for r in subset],[r['metrics'][key] for r in subset],color=color,alpha=.7,marker='.',label=f'{model} seed{seed}')
                        ax.set(xlabel='Pixel F1',ylabel=key,title=f'DeepCrack validation: {key} vs pixel F1');ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.savefig(plots/'validation_tradeoffs.png',dpi=160);plt.close(fig)

    def report(self,error=None):
        self.plot()
        lines=['# M4 stabilization v2 results','',
            '**Status: '+('partial / failed' if error else 'complete')+'**. Preregistered claims are in `claim.md`. Every selection uses the frozen 210/45/45 DeepCrack seed-42 split; training seeds change randomness only. The consumed 237-image DeepCrack test was never evaluated or used as new evidence.','']
        if error:lines += ['Reason: '+error,'']
        if (self.out/'tits_manifest.json').exists():
            manifest=read(self.out/'tits_manifest.json')
            lines += ['## Dataset provenance','',f'TITS: {manifest["source_images"]} source images, 66 uniquely paired annotated native masks across five sensor groups. Supplied layout: `{manifest["official_layout"]}`. All 66 annotated pairs form the locked cohort; unannotated images are excluded. No TITS-finetuned checkpoints are used. Grayscale images are promoted to three identical channels. Eight masks have larger native dimensions than their source images; inference remains 256×256 and probabilities are projected to native GT size. Black pixels are cracks. No byte duplicate was found against the 300 DeepCrack training-source images. The external folder is read-only; hashes and metadata are recorded.','',
                'DeepCrack source-ID overlap exists between train and validation/test in the historical source audit, despite fixed membership. Validation is therefore not claimed to be source-independent; this study does not silently repair or reshuffle the split.','']
        lines += ['## Fixed-epoch diagnosis','',
            'All mandatory candidate runs use 100 epochs with no early stopping. Native pixel F1, clDice and strict CR are logged each epoch at raw threshold 0.5, plus node F1 on historical native training graphs. Epoch selection uses the trailing five complete epochs of (clDice + strict CR)/2, retaining the endpoint weights, with the earliest epoch winning ties. Checkpoints include every tenth and final epoch and each improving smoothed selection.','',
            'A retains the historical five-epoch warmup and node-F1 ReduceLROnPlateau schedule, but disables early stopping. B combines effective-batch-8 gradient accumulation with cosine LR 1e-3 to 1e-5. C adds an eight-view deterministic augmented graph bank shared across seeds; a random cached view per training source is chosen each epoch. D adds EMA parameters at decay 0.99 after each optimizer step, copying current BatchNorm buffers. Accumulation retains one graph per forward (including per-graph BatchNorm); the final two-graph group is normalized by two.','']
        for candidate in ['A','B','C','D']:
            if (self.out/f'plots/{candidate}_epochs.png').exists():lines += [f'![{candidate} epoch curves](plots/{candidate}_epochs.png)','']
        if (self.out/'smoothed_metric_peaks.json').exists():
            lines += ['| Candidate | Seed | Node F1 peak | Pixel F1 peak | clDice peak | Strict CR peak | Connectivity peaks after node peak |','| --- | --- | --- | --- | --- | --- | --- |']
            for row in read(self.out/'smoothed_metric_peaks.json'):
                lines.append(f'| {row["candidate"]} | {row["seed"]} | {row.get("node_f1_smoothed_peak_epoch")} | {row.get("pixel_f1_smoothed_peak_epoch")} | {row.get("cldice_smoothed_peak_epoch")} | {row.get("strict_cr_smoothed_peak_epoch")} | {row["connectivity_peak_after_node_peak"]} |')
        selection=read(self.out/'selection.json') if (self.out/'selection.json').exists() else None
        lines += ['', '## Validation selection','']
        if selection:
            lines += [f'Winning recipe: **{selection["winner"]}**. Replacement compares three-seed mean processed validation strict CR (minimum +1 pp) and clDice (no decrease) sequentially against the current incumbent. Raw candidate threshold is 0.5; historical controls retain previously locked thresholds. The processing search uses the exact frozen historical hysteresis/min-area/bridging grids and original per-seed clDice guard.','',
                '| Candidate | Mode | Strict CR mean ± sample SD | clDice mean ± sample SD | Pixel F1 mean ± sample SD | Accepted |','| --- | --- | --- | --- | --- | --- |']
            historical=[c['validation'] for c in self.old_lock['cases'] if c['model']=='M4']
            hs=summary(historical)
            fmt=lambda v:f'{v["mean"]:.6f} ± {v["sd"]:.6f}'
            lines.append('| Historical M4 | processed | '+' | '.join(fmt(hs[k]) for k in KEYS)+' | incumbent |')
            for trial in selection['trials']:
                for mode in ['raw','processed']:
                    lines.append(f'| {trial["candidate"]} | {mode} | '+' | '.join(fmt(trial[mode][k]) for k in KEYS)+f' | {trial["accepted"]} |')
        else:lines += ['No complete three-seed candidate selection is available.','']
        results=read(self.out/'held_out_results.json')['results'] if (self.out/'held_out_results.json').exists() else []
        if results:
            for mode in ['raw','processed']:
                lines += ['',f'## Held-out TITS — {mode}','', '| Model | Strict CR | clDice | Pixel F1 | Small [0,50) CR | Medium [50,200) CR | Large [200,inf) CR |','| --- | --- | --- | --- | --- | --- | --- |']
                for model in ['Old M4','New M4','M2','DeepCrack']:
                    subset=[r for r in results if r['mode']==mode and r['model']==model]
                    scores=summary([r['held_out'] for r in subset]);size=[mean_sd([r['held_out']['size_bins'][i]['strict_cr'] for r in subset]) for i in range(3)]
                    lines.append(f'| {model} | '+' | '.join(fmt(scores[k]) for k in KEYS)+' | '+' | '.join(fmt(s) for s in size)+' |')
            lines += ['', '## Validation minus held-out gaps (percentage points)','', '| Model | Mode | Strict CR gap | clDice gap | Pixel F1 gap |','| --- | --- | --- | --- | --- |']
            for mode in ['raw','processed']:
                for model in ['Old M4','New M4','M2','DeepCrack']:
                    subset=[r for r in results if r['mode']==mode and r['model']==model]
                    lines.append(f'| {model} | {mode} | '+' | '.join(fmt(mean_sd([r['val_minus_held_out_pp'][k] for r in subset])) for k in KEYS)+' |')
            lines += ['', 'All SDs use ddof=1 over the three training seeds. These describe training variability on the same held-out cohort, not independent dataset replicates or confidence intervals. Detailed per-seed size-bin counts are retained in `held_out_results.json`.','']
        if (self.out/'validation_tradeoffs.json').exists():lines += ['## Validation trade-offs','', '![Validation trade-offs](plots/validation_tradeoffs.png)','', 'Each curve sweeps the hysteresis low threshold within the frozen grid, retaining its seed-specific locked high threshold, min-area and bridging settings. Curves use DeepCrack validation only.','']
        if (self.out/'complexity.json').exists():
            lines += ['## Model cost','', '| Model | Parameters | Nodes / directed edges | GFLOPs fvcore | GFLOPs analytic including attention |','| --- | --- | --- | --- | --- |']
            for name,cost in read(self.out/'complexity.json').items():lines.append(f'| {name} | {cost["parameters"]} | {cost["nodes"]} / {cost["directed_edges"]} | {cost["gflops_fvcore"]:.9f} | {cost.get("gflops_analytic","CNN: see fvcore")} |')
            lines += ['', 'Both counters use 2×MACs. Analytic graph cost includes attention matrix products; fvcore unsupported-op dictionaries are retained. Input is the first DeepCrack validation image with historical 256×256 preprocessing, not a consumed-test image. Preprocessing and CPU morphology are excluded.','']
        claim1='inconclusive';claim2='inconclusive'
        if selection:
            current,old=selection['winning_mean'],selection['historical_mean']
            claim1='supported' if selection['winner']!='historical' and current['strict_cr']>old['strict_cr'] and current['cldice']>old['cldice'] else 'not supported'
        if results:
            new=[r['held_out'] for r in results if r['model']=='New M4' and r['mode']=='raw']
            dc=[r['held_out'] for r in results if r['model']=='DeepCrack' and r['mode']=='raw']
            claim2='supported' if np.mean([r['strict_cr'] for r in new])>=np.mean([r['strict_cr'] for r in dc]) and np.mean([r['size_bins'][0]['strict_cr'] for r in new])>=np.mean([r['size_bins'][0]['strict_cr'] for r in dc]) else 'not supported'
        lines += ['## Pre-registered verdicts','',f'**Claim 1: {claim1}.** The inference concerns validation clDice and strict CR only, under the frozen split and three-seed rule. A lower training loss alone does not establish the claim. If no candidate replaces the incumbent this is a negative result; no additional recipe is introduced after results.','',
            f'**Claim 2: {claim2}.** This verdict compares RAW new M4 against RAW DeepCrack on mean strict CR and small-crack [0,50) connectivity only, at their locked fixed thresholds. Processed results are separate and do not substitute for raw evidence. Zero-shot TITS transfer may hurt every model, and any other metric/model contrast is exploratory.','',
            '## Negative results, failures and skipped work','', '- Optional M2 retraining was skipped; its three historical seed checkpoints and locked recipes are retained.',
            '- Candidate rejections, including before/after means, are recorded in `selection.json` and the validation table.',
            '- Runtime failures (if any) are recorded in `failures.jsonl`; incomplete runs retain all epoch logs and immutable checkpoints.',
            '- No consumed DeepCrack test run, TITS tuning, source split reshuffling or post-confirmation config changes were performed.','',
            '## Metric definitions','',
            'Strict CR counts all 8-connected GT components, requires at least 50% native GT-area coverage and exactly one connected region in GT∩prediction, and pools preserved/total components across images. Earlier paper_v2 topology excludes components below 20 pixels and accepts coverage by one predicted component without requiring connected GT intersection. Bridging outside GT therefore need not fix strict CR. clDice averages historical per-image skeleton scores with historical empty-mask handling; pixel F1 pools TP/FP/FN. Native size bins are [0,50), [50,200), [200,inf) GT-component area pixels.','',
            '## Reproduction and timing','',
            'Run `REPRODUCE.ps1` from a fresh preregistered output folder. Per-run exact commands are in `runs/<candidate>/seed_<seed>/command.json`. `protocol.json`, per-run configs, `checkpoint_manifest.json`, source hashes and input hashes record the full settings. Runs cannot overwrite completed or interrupted runs. Held-out confirmation has an exclusive consumed marker and cannot retry. `events.jsonl` and `trial_records.json` include wall times and statuses.','']
        (self.out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
        checkpoints=[];trials=[]
        for candidate in ['A','B','C','D']:
            for seed in SEEDS:
                directory=self.out/f'runs/{candidate}/seed_{seed}'
                result=read(directory/'training.json') if (directory/'training.json').exists() else None
                trials.append(dict(candidate=candidate,seed=seed,status='completed' if result else 'incomplete' if directory.exists() else 'not_started',training=result))
                for path in directory.glob('checkpoint_epoch_*.pt'):
                    checkpoints.append(dict(candidate=candidate,seed=seed,path=str(path),sha256=sha(path),selected=bool(result and str(path)==result['selected_checkpoint'])))
        if not (self.out/'checkpoint_manifest.json').exists():save(self.out/'checkpoint_manifest.json',checkpoints)
        if not (self.out/'trial_records.json').exists():save(self.out/'trial_records.json',dict(runs=trials,failures=[json.loads(s) for s in (self.out/'failures.jsonl').read_text().splitlines()] if (self.out/'failures.jsonl').exists() else [],
            total_wall_seconds=time.time()-(self.deadline-9*3600),status='partial' if error else 'complete'))
        print('REPORT:',self.out/'RESULTS.md',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','run','train-one','lock','confirm','report'])
    parser.add_argument('--output',required=True);parser.add_argument('--candidate',choices=['A','B','C','D']);parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args();study=Study(args.output)
    append(study.out/'commands.jsonl',dict(time=utc(),argv=sys.argv,executable=sys.executable,cwd=str(Path.cwd())))
    try:
        if args.stage=='preflight':study.preflight()
        elif args.stage=='run':
            study.train_all();study.confirm();study.report()
        elif args.stage=='train-one':
            study.preflight();study.prepare()
            if args.candidate in ['C','D']:study.augment()
            study.train(args.candidate,args.seed);study.calibrate(args.candidate,args.seed)
        elif args.stage=='lock':study.preflight();study.prepare();study.select_and_lock()
        elif args.stage=='confirm':study.confirm();study.report()
        else:study.report()
    except Exception as exc:
        append(study.out/'failures.jsonl',dict(time=utc(),stage=args.stage,candidate=args.candidate,seed=args.seed,error=str(exc),traceback=traceback.format_exc()))
        study.event('failure',error=str(exc));study.report(str(exc));raise


if __name__=='__main__':
    main()
