"""Budgeted, validation-first M2/M4 experiment; no legacy output mutations.

Run from pipeline-crack-gnn with Python 3.11. `run --output <new-directory>`
executes all stages; explicit stage commands permit inspected resumptions.
Only final-test may decode test images/labels; it requires an immutable lock and
an exclusive started marker. Protocol may hash same-name image files solely to
disambiguate split-qualified identities. Failed test evaluations cannot retry.
"""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'DeepSegmentor'))

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.io import imread
from skimage.morphology import skeletonize
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from improvement_core import (BINS, GRIDS, OVERLAPS, MetricSet, Recipe, process,
                              qualifies, rank, skeleton_recall_loss, projected_run_seconds)
from src.models.hybrid_model import build_model
from src.train import load_config, FocalLoss
from src.utils.dataset import get_splits
from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data
from src.preprocessing.build_dataset import assign_node_labels
from models.deepcrack_networks import DeepCrackNet, BinaryFocalLoss
import crack_metrics as cm

PITCH = BASE / 'results/paper_v2/pitch_rerun_20261014_01'
DATA = ROOT / 'DeepCrack'
STEPS = {'protocol': 3600, 'search': 7200, 'train': 14400, 'final-test': 7200, 'report': 3600}
SPECS = {
    'M2': ('configs/shallow_gnn.yaml', 'results/M2_shallow_gnn/best_model.pt', .70),
    'M4': ('configs/step2_loss/hybrid_full_focal.yaml', 'results/step2_loss/M4_hybrid_full/best_model.pt', .66),
    'DeepCrack': (None, 'results/multiseed/deepcrack/seed_42/best_model.pt', .36),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot():
    return {str(p.relative_to(BASE)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in (BASE / 'results').rglob('*') if p.is_file()
            and not any(part.startswith('improve_m2_m4_') for part in p.parts)}


class BudgetExceeded(RuntimeError):
    pass


class Experiment:
    def __init__(self, output):
        self.out = Path(output).resolve()
        if not self.out.is_relative_to(BASE / 'results/paper_v2'):
            raise ValueError('Output must be below results/paper_v2')
        if not self.out.name.startswith('improve_m2_m4_'):
            raise ValueError('Output directory name must start improve_m2_m4_')
        self.step_name, self.deadline = None, None
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_cache = {}
        self.items_cache = {}

    def event(self, kind, **values):
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(time=utc(), kind=kind, stage=self.step_name, **values), allow_nan=False)+'\n')

    @contextmanager
    def stage(self, name):
        marker = self.out / f'{name}_started.json'
        if marker.exists():
            started = read(marker)['unix_time']
        else:
            started = time.time()
            save(marker, dict(unix_time=started, time=utc(), budget_seconds=STEPS[name]))
        self.step_name = name
        self.deadline = started + STEPS[name]
        self.event('stage_start', budget_seconds=STEPS[name])
        self.check()
        try:
            yield
            self.check()
        finally:
            elapsed = time.time()-started
            self.event('stage_end', wall_seconds=elapsed)
            print(f'STAGE {name}: {elapsed:.1f}s / {STEPS[name]}s', flush=True)
            self.step_name, self.deadline = None, None

    def check(self, estimate=0):
        if self.deadline is not None and time.time()+estimate >= self.deadline:
            raise BudgetExceeded(f'{self.step_name}: next operation estimate {estimate:.1f}s exceeds remaining {self.deadline-time.time():.1f}s')

    @contextmanager
    def operation(self, name):
        self.check()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.event('operation', name=name, wall_seconds=time.perf_counter()-start)
        self.check()

    @property
    def protocol(self):
        return read(self.out / 'protocol.json')

    def paths(self, split, stem, labels=False):
        if split == 'test' and self.step_name != 'final-test':
            raise RuntimeError('TEST ACCESS DENIED outside locked final-test stage')
        directory = DATA / f'{split}_{"lab" if labels else "img"}'
        for extension in ('.jpg', '.png', '.jpeg', '.bmp'):
            path = directory / f'{stem}{extension}'
            if path.exists():
                return path
        raise FileNotFoundError(directory / stem)

    def items(self, stems, split='train'):
        key = split, tuple(stems)
        if split == 'test' and self.step_name != 'final-test':
            raise RuntimeError('TEST ACCESS DENIED')
        if key not in self.items_cache:
            rows = []
            for stem in stems:
                self.check()
                image = cv2.cvtColor(cv2.imread(str(self.paths(split, stem))), cv2.COLOR_BGR2RGB)
                gt = cv2.imread(str(self.paths(split, stem, True)), cv2.IMREAD_GRAYSCALE) > 127
                if image.shape[:2] != gt.shape:
                    raise ValueError(f'Shape mismatch: {stem}')
                rows.append(dict(stem=stem, image=image, gt=gt, split=split))
            self.items_cache[key] = rows
        return self.items_cache[key]

    def graph_input(self, item, segments=300):
        if item['split'] == 'test' and self.step_name != 'final-test':
            raise RuntimeError('TEST ACCESS DENIED')
        key = item['split'], item['stem'], segments
        if key not in self.input_cache:
            small = cv2.resize(item['image'], (256, 256), interpolation=cv2.INTER_LINEAR)
            seg = compute_superpixels(small, segments, 10., 1., enforce_connectivity=False)
            features, positions = extract_node_features(small, seg)
            ei, ew = symmetrize_graph(*build_pirm_graph(features, 8, .1))
            graph = build_pyg_data(features, ei, ew, positions)
            labels = np.unique(seg)
            mapped = cv2.resize(seg, (item['gt'].shape[1], item['gt'].shape[0]), interpolation=cv2.INTER_NEAREST)
            inverse = np.searchsorted(labels, mapped)
            if not np.array_equal(labels[inverse], mapped):
                raise ValueError('Superpixel projection lost node identity')
            self.input_cache[key] = graph, inverse
        return self.input_cache[key]

    def descriptor(self, name):
        cfg, checkpoint, tau = SPECS[name]
        return dict(name=name, seed=42, config=load_config(BASE/cfg) if cfg else None,
                    checkpoint=str(BASE/checkpoint), checkpoint_sha256=sha(BASE/checkpoint),
                    segments=300, raw_tau=tau, origin='historical_pitch')

    def model(self, descriptor):
        model = (build_model(descriptor['config']) if descriptor['config'] else
                 DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch'))
        if sha(descriptor['checkpoint']) != descriptor['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after configuration lock')
        model.load_state_dict(torch.load(descriptor['checkpoint'], map_location='cpu', weights_only=True))
        return model.to(self.dev).eval()

    def probabilities(self, descriptor, items, tag, split='val'):
        directory = self.out / 'probabilities' / split / tag
        identity = dict(checkpoint_sha256=descriptor['checkpoint_sha256'], segments=descriptor['segments'],
                        stems=[i['stem'] for i in items], split=split)
        if (directory/'manifest.json').exists():
            if read(directory/'manifest.json') != identity:
                raise ValueError('Probability cache identity mismatch')
            return [np.load(directory/f'{i:03d}.npz')['probability'] for i in range(len(items))]
        directory.mkdir(parents=True, exist_ok=False)
        model = self.model(descriptor)
        probabilities = []
        with self.operation(f'predict_{split}_{tag}'), torch.no_grad():
            for index, item in enumerate(items):
                self.check()
                if descriptor['config']:
                    graph, inverse = self.graph_input(item, descriptor['segments'])
                    graph = graph.to(self.dev)
                    logits = (model(graph.x, graph.edge_index, graph.pos)
                              if descriptor['config']['model']['type'] == 'hybrid'
                              else model(graph.x, graph.edge_index))
                    probability = torch.softmax(logits, 1)[:, 1].cpu().numpy()[inverse]
                else:
                    image = cv2.resize(item['image'], (256, 256), interpolation=cv2.INTER_LINEAR)
                    tensor = ((torch.from_numpy(image).permute(2, 0, 1).float()/255-.5)/.5)[None].to(self.dev)
                    fused = model(tensor)[-1]
                    fused = F.interpolate(fused, size=item['gt'].shape, mode='bilinear', align_corners=True)
                    probability = torch.sigmoid(fused)[0, 0].cpu().numpy()
                probability = probability.astype(np.float32)
                with (directory/f'{index:03d}.npz').open('xb') as f:
                    np.savez_compressed(f, probability=probability, stem=item['stem'])
                probabilities.append(probability)
        save(directory/'manifest.json', identity)
        del model
        torch.cuda.empty_cache()
        return probabilities

    def freeze(self, cache_policy):
        if (self.out/'protocol.json').exists():
            raise FileExistsError('Protocol already frozen')
        if self.dev.type != 'cuda':
            raise RuntimeError('CUDA required; do not silently change the training budget')
        with self.stage('protocol'):
            save(self.out/'existing_artifacts.json', snapshot())
            files = sorted((BASE/'data/graphs/train').glob('*.pt'))
            if len(files) != 300:
                raise ValueError('Expected 300 training-source graphs')
            train, val, unused = get_splits(files, .7, .15, 42)
            stems = [p.stem for p in files]
            split = dict(train=[stems[i] for i in train], val=[stems[i] for i in val],
                         unused=[stems[i] for i in unused], test=sorted(p.stem for p in (BASE/'data/graphs/test').glob('*.pt')))
            if [len(split[k]) for k in ('train', 'val', 'unused', 'test')] != [210, 45, 45, 237]:
                raise ValueError('Split counts mismatch')
            groups = [set(split[k]) for k in ('train', 'val', 'unused')]
            if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
                raise ValueError('Overlapping split stems')
            # Upstream train/test directories reuse ten names for different images.
            # Filename equality across directories is not sample identity. Hash only;
            # no test image decoding, label loading, prediction, or metric computation.
            collisions=[]
            for stem in sorted(set(stems)&set(split['test'])):
                train_image=self.paths('train',stem)
                candidates=[DATA/f'test_img/{stem}{ext}' for ext in ('.jpg','.png','.jpeg','.bmp')]
                test_image=next((p for p in candidates if p.exists()),None)
                if test_image is None: raise FileNotFoundError(f'test image identity: {stem}')
                a,b=sha(train_image),sha(test_image)
                if a==b: raise ValueError(f'Identical train-source/test image bytes: {stem}')
                collisions.append(dict(stem=stem,train_image_sha256=a,test_image_sha256=b,identical_bytes=False))
            save(self.out/'split_identity_audit.json',dict(identity_key='(raw_directory_split, stem)',collisions=collisions,
                                                          decoded_test_images=0,loaded_test_labels=0))
            audit = {}
            for name in SPECS:
                source = read(PITCH/f'historical_{name}.json')
                actual_hash = sha(ROOT/source['checkpoint'])
                if actual_hash != source['checkpoint_sha256']:
                    raise ValueError(f'Historical checkpoint hash mismatch: {name}')
                accuracy = source['accuracy']
                counts = accuracy['overlap_sweep'][2]
                if counts['total'] != 992 or counts['preserved'] != {'M2':419, 'M4':297, 'DeepCrack':334}[name]:
                    raise ValueError('Historical component counts changed')
                if counts['micro'] != counts['preserved']/counts['total']:
                    raise ValueError('Historical summary inconsistent with counts')
                if [b['total'] for b in accuracy['size_buckets']] != [167,274,551]:
                    raise ValueError('Historical size-bin counts changed')
                audit[name] = dict(checkpoint_sha256=actual_hash, cldice=accuracy['cldice'],
                                   strict_cr=counts['micro'], pixel_f1=accuracy['pixel_f1'],
                                   fresh_test_reproduction='deferred_until_final_lock')
            save(self.out/'historical_audit.json', audit)
            provenance = []
            for stem in [stems[0], stems[150], stems[-1]]:
                self.check()
                graph = torch.load(BASE/f'data/graphs/train/{stem}.pt', weights_only=False)
                seg = np.load(BASE/f'data/superpixels/train/{stem}.npy')
                image = imread(self.paths('train', stem))
                mask = imread(self.paths('train', stem, True))
                features, positions = extract_node_features(image, seg)
                ei, ew = symmetrize_graph(*build_pirm_graph(features, 8, .1))
                if not (np.allclose(features, graph.x.numpy(), atol=1e-6) and
                        np.allclose(positions, graph.pos.numpy(), atol=1e-6) and
                        np.array_equal(ei, graph.edge_index.numpy()) and
                        np.array_equal(assign_node_labels(seg, mask, .3), graph.y.numpy())):
                    raise ValueError(f'Native cache cannot be reproduced from its superpixels: {stem}')
                flags = [flag for flag in (False, True) if np.array_equal(
                    compute_superpixels(image, 300, 10., 1., enforce_connectivity=flag), seg)]
                if not flags:
                    raise ValueError(f'Cached SLIC map provenance unresolved: {stem}')
                provenance.append(dict(stem=stem, reproduced_connectivity_flags=flags))
            environment = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                               gpu=torch.cuda.get_device_name(0), platform=platform.platform(),
                               tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                               tf32_cudnn=torch.backends.cudnn.allow_tf32,
                               git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
                               git_status=subprocess.check_output(['git','status','--short'], cwd=ROOT, text=True))
            files_to_hash = [Path(__file__), Path(__file__).with_name('improvement_core.py'), ROOT/'crack_metrics.py',
                             BASE/'src/train.py', BASE/'src/preprocessing/build_dataset.py',
                             ROOT/'DeepSegmentor/models/deepcrack_networks.py',
                             BASE/'src/models/hybrid_model.py',BASE/'src/preprocessing/features.py',
                             BASE/'src/preprocessing/superpixel.py',BASE/'src/preprocessing/pirm_graph.py']
            protocol = dict(created=utc(), seeds=[42,1,2], split_seed=42, split=split,
                            budgets_seconds=STEPS, grids=GRIDS, min_absolute_cr_gain=.01,
                            size_bins=BINS, metrics=['strict_cr','cldice','pixel_f1'],
                            baseline='DeepSegmentor DeepCrackNet BN ngf64; raw + processed operating points',
                            training_cache_policy=cache_policy, cached_slic_provenance=provenance,
                            split_identity_audit=collisions,
                            existing_source_overlap_audit=read(BASE/'results/paper_v2/source_overlap.json'),
                            comparator_calibration='Seed-42 recipe fixed across seeds; M4 and blends recalibrated per seed',
                            environment=environment, sources={str(p.relative_to(ROOT)):sha(p) for p in files_to_hash},
                            native_training_graph_hashes={p.stem:sha(p) for p in files},
                            native_training_map_hashes={p.stem:sha(BASE/f'data/superpixels/train/{p.stem}.npy') for p in files})
            save(self.out/'protocol.json', protocol)
            self.event('protocol_frozen', split_sizes={k:len(v) for k,v in split.items()})

    def verify_sources(self):
        for name, digest in self.protocol['sources'].items():
            if sha(ROOT/name) != digest:
                raise ValueError(f'Experiment source changed since freeze: {name}')

    def search_recipe(self, probabilities, metrics, raw, tag, guard=None):
        """Independent and cumulative ablations; only +1pp transitions replace incumbent."""
        baseline = metrics.score(process(p, raw) for p in probabilities)
        guard = baseline['cldice'] if guard is None else guard
        trials_path = self.out/'validation_trials'/f'{tag}.jsonl'
        if trials_path.exists():
            raise FileExistsError(f'Trials already exist: {tag}')
        trials_path.parent.mkdir(exist_ok=True)
        seen = {}
        def evaluate(recipe, family):
            key = tuple(recipe.json().values())
            if key in seen:
                return seen[key]
            self.check()
            started = time.perf_counter()
            score = metrics.score(process(p, recipe) for p in probabilities)
            elapsed = time.perf_counter()-started
            self.check(estimate=elapsed)
            with trials_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(family=family, recipe=recipe.json(), metrics=score,
                                        wall_seconds=elapsed, guard=guard))+'\n')
            seen[key] = score
            return score
        def choose(incumbent, incumbent_score, recipes, family):
            choices = []
            for recipe in recipes:
                score = evaluate(recipe, family)
                if qualifies(score, incumbent_score, guard):
                    operations = int(recipe.low != recipe.high)+int(recipe.min_area>0)+int(recipe.max_gap>0)
                    choices.append((rank(score), -operations, recipe, score))
            if not choices:
                return incumbent, incumbent_score
            best = max(choices, key=lambda row:(row[0],row[1]))
            return best[2], best[3]
        thresholds = [replace(raw, low=lo, high=hi) for hi in GRIDS['thresholds']
                      for lo in GRIDS['thresholds'] if lo <= hi]
        with self.operation(f'validation_search_{tag}'):
            hysteresis, hs = choose(raw, baseline, thresholds, 'hysteresis')
            independent = {}
            for family, recipes in [('component_removal', [replace(raw,min_area=a) for a in GRIDS['min_areas']]),
                                    ('gap_bridging', [replace(raw,max_gap=g) for g in GRIDS['max_gaps']])]:
                recipe, score = choose(raw, baseline, recipes, 'independent_'+family)
                independent[family] = dict(recipe=recipe.json(), before=baseline, after=score,
                                           improved=recipe != raw)
            area, ars = choose(hysteresis, hs, [replace(hysteresis,min_area=a) for a in GRIDS['min_areas']], 'cumulative_removal')
            gap, gs = choose(area, ars, [replace(area,max_gap=g) for g in GRIDS['max_gaps']], 'cumulative_bridging')
            # Also retain a superior independent recipe rather than losing it to the greedy chain.
            alternatives = [(gap, gs)] + [(Recipe(**v['recipe']), v['after']) for v in independent.values()]
            qualifying = [(r,s) for r,s in alternatives if qualifies(s,baseline,guard)]
            selected, selected_score = max(qualifying, key=lambda rs:rank(rs[1])) if qualifying else (raw, baseline)
        result = dict(tag=tag, guard=guard, baseline=baseline, recipe=selected.json(), metrics=selected_score,
                      independent=independent, cumulative=[dict(change='hysteresis',recipe=hysteresis.json(),metrics=hs),
                      dict(change='component_removal',recipe=area.json(),metrics=ars),
                      dict(change='gap_bridging',recipe=gap.json(),metrics=gs)], improved=selected != raw,
                      trials=str(trials_path), unique_candidates=len(seen))
        save(self.out/'validation'/f'{tag}.json', result)
        print(f'VAL {tag}: CR {baseline["strict_cr"]:.4f} -> {selected_score["strict_cr"]:.4f}; clDice {baseline["cldice"]:.4f} -> {selected_score["cldice"]:.4f}; {selected.json()}', flush=True)
        return result

    def search_blend(self, p2, p4, metrics, m4_result, tag, guard):
        recipe = Recipe(**m4_result['recipe'])
        incumbent = dict(alpha=0., recipe=recipe.json(), metrics=m4_result['metrics'])
        def selection_key(candidate):
            settings = candidate['recipe']
            operations = (int(settings['low'] != settings['high'])
                          + int(settings['min_area'] > 0) + int(settings['max_gap'] > 0))
            return rank(candidate['metrics']), -operations
        rows = []
        with self.operation(f'blend_search_{tag}'):
            for alpha in GRIDS['alphas']:
                self.check()
                probabilities = [np.float32(alpha)*a+np.float32(1-alpha)*b for a,b in zip(p2,p4)]
                # Same frozen threshold grid; retain the M4 processing stages for the blend.
                best = None
                for lo,hi in ((lo,hi) for hi in GRIDS['thresholds'] for lo in GRIDS['thresholds'] if lo<=hi):
                    candidate = replace(recipe, low=lo, high=hi)
                    start = time.perf_counter()
                    score = metrics.score(process(p,candidate) for p in probabilities)
                    rows.append(dict(alpha=alpha,recipe=candidate.json(),metrics=score,wall_seconds=time.perf_counter()-start))
                    self.check()
                    choice = dict(alpha=alpha,recipe=candidate.json(),metrics=score)
                    if qualifies(score,m4_result['metrics'],guard) and (best is None or selection_key(choice)>selection_key(best)):
                        best = choice
                # Include inherited hysteresis, so threshold compression cannot discard it.
                score = metrics.score(process(p,recipe) for p in probabilities)
                rows.append(dict(alpha=alpha,recipe=recipe.json(),metrics=score))
                choice = dict(alpha=alpha,recipe=recipe.json(),metrics=score)
                if qualifies(score,m4_result['metrics'],guard) and (best is None or selection_key(choice)>selection_key(best)):
                    best = choice
                if best is not None and selection_key(best)>selection_key(incumbent):
                    incumbent = best
            if incumbent['alpha']!=0:
                mixed=[np.float32(incumbent['alpha'])*a+np.float32(1-incumbent['alpha'])*b for a,b in zip(p2,p4)]
                # Refine the selected blend's area and gap settings with the same frozen grids.
                for family,field in [('min_areas','min_area'),('max_gaps','max_gap')]:
                    baseline=dict(incumbent)
                    for value in GRIDS[family]:
                        candidate=replace(Recipe(**baseline['recipe']),**{field:value})
                        score=metrics.score(process(p,candidate) for p in mixed)
                        rows.append(dict(alpha=incumbent['alpha'],recipe=candidate.json(),metrics=score,family=family))
                        self.check()
                        if qualifies(score,baseline['metrics'],guard) and rank(score)>rank(incumbent['metrics']):
                            incumbent=dict(alpha=incumbent['alpha'],recipe=candidate.json(),metrics=score)
            save(self.out/'validation_trials'/f'{tag}.json', rows)
        result = dict(tag=tag,baseline=m4_result['metrics'],guard=guard,**incumbent,
                      improved=incumbent['alpha']!=0, trials=str(self.out/'validation_trials'/f'{tag}.json'))
        save(self.out/'validation'/f'{tag}.json', result)
        print(f'VAL {tag}: alpha={result["alpha"]}; CR={result["metrics"]["strict_cr"]:.4f}',flush=True)
        return result

    def search(self):
        self.verify_sources()
        with self.stage('search'):
            items = self.items(self.protocol['split']['val'])
            metrics = MetricSet([i['gt'] for i in items])
            probabilities, results = {}, {}
            for name in SPECS:
                probabilities[name] = self.probabilities(self.descriptor(name), items, f'{name}_original_seed42')
                tau = SPECS[name][2]
                results[name] = self.search_recipe(probabilities[name], metrics, Recipe(tau,tau), f'{name}_original_seed42')
                # Exact wrapper equivalence, tested only on validation images.
                model = self.model(self.descriptor(name))
                wrapper = (cm.DeepCrackSegmentorWrapper(model,tau,str(self.dev)) if name=='DeepCrack'
                           else cm.GNNSegmentorWrapper(model,self.descriptor(name)['config']['model']['type'],tau,str(self.dev)))
                for i in (0,1):
                    if not np.array_equal(wrapper.predict_mask(items[i]['image']).astype(bool), probabilities[name][i]>=np.float32(tau)):
                        raise ValueError(f'Historical probability projection mismatch: {name}')
                del wrapper,model
            results['blend'] = self.search_blend(probabilities['M2'],probabilities['M4'],metrics,results['M4'],
                                                 'blend_original_seed42',results['M4']['baseline']['cldice'])
            save(self.out/'original_processing_locked.json',dict(protocol_sha256=sha(self.out/'protocol.json'),grids=GRIDS,models=results))

    def native_graphs(self, stems, segments):
        graphs = []
        with self.operation(f'native_graphs_{segments}_{len(stems)}'):
            for stem in stems:
                self.check()
                if segments == 300:
                    path = BASE/f'data/graphs/train/{stem}.pt'
                    segpath = BASE/f'data/superpixels/train/{stem}.npy'
                    if (sha(path) != self.protocol['native_training_graph_hashes'][stem]
                        or sha(segpath) != self.protocol['native_training_map_hashes'][stem]):
                        raise ValueError(f'Training input changed: {stem}')
                    graph = torch.load(path, weights_only=False)
                    seg = np.load(segpath)
                else:
                    path = self.out/f'graphs/segments_{segments}/{stem}.pt'
                    segpath = self.out/f'superpixels/segments_{segments}/{stem}.npy'
                    if path.exists():
                        graph = torch.load(path, weights_only=False)
                        seg = np.load(segpath)
                    else:
                        image = imread(self.paths('train',stem))
                        mask = imread(self.paths('train',stem,True))
                        seg = compute_superpixels(image,segments,10.,1.,enforce_connectivity=False)
                        features,positions = extract_node_features(image,seg)
                        ei,ew = symmetrize_graph(*build_pirm_graph(features,8,.1))
                        labels = assign_node_labels(seg,mask,.3)
                        graph = build_pyg_data(features,ei,ew,positions,labels)
                        path.parent.mkdir(parents=True,exist_ok=True)
                        segpath.parent.mkdir(parents=True,exist_ok=True)
                        with path.open('xb') as f:
                            torch.save(graph,f)
                        with segpath.open('xb') as f:
                            np.save(f,seg)
                gt = cv2.imread(str(self.paths('train',stem,True)),cv2.IMREAD_GRAYSCALE)>127
                labels,inverse = np.unique(seg,return_inverse=True)
                if len(labels) != graph.num_nodes or seg.shape != gt.shape:
                    raise ValueError('Native training superpixel/node mismatch')
                skel = skeletonize(gt)
                counts = np.bincount(inverse.ravel(),weights=skel.ravel(),minlength=len(labels)).astype(np.float32)
                graph.skeleton_counts = torch.from_numpy(counts)
                graphs.append(graph)
        return graphs

    def run_config(self, cell, seed, loss, segments, weight=0., model_name='M4'):
        cfg = load_config(BASE/SPECS[model_name][0]) if model_name != 'DeepCrack' else None
        if cfg:
            cfg['training']['seed'] = seed
            cfg['training']['loss'] = loss
            cfg['superpixels']['n_segments'] = segments
            # 300 uses immutable legacy caches; new 600-segment graphs always use False.
            cfg['superpixels']['enforce_connectivity'] = segments == 300
        config = dict(cell=cell,seed=seed,loss=loss,skeleton_weight=weight,segments=segments,
                      model_name=model_name,config=cfg,protocol_sha256=sha(self.out/'protocol.json'),
                      split_seed=42,raw_tau=SPECS[model_name][2])
        config_path = self.out/f'runs/{cell}/seed_{seed}/config.json'
        save(config_path,config)
        return config_path

    def train_one(self, config_path):
        cfg = read(config_path)
        if cfg['protocol_sha256'] != sha(self.out/'protocol.json'):
            raise ValueError('Run configuration does not match protocol')
        self.verify_sources()
        directory = Path(config_path).parent
        if (directory/'training.json').exists():
            raise FileExistsError('Run already completed')
        seed = cfg['seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        command = subprocess.list2cmdline([sys.executable,str(Path(__file__).resolve()),'train-one',
                                          '--output',str(self.out),'--config',str(Path(config_path).resolve())])
        save(directory/'command.json',dict(command=command,cwd=str(BASE)))
        start = time.perf_counter()
        with self.operation(f'train_{cfg["cell"]}_seed{seed}'):
            result = (self.train_dc(cfg,directory) if cfg['model_name']=='DeepCrack'
                      else self.train_graph(cfg,directory))
        result['wall_seconds_including_setup'] = time.perf_counter()-start
        result['command'] = command
        save(directory/'training.json',result)
        descriptor = dict(name=cfg['cell'],seed=seed,config=cfg['config'],checkpoint=result['best_checkpoint'],
                          checkpoint_sha256=sha(result['best_checkpoint']),segments=cfg['segments'],
                          raw_tau=cfg['raw_tau'],origin='new_training',training=str(directory/'training.json'))
        save(directory/'descriptor.json',descriptor)
        return descriptor,result

    def save_epoch(self, model, directory, epoch):
        path = directory/f'checkpoint_epoch_{epoch:03d}.pt'
        with path.open('xb') as f:
            torch.save(model.state_dict(),f)
        return str(path)

    def epoch_event(self, directory, record):
        with (directory/'epochs.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        self.event('epoch',run=str(directory.relative_to(self.out)),**record)
        print(f'{directory.parent.name} seed {directory.name}: epoch {record["epoch"]}, val F1={record["val_f1"]:.4f}, {record["wall_seconds"]:.2f}s',flush=True)

    def train_graph(self, run, directory):
        train = self.native_graphs(self.protocol['split']['train'],run['segments'])
        val = self.native_graphs(self.protocol['split']['val'],run['segments'])
        cfg = run['config']; settings = cfg['training']
        model = build_model(cfg).to(self.dev)
        labels = torch.cat([g.y for g in train])
        counts = torch.bincount(labels,minlength=2).float()
        if (counts == 0).any():
            raise ValueError('Missing training class')
        weights = (counts.sum()/(2*counts)).to(self.dev)
        if run['loss']=='focal':
            alpha = counts.flip(0)/counts.sum()
            criterion = FocalLoss(alpha.to(self.dev),gamma=2.)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
        train_loader = DataLoader(train,batch_size=1,shuffle=True)
        val_loader = DataLoader(val,batch_size=1,shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
        warm = settings.get('warmup_epochs',0)
        warmup = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:(epoch+1)/warm if warm and epoch<warm else 1.) if warm else None
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=5)
        best, patience, checkpoint = None,0,None
        times = []; projections = []; start = time.perf_counter()
        for epoch in range(settings['epochs']):
            self.check()
            if len(times)>=3:
                estimate = max(times[-3:])*(settings['epochs']-epoch)
                self.check(estimate=estimate+60)
            torch.cuda.synchronize(); epstart=time.perf_counter()
            model.train(); loss_sum=0.; n_nodes=0
            for graph in train_loader:
                self.check(); graph=graph.to(self.dev); optimizer.zero_grad()
                logits=(model(graph.x,graph.edge_index,graph.pos) if cfg['model']['type']=='hybrid'
                        else model(graph.x,graph.edge_index))
                loss=criterion(logits,graph.y.long())
                if run['skeleton_weight']:
                    loss=loss+run['skeleton_weight']*skeleton_recall_loss(logits,graph.skeleton_counts)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                if 'grad_clip_norm' in settings:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip_norm'])
                optimizer.step(); loss_sum+=float(loss.item())*graph.num_nodes; n_nodes+=graph.num_nodes
            model.eval(); predictions=[]; labels=[]
            with torch.no_grad():
                for graph in val_loader:
                    self.check(); graph=graph.to(self.dev)
                    logits=(model(graph.x,graph.edge_index,graph.pos) if cfg['model']['type']=='hybrid'
                            else model(graph.x,graph.edge_index))
                    predictions.append(logits.argmax(1).cpu().numpy()); labels.append(graph.y.cpu().numpy())
            score=float(f1_score(np.concatenate(labels),np.concatenate(predictions),zero_division=0))
            if warmup and epoch<warm:
                warmup.step()
            else:
                plateau.step(score)
            # Match existing EarlyStopping: equal scores save and reset patience.
            if best is None or score>=best:
                best=score; patience=0; checkpoint=self.save_epoch(model,directory,epoch+1)
            else:
                patience+=1
            torch.cuda.synchronize(); elapsed=time.perf_counter()-epstart; times.append(elapsed)
            self.epoch_event(directory,dict(epoch=epoch+1,train_loss=loss_sum/n_nodes,val_f1=score,
                                           wall_seconds=elapsed,lr=float(optimizer.param_groups[0]['lr'])))
            if len(times)==3:
                projected=projected_run_seconds(times,settings['epochs'])
                projections.append(projected)
                self.event('timing_gate',run=str(directory),projected_seconds=projected,
                           skeleton_weights=[.3] if projected>1800 else [.1,.3,.5])
            if patience>=settings['early_stopping_patience']:
                break
        result=dict(best_checkpoint=checkpoint,best_val_f1=best,epochs=len(times),epoch_seconds=times,
                    projected_full_run_seconds=projections[0] if projections else sum(times),
                    train_seconds=time.perf_counter()-start,training_class_counts=counts.tolist(),
                    class_weights=weights.cpu().tolist(),loss=run['loss'],skeleton_weight=run['skeleton_weight'])
        del model,train_loader,val_loader,train,val
        torch.cuda.empty_cache()
        return result

    def train_dc(self, run, directory):
        train_items=self.items(self.protocol['split']['train']); val_items=self.items(self.protocol['split']['val'])
        def arrays(items):
            images=np.stack([cv2.resize(i['image'],(256,256),interpolation=cv2.INTER_LINEAR) for i in items])
            masks=np.stack([cv2.resize(i['gt'].astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST) for i in items])
            return images,masks
        train_images,train_masks=arrays(train_items); val_images,val_masks=arrays(val_items)
        model=DeepCrackNet(in_nc=3,num_classes=1,ngf=64,norm='batch').to(self.dev)
        criterion=BinaryFocalLoss(alpha=1,gamma=2,logits=True).to(self.dev)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=2e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
        times=[]; best=-1.; patience=0; checkpoint=None; start=time.perf_counter()
        for epoch in range(30):
            self.check()
            if len(times)>=3:
                self.check(estimate=max(times[-3:])*(30-epoch)+60)
            torch.cuda.synchronize(); epstart=time.perf_counter(); model.train(); loss_sum=0.
            order=torch.randperm(len(train_images)).tolist()
            for offset in range(0,len(order),4):
                self.check(); ids=order[offset:offset+4]; images=[]; masks=[]
                for i in ids:
                    image=train_images[i]; mask=train_masks[i]
                    if np.random.rand()>.5: image=np.fliplr(image); mask=np.fliplr(mask)
                    if np.random.rand()>.5: image=np.flipud(image); mask=np.flipud(mask)
                    images.append(image.copy()); masks.append(mask.copy())
                tensor=((torch.from_numpy(np.stack(images)).permute(0,3,1,2).float()/255-.5)/.5).to(self.dev)
                target=torch.from_numpy(np.stack(masks)[:,None].astype(np.float32)).to(self.dev)
                optimizer.zero_grad(); outputs=model(tensor)
                loss=sum(w*criterion(o,target) for w,o in zip([.5,.75,1.,.75,.5,1.],outputs))
                if not torch.isfinite(loss): raise ValueError('Nonfinite DeepCrack loss')
                loss.backward(); optimizer.step(); loss_sum+=float(loss.item())*len(ids)
            model.eval(); all_probabilities=[]
            with torch.no_grad():
                for image in val_images:
                    tensor=((torch.from_numpy(image).permute(2,0,1).float()/255-.5)/.5)[None].to(self.dev)
                    all_probabilities.append(torch.sigmoid(model(tensor)[-1])[0,0].cpu().numpy())
            flat=np.stack(all_probabilities).ravel(); gt=val_masks.ravel().astype(bool)
            score=max(float(2*((flat>=tau)&gt).sum()/max(int((flat>=tau).sum()+gt.sum()),1))
                      for tau in np.arange(.2,.85,.05))
            scheduler.step(score)
            if score>best:
                best=score; patience=0; checkpoint=self.save_epoch(model,directory,epoch+1)
            else: patience+=1
            torch.cuda.synchronize(); elapsed=time.perf_counter()-epstart; times.append(elapsed)
            self.epoch_event(directory,dict(epoch=epoch+1,train_loss=loss_sum/len(train_images),val_f1=score,
                                           wall_seconds=elapsed,lr=float(optimizer.param_groups[0]['lr'])))
            if patience>=12: break
        result=dict(best_checkpoint=checkpoint,best_val_f1=best,epochs=len(times),epoch_seconds=times,
                    projected_full_run_seconds=projected_run_seconds(times,30) if len(times)>=3 else sum(times),
                    train_seconds=time.perf_counter()-start,
                    calibration='No seed-specific operating-point search; seed-42 raw tau and post-processing transferred')
        del model; torch.cuda.empty_cache()
        return result

    def train(self):
        self.verify_sources()
        original=read(self.out/'original_processing_locked.json')['models']
        with self.stage('train'):
            items=self.items(self.protocol['split']['val']); metrics=MetricSet([i['gt'] for i in items])
            guard=original['M4']['baseline']['cldice']
            p2=self.probabilities(self.descriptor('M2'),items,'M2_original_seed42')
            incumbent_descriptor=self.descriptor('M4'); incumbent_result=original['M4']
            candidates=[]; training_results=[]; timing_restricted=False
            def candidate(cell,loss,segments,weight=0.):
                nonlocal incumbent_descriptor,incumbent_result,timing_restricted
                config_path=self.run_config(cell,42,loss,segments,weight)
                descriptor,training=self.train_one(config_path); training_results.append(training)
                if training['projected_full_run_seconds']>1800 or training['train_seconds']>1800:
                    timing_restricted=True
                probabilities=self.probabilities(descriptor,items,f'{cell}_seed42')
                result=self.search_recipe(probabilities,metrics,Recipe(.66,.66),f'{cell}_seed42',guard)
                accepted=qualifies(result['metrics'],incumbent_result['metrics'],guard)
                candidates.append(dict(cell=cell,descriptor=descriptor,validation=result,accepted=accepted,
                                       incumbent_before=incumbent_result['metrics']))
                if accepted:
                    incumbent_descriptor,incumbent_result=descriptor,result
                save(self.out/'selection'/f'{cell}.json',candidates[-1])
                self.event('candidate_selection',cell=cell,accepted=accepted,metrics=result['metrics'])
                return descriptor,training,result
            candidate('M4_wce','weighted_ce',300)
            for weight in ([.3] if timing_restricted else [.1,.3,.5]):
                if timing_restricted and weight!=.3:
                    self.event('skipped',cell=f'M4_skeleton_{weight}',reason='30-minute timing gate')
                    continue
                candidate(f'M4_wce_skeleton_{str(weight).replace(".","p")}','weighted_ce',300,weight)
            selected_config=incumbent_descriptor['config']
            selected_loss=selected_config['training'].get('loss','focal')
            selected_weight=next((c['validation'] and read(Path(c['descriptor']['training']).parent/'config.json')['skeleton_weight']
                                  for c in candidates if c['descriptor']==incumbent_descriptor),0.)
            candidate('M4_finer600',selected_loss,600,selected_weight)
            selected_run=next((read(Path(c['descriptor']['training']).parent/'config.json')
                               for c in candidates if c['descriptor']==incumbent_descriptor),None)
            winner=dict(descriptor=incumbent_descriptor,validation=incumbent_result,
                        loss=selected_run['loss'] if selected_run else 'focal',
                        skeleton_weight=selected_run['skeleton_weight'] if selected_run else 0.,
                        segments=incumbent_descriptor['segments'],timing_restricted=timing_restricted)
            save(self.out/'selected_model.json',winner)
            final_m4={42:(incumbent_descriptor,incumbent_result)}
            for seed in (1,2):
                # Reserve full measured training upper bounds plus calibration for remaining M4 seeds.
                estimate=max(t['projected_full_run_seconds'] for t in training_results)*(3-seed)+180
                self.check(estimate=estimate)
                path=self.run_config('M4_selected',seed,winner['loss'],winner['segments'],winner['skeleton_weight'])
                descriptor,training=self.train_one(path); training_results.append(training)
                probabilities=self.probabilities(descriptor,items,f'M4_selected_seed{seed}')
                result=self.search_recipe(probabilities,metrics,Recipe(.66,.66),f'M4_selected_seed{seed}',guard)
                final_m4[seed]=descriptor,result
            comparator={42:dict(M2=self.descriptor('M2'),DeepCrack=self.descriptor('DeepCrack'))}
            # Preflight the complete optional package, conservatively using current graph timing
            # and the historical DC elapsed time with 4x headroom. Abort package, not budgets.
            dc_history=read(BASE/'results/multiseed/deepcrack/seed_42/result.json')['train_time_sec']
            search_seconds=[r['wall_seconds'] for r in (json.loads(line) for line in (self.out/'events.jsonl').read_text().splitlines())
                            if r.get('kind')=='operation' and r.get('name','').startswith('blend_search_')]
            calibration_reserve=3*max(search_seconds,default=180)+180
            optional_estimate=2*(max(t['projected_full_run_seconds'] for t in training_results)+4*dc_history)+calibration_reserve
            optional_started=False
            if time.time()+optional_estimate<self.deadline:
                optional_started=True
                try:
                    for seed in (1,2):
                        self.check(estimate=calibration_reserve)
                        path=self.run_config('M2_comparator',seed,'weighted_ce',300,model_name='M2')
                        descriptor,_=self.train_one(path)
                        comparator[seed]=dict(M2=descriptor)
                    for seed in (1,2):
                        self.check(estimate=calibration_reserve)
                        path=self.run_config('DeepCrack_comparator',seed,'focal',300,model_name='DeepCrack')
                        descriptor,_=self.train_one(path)
                        comparator[seed]['DeepCrack']=descriptor
                except BudgetExceeded as ex:
                    if time.time()+calibration_reserve>=self.deadline:
                        raise
                    optional_started=False
                    self.event('optional_package_stopped',reason=str(ex),fallback='fixed_seed42')
            else:
                self.event('skipped',cell='M2_and_DeepCrack_seeds1_2',reason='complete optional package cannot fit',estimate_seconds=optional_estimate)
            cases=[]
            for seed in (42,1,2):
                m4_descriptor,m4_result=final_m4[seed]
                comp=comparator[seed] if optional_started else comparator[42]
                p2_seed=self.probabilities(comp['M2'],items,f'M2_final_seed{seed}') if optional_started and seed!=42 else p2
                p4_seed=self.probabilities(m4_descriptor,items,f'M4_final_seed{seed}')
                blend=self.search_blend(p2_seed,p4_seed,metrics,m4_result,f'blend_final_seed{seed}',guard)
                m2_recipe=original['M2']['recipe']; dc_recipe=original['DeepCrack']['recipe']
                m2_val=metrics.score(process(p,m2_recipe) for p in p2_seed)
                cases.append(dict(model='M4',seed=seed,descriptor=m4_descriptor,recipe=m4_result['recipe'],validation=m4_result['metrics'],calibration='per_seed_validation'))
                cases.append(dict(model='M2+M4',seed=seed,descriptor=m4_descriptor,m2_descriptor=comp['M2'],
                                  alpha=blend['alpha'],recipe=blend['recipe'],validation=blend['metrics'],calibration='per_seed_validation_blend'))
                if optional_started or seed==42:
                    cases.append(dict(model='M2',seed=seed,descriptor=comp['M2'],recipe=m2_recipe,validation=m2_val,calibration='transferred_seed42_recipe'))
                    dc_prob=self.probabilities(comp['DeepCrack'],items,f'DeepCrack_final_seed{seed}')
                    for label,recipe in [('DeepCrack raw',Recipe(.36,.36).json()),('DeepCrack processed',dc_recipe)]:
                        cases.append(dict(model=label,seed=seed,descriptor=comp['DeepCrack'],recipe=recipe,
                                          validation=metrics.score(process(p,recipe) for p in dc_prob),calibration='transferred_seed42_recipe'))
            lock=dict(created=utc(),protocol_sha256=sha(self.out/'protocol.json'),
                      original_processing_sha256=sha(self.out/'original_processing_locked.json'),
                      comparator_mode='three_seeds' if optional_started else 'fixed_seed42',
                      cases=cases,originals={name:self.descriptor(name) for name in SPECS},
                      grids=GRIDS,selection=winner,negative_results=[c['cell'] for c in candidates if not c['accepted']],
                      calibration_asymmetry='M4 and blends recalibrated per seed; M2/DeepCrack use locked seed-42 operating points',
                      finer_slic_confound='600-segment native training also changes cached connectivity=True to False')
            save(self.out/'locked_configs.json',lock)
            save(self.out/'locked_configs.sha256.json',dict(sha256=sha(self.out/'locked_configs.json')))
            self.event('final_configuration_locked',comparator_mode=lock['comparator_mode'],cases=len(cases))

    def complexity(self, descriptor, item):
        from fvcore.nn import FlopCountAnalysis
        model=self.model(descriptor)
        parameters=sum(p.numel() for p in model.parameters())
        if descriptor['config']:
            graph,_=self.graph_input(item,descriptor['segments']); graph=graph.to(self.dev)
            inputs=(graph.x,graph.edge_index,graph.pos) if descriptor['config']['model']['type']=='hybrid' else (graph.x,graph.edge_index)
            nodes=int(graph.num_nodes); edges=int(graph.edge_index.shape[1])
        else:
            inputs=(torch.zeros(1,3,256,256,device=self.dev),); nodes=edges=None
        with torch.no_grad():
            fa=FlopCountAnalysis(model,inputs)
            fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
            gflops=2*float(fa.total())/1e9
        result=dict(parameters=parameters,gflops_fvcore=gflops,unsupported=dict(fa.unsupported_ops()),
                    convention='2 x fvcore MACs / 1e9',input='first locked test image; historical 256-square preprocessing',
                    nodes=nodes,directed_edges=edges)
        if descriptor['config']:
            m=descriptor['config']['model']; width=m.get('hidden_dim',64)
            layers=m.get('gnn_layers',2) if m['type']=='hybrid' else m.get('num_layers',2)
            macs=0; din=m['in_dim']
            for _ in range(layers): macs+=nodes*2*din*width+edges*din; din=width
            if m['type']=='hybrid':
                if m.get('use_pe',True): macs+=nodes*(width+m.get('pe_dim',32))*width
                macs+=m.get('transformer_layers',2)*(4*nodes*width*width+2*nodes*nodes*width+8*nodes*width*width)
            macs+=nodes*width*2
            result['gflops_analytic']=2*macs/1e9
        del model; torch.cuda.empty_cache()
        return result

    def final_test(self):
        self.verify_sources()
        lock_path=self.out/'locked_configs.json'
        if sha(lock_path)!=read(self.out/'locked_configs.sha256.json')['sha256']:
            raise ValueError('Final configuration lock changed')
        lock=read(lock_path)
        if lock['protocol_sha256']!=sha(self.out/'protocol.json') or lock['original_processing_sha256']!=sha(self.out/'original_processing_locked.json'):
            raise ValueError('Protocol or processing lock changed')
        # Exclusive, irrevocable consumption marker is written before any test labels open.
        save(self.out/'test_consumed.json',dict(time=utc(),lock_sha256=sha(lock_path),retry_allowed=False))
        with self.stage('final-test'):
            items=self.items(self.protocol['split']['test'],'test')
            metrics=MetricSet([i['gt'] for i in items]); cache={}; complexity={}
            def probs(descriptor):
                key=descriptor['checkpoint_sha256'],descriptor['segments']
                if key not in cache:
                    tag=f'{descriptor["name"]}_seed{descriptor["seed"]}_{key[0][:10]}'
                    cache[key]=self.probabilities(descriptor,items,tag,'test')
                return cache[key]
            reproduction={}
            with self.operation('historical_reproduction'):
                for name,descriptor in lock['originals'].items():
                    tau=SPECS[name][2]
                    score=metrics.score((p>=np.float32(tau) for p in probs(descriptor)),True)
                    expected=read(PITCH/f'historical_{name}.json')['accuracy']
                    reference=dict(cldice=expected['cldice'],strict_cr=expected['overlap_sweep'][2]['micro'],pixel_f1=expected['pixel_f1'])
                    matched=all(round(score[k],6)==round(v,6) for k,v in reference.items())
                    matched=matched and score['n_total']==992 and score['n_preserved']==expected['overlap_sweep'][2]['preserved']
                    matched=matched and [b['total'] for b in score['size_bins']]==[167,274,551]
                    matched=matched and all(round(b['strict_cr'],6)==round(old['micro'],6)
                                           for b,old in zip(score['size_bins'],expected['size_buckets']))
                    matched=matched and all(round(b['strict_cr'],6)==round(old['micro'],6)
                                           for b,old in zip(score['overlap_sweep'],expected['overlap_sweep']))
                    reproduction[name]=dict(metrics=score,reference=reference,matched=matched)
                    save(self.out/'test'/f'reproduction_{name}.json',reproduction[name])
                    if not matched:
                        raise ValueError(f'PITCH_RERUN reproduction failed for {name}; test is consumed and no comparisons are accepted')
            results=[]
            for index,case in enumerate(lock['cases']):
                self.check(); descriptor=case['descriptor']; p=probs(descriptor)
                if case['model']=='M2+M4':
                    p2=probs(case['m2_descriptor']); alpha=np.float32(case['alpha']); beta=np.float32(1-case['alpha'])
                    p=[alpha*a+beta*b for a,b in zip(p2,p)]
                with self.operation(f'test_metrics_{case["model"]}_seed{case["seed"]}'):
                    score=metrics.score((process(probability,case['recipe']) for probability in p),True)
                key=descriptor['checkpoint_sha256'],descriptor['segments']
                if key not in complexity:
                    complexity[key]=self.complexity(descriptor,items[0])
                cost=dict(complexity[key])
                if case['model']=='M2+M4':
                    d2=case['m2_descriptor']; k2=d2['checkpoint_sha256'],d2['segments']
                    if k2 not in complexity: complexity[k2]=self.complexity(d2,items[0])
                    cost=dict(parameters=cost['parameters']+complexity[k2]['parameters'],
                              gflops_fvcore=cost['gflops_fvcore']+complexity[k2]['gflops_fvcore'],
                              constituent_costs=[cost,complexity[k2]],convention='sum of both networks; 2 x MACs')
                result=dict(model=case['model'],seed=case['seed'],test=score,validation=case['validation'],complexity=cost,
                            val_minus_test_pp={k:100*(case['validation'][k]-score[k]) for k in ('strict_cr','cldice','pixel_f1')},
                            recipe=case['recipe'],alpha=case.get('alpha'),calibration=case['calibration'])
                save(self.out/'test'/f'case_{index:02d}.json',result); results.append(result)
                print(f'TEST {case["model"]} seed{case["seed"]}: CR={score["strict_cr"]:.4f}, clDice={score["cldice"]:.4f}, F1={score["pixel_f1"]:.4f}',flush=True)
            existing=read(self.out/'existing_artifacts.json'); after=snapshot()
            if existing!=after:
                save(self.out/'original_artifact_changes.json',dict(before=existing,after=after))
                raise ValueError('Existing result artifacts changed during experiment')
            save(self.out/'final_results.json',dict(time=utc(),lock_sha256=sha(lock_path),reproduction=reproduction,
                                                   cases=results,original_artifacts_unchanged=True))

    def report(self):
        with self.stage('report'):
            final=read(self.out/'final_results.json') if (self.out/'final_results.json').exists() else None
            lock=read(self.out/'locked_configs.json') if (self.out/'locked_configs.json').exists() else None
            protocol=self.protocol if (self.out/'protocol.json').exists() else None
            lines=['# M2/M4 one-day improvement experiment','',f'Generated {utc()}.', '',
                   f'Status: **{"complete" if final else "stopped; no accepted final comparison"}**.', '',
                   'Baseline: DeepSegmentor DeepCrackNet, batch normalization, ngf=64. Raw and post-processed rows use the same checkpoint for each seed; the paper_v2 VGG/HED reimplementation is excluded.', '',
                   'Seeds: 42, 1, 2. Train/validation membership is fixed to seed 42: 210 training, 45 validation, 45 unused; independent test has 237 images.', '',
                   'Selection retains the incumbent unless validation strict CR improves by at least 1 percentage point and the validation clDice guard is satisfied. M4 and blends are recalibrated per seed; M2 and DeepCrack transfer the locked seed-42 thresholds and processing without a new search.', '',
                   'Historical native training caches use connectivity=True. Inference always uses False. The finer-SLIC condition changes native training to connectivity=False as well as increasing segments to 600, so its effect cannot be attributed to segment count alone.', '',
                   'Dataset provenance limitation: upstream train/test directories reuse ten stems for byte-distinct images, so sample identity includes the raw-directory split. The prior source-ID audit reports 19/237 test crops sharing inferred source IDs with the train folder, and 39/45 seed-42 validation crops sharing source IDs with training. The requested PITCH_RERUN split is preserved; this is not a source-disjoint benchmark.', '',
                   '## Validation ablations','',
                   '| Run | Change | Before CR | After CR | Before clDice | After clDice | Before F1 | After F1 | Selected settings |',
                   '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
            negatives=[]
            for path in sorted((self.out/'validation').glob('*.json')):
                value=read(path); baseline=value['baseline']; selected=value['metrics']
                if not value['improved']: negatives.append(f'{path.stem}: incumbent retained; no eligible +1pp improvement.')
                def row(change,before,after,settings):
                    lines.append(f'| {path.stem} | {change} | {before["strict_cr"]:.6f} | {after["strict_cr"]:.6f} | {before["cldice"]:.6f} | {after["cldice"]:.6f} | {before["pixel_f1"]:.6f} | {after["pixel_f1"]:.6f} | `{json.dumps(settings)}` |')
                if 'independent' in value:
                    for name,entry in value['independent'].items():
                        row('independent '+name,entry['before'],entry['after'],entry['recipe'])
                        if not entry['improved']: negatives.append(f'{path.stem}, independent {name}: no eligible improvement.')
                    before=baseline
                    for entry in value['cumulative']:
                        row('cumulative '+entry['change'],before,entry['metrics'],entry['recipe'])
                        if entry['metrics']['strict_cr']-before['strict_cr']<.01-1e-12:
                            negatives.append(f'{path.stem}, cumulative {entry["change"]}: incumbent retained.')
                        before=entry['metrics']
                row('final selection',baseline,selected,dict(recipe=value['recipe'],alpha=value.get('alpha')))
            lines+=['','All candidate metrics and timings are retained in validation_trials/; the tables show accepted selections and retained incumbents.','',
                    '## Training and model selection','',
                    '| Candidate | Accepted | Val strict CR | Val clDice | Val pixel F1 |','| --- | --- | --- | --- | --- |']
            for path in sorted((self.out/'selection').glob('*.json')):
                value=read(path); score=value['validation']['metrics']
                lines.append(f'| {value["cell"]} | {value["accepted"]} | {score["strict_cr"]:.6f} | {score["cldice"]:.6f} | {score["pixel_f1"]:.6f} |')
                if not value['accepted']: negatives.append(f'{value["cell"]}: did not replace the current model under the +1pp/clDice rule.')
            events=[json.loads(line) for line in (self.out/'events.jsonl').read_text().splitlines()]
            timing=[e for e in events if e['kind']=='timing_gate']
            lines+=['',f'30-minute timing gate decisions: `{json.dumps(timing)}`','']
            if final:
                rows=final['cases']; groups={name:[r for r in rows if r['model']==name] for name in dict.fromkeys(r['model'] for r in rows)}
                def mean_sd(values):
                    return f'{np.mean(values):.6f} ± {np.std(values,ddof=1):.6f}' if len(values)>1 else f'{values[0]:.6f} (SD unavailable; n=1)'
                lines+=['## Main benchmark','',f'Comparator mode: **{lock["comparator_mode"]}**. Historical reproduction controls all passed.','',
                        '| Model | Seeds | clDice | Strict micro CR | Pixel F1 | Parameters | GFLOPs fvcore |','| --- | --- | --- | --- | --- | --- | --- |']
                for name,runs in groups.items():
                    lines.append(f'| {name} | {", ".join(str(r["seed"]) for r in runs)} | {mean_sd([r["test"]["cldice"] for r in runs])} | {mean_sd([r["test"]["strict_cr"] for r in runs])} | {mean_sd([r["test"]["pixel_f1"] for r in runs])} | {runs[0]["complexity"]["parameters"]} | {mean_sd([r["complexity"]["gflops_fvcore"] for r in runs])} |')
                lines+=['','GFLOPs use 2×MACs on the first locked test image after historical 256×256 preprocessing. Actual graph sizes and unsupported operations are recorded in final_results.json; these counts may omit unsupported attention operations. Historical PITCH_RERUN used synthetic graphs. No latency optimization or new latency benchmark was performed.','',
                        '## Size connectivity','', '| Model | [0,50) | [50,200) | [200,inf) |','| --- | --- | --- | --- |']
                for name,runs in groups.items():
                    lines.append('| '+name+' | '+' | '.join(mean_sd([r['test']['size_bins'][i]['strict_cr'] for r in runs]) for i in range(3))+' |')
                lines+=['','## Overlap sweep','', '| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |','| --- | --- | --- | --- | --- | --- |']
                for name,runs in groups.items():
                    lines.append('| '+name+' | '+' | '.join(mean_sd([r['test']['overlap_sweep'][i]['strict_cr'] for r in runs]) for i in range(5))+' |')
                lines+=['','## Validation–test gaps','', 'Positive gaps mean validation was higher. Values are percentage points; means and SD are computed across matched seed results.','',
                        '| Model | Val strict CR | Val clDice | Val pixel F1 | Val−test CR pp | Val−test clDice pp | Val−test F1 pp |',
                        '| --- | --- | --- | --- | --- | --- | --- |']
                for name,runs in groups.items():
                    lines.append('| '+name+' | '+' | '.join(mean_sd([r['validation'][k] for r in runs]) for k in ('strict_cr','cldice','pixel_f1'))+' | '+
                                 ' | '.join(mean_sd([r['val_minus_test_pp'][k] for r in runs]) for k in ('strict_cr','cldice','pixel_f1'))+' |')
                lines+=['','## Historical before / final after','', '| Model | Historical seed-42 CR | Final CR | Historical clDice | Final clDice | Historical pixel F1 | Final pixel F1 |',
                        '| --- | --- | --- | --- | --- | --- | --- |']
                for name,group in [('M2','M2'),('M4','M4'),('DeepCrack','DeepCrack processed')]:
                    old=final['reproduction'][name]['metrics']; runs=groups[group]
                    lines.append(f'| {group} | {old["strict_cr"]:.6f} | {mean_sd([r["test"]["strict_cr"] for r in runs])} | {old["cldice"]:.6f} | {mean_sd([r["test"]["cldice"] for r in runs])} | {old["pixel_f1"]:.6f} | {mean_sd([r["test"]["pixel_f1"] for r in runs])} |')
                lines+=['','Historical values are single seed-42 controls; final multi-seed differences are descriptive rather than paired loss-only effects.','',
                        '## M4 versus DeepCrack','']
                if lock['comparator_mode']=='three_seeds':
                    for comparator_name in ('DeepCrack raw','DeepCrack processed'):
                        deltas={k:float(np.mean([r['test'][k] for r in groups['M4']])-np.mean([r['test'][k] for r in groups[comparator_name]]))
                                for k in ('strict_cr','cldice','pixel_f1')}
                        lines.append(f'- M4 minus {comparator_name}, three-seed means: strict CR {100*deltas["strict_cr"]:+.3f} pp; clDice {100*deltas["cldice"]:+.3f} pp; pixel F1 {100*deltas["pixel_f1"]:+.3f} pp. No statistical-significance claim is made.')
                else:
                    lines.append('Comparators are fixed seed-42 checkpoints. This does not support a three-seed claim that M4 beats DeepCrack.')
            else:
                lines+=['## Final test','', 'No accepted final test comparison is available. No performance claim is made.']
            for event in events:
                if event['kind'] in ('skipped','optional_package_stopped'):
                    negatives.append(f'{event.get("cell","optional comparator package")}: {event["reason"]}.')
            lines+=['','## Negative results and omitted work','']+(['- '+n for n in negatives] or ['- None recorded.'])
            lines+=['','## Metric definitions','',
                    'Strict pitch CR counts all GT components and requires a connected GT/prediction intersection at sufficient coverage. paper_v2 topology excludes components below 20 pixels and accepts coverage by one predicted component without requiring the intersection to be connected. These are different metrics.', '',
                    'Bridging outside GT can connect a prediction without repairing its strict GT intersection. clDice follows crack_metrics.py, including its empty-mask handling; pixel F1 pools pixel confusion counts across images.', '',
                    'Size bins are [0,50), [50,200), [200,inf). Test overlap sweeps are descriptive evaluations of locked masks, not threshold fitting.','',
                    '## Wall times','', '| Stage/operation | Seconds |','| --- | --- |']
            for event in events:
                if event['kind'] in ('stage_end','operation'):
                    lines.append(f'| {event.get("name",event.get("stage"))} | {event["wall_seconds"]:.3f} |')
            lines+=['','Per-epoch timing is retained in runs/*/seed_*/epochs.jsonl and events.jsonl.','', '## New checkpoints and exact commands','']
            manifests=[]
            for path in sorted((self.out/'runs').glob('*/seed_*/training.json')):
                value=read(path); config=path.parent/'config.json'; command=read(path.parent/'command.json')
                checkpoints=[dict(path=str(p),sha256=sha(p)) for p in sorted(path.parent.glob('checkpoint_epoch_*.pt'))]
                manifests.append(dict(run=str(path.parent),config=str(config),best_checkpoint=value['best_checkpoint'],
                                      checkpoints=checkpoints,command=command))
                lines += [f'- **{path.parent.parent.name}/{path.parent.name}**: best checkpoint `{value["best_checkpoint"]}`; all immutable checkpoint paths are in checkpoint_manifest.json.',
                          f'  - Config: `{config}`',f'  - Command (cwd `{command["cwd"]}`): `{command["command"]}`']
            save(self.out/'checkpoint_manifest.json',manifests)
            failure_files=sorted(self.out.glob('failure_*.json'))
            if failure_files:
                lines+=['','## Stop reason','']+[f'- {read(p)["error"]}' for p in failure_files]
            lines+=['','## Reproduction','',
                    'Run commands from pipeline-crack-gnn. The immutable configs and checkpoint manifest capture individual runs. To reproduce the complete workflow, use a new output folder:', '',
                    '```powershell',f'py -3.11 paper_v2/improve_m2_m4.py run --output results/paper_v2/improve_m2_m4_REPRODUCE --cache-policy preserve',
                    '```','', 'Stage commands and exact invocations are recorded in events.jsonl. A consumed test evaluation cannot be rerun in the same output directory.','']
            with (self.out/'RESULTS.md').open('x',encoding='utf-8') as f:
                f.write('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['run','protocol','search','train','train-one','final-test','report'])
    parser.add_argument('--output',required=True)
    parser.add_argument('--config')
    parser.add_argument('--cache-policy',choices=['preserve'],default='preserve')
    parser.add_argument('--started-at',help='UTC ISO timestamp including protocol implementation work')
    args=parser.parse_args()
    if args.stage in ('run','protocol'):
        path=Path(args.output).resolve()
        path.mkdir(parents=True,exist_ok=False)
        if args.started_at:
            started=datetime.fromisoformat(args.started_at).timestamp()
            save(path/'protocol_started.json',dict(unix_time=started,time=args.started_at,budget_seconds=STEPS['protocol']))
    experiment=Experiment(args.output)
    experiment.event('invocation',argv=sys.argv,cwd=os.getcwd())
    try:
        if args.stage=='run':
            experiment.freeze(args.cache_policy)
            experiment.search()
            experiment.train()
            experiment.final_test()
            experiment.report()
        elif args.stage=='protocol': experiment.freeze(args.cache_policy)
        elif args.stage=='search': experiment.search()
        elif args.stage=='train': experiment.train()
        elif args.stage=='final-test': experiment.final_test()
        elif args.stage=='report': experiment.report()
        elif args.stage=='train-one':
            if not args.config: parser.error('train-one requires --config')
            with experiment.stage('train'): experiment.train_one(args.config)
    except Exception as ex:
        name='failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'
        save(experiment.out/name,dict(time=utc(),error=f'{type(ex).__name__}: {ex}',traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True)
        if not (experiment.out/'RESULTS.md').exists():
            try: experiment.report()
            except Exception: print(traceback.format_exc(),flush=True)
        raise SystemExit(1)


if __name__=='__main__':
    main()
