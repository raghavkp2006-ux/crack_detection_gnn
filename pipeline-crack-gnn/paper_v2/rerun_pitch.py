"""Rerun the attached pitch's experiments without replacing primary artifacts.

Historical 256-square inference and original checkpoints are kept separate from
the current five-seed node predictions. No training or threshold fitting occurs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

import cv2
import numpy as np
from scipy import ndimage as ndi
import torch

import common

ROOT = Path(common.BASE).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'DeepSegmentor'))
import crack_metrics as cm
from models.deepcrack_networks import DeepCrackNet
from src.models.hybrid_model import build_model
from src.train import load_config
from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph
from benchmark_efficiency import analytic_gmacs, fvcore_count, q
from stats_core import load_labels, load_probs, decide

S8 = np.ones((3, 3), int)
OVERLAPS = [0.3, 0.4, 0.5, 0.6, 0.7]
BUCKETS = ['Small (<50px)', 'Medium (50-200px)', 'Large (>200px)']
SPECS = {
    'M1': ('configs/baseline_deep_gnn.yaml', 'results/M1_deep_gnn/best_model.pt', .60),
    'M2': ('configs/shallow_gnn.yaml', 'results/M2_shallow_gnn/best_model.pt', .70),
    'M3': ('configs/hybrid_no_pe.yaml', 'results/M3_hybrid_no_pe/best_model.pt', .74),
    'M4': ('configs/step2_loss/hybrid_full_focal.yaml', 'results/step2_loss/M4_hybrid_full/best_model.pt', .66),
    'DeepCrack': (None, 'results/multiseed/deepcrack/seed_42/best_model.pt', .36),
}


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def original_snapshot(output):
    return {str(p.relative_to(common.BASE)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in Path(common.BASE, 'results').rglob('*')
            if p.is_file() and not p.is_relative_to(output)}


def load_historical(key, device):
    cfgp, ckpt, tau = SPECS[key]
    path = Path(common.BASE, ckpt)
    if key == 'DeepCrack':
        cfg = None
        model = DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch')
    else:
        cfg = load_config(common.rel(cfgp))
        model = build_model(cfg)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.to(device).eval()
    wrapper = (cm.DeepCrackSegmentorWrapper(model, tau, str(device)) if cfg is None
               else cm.GNNSegmentorWrapper(model, cfg['model']['type'], tau, str(device)))
    return model, wrapper, cfg, path


def image_components(gt):
    labels, _ = ndi.label(gt, S8)
    parts = []
    for cid, sl in enumerate(ndi.find_objects(labels), 1):
        mask = labels[sl] == cid
        area = int(mask.sum())
        bucket = 0 if area < 50 else (1 if area < 200 else 2)
        parts.append((sl, mask, area, bucket))
    return parts


def component_counts(pred, parts):
    """Same strict intersection-connectivity definition as crack_metrics.py.

    Cache intersection status once for all five overlap thresholds. The crop
    contains the complete GT component, so its connectivity is unchanged.
    """
    pl, _ = ndi.label(pred, S8)
    counts = np.zeros((len(OVERLAPS), 3, 4), np.int64)
    for sl, mask, area, bucket in parts:
        overlap = mask & pred[sl]
        coverage = float(overlap.sum()) / area
        _, nc = ndi.label(overlap, S8)
        preserved_shape = nc == 1 and len(np.unique(pl[sl][overlap])) == 1
        for ti, threshold in enumerate(OVERLAPS):
            category = 2 if coverage < threshold else (1 if preserved_shape else 3)
            counts[ti, bucket, 0] += 1
            counts[ti, bucket, category] += 1
    assert np.array_equal(counts[:, :, 0], counts[:, :, 1:].sum(2))
    return counts


def summarize(counts, cldice, pixel_confusion):
    # counts: images x thresholds x size buckets x [total,preserved,missed,fragmented]
    overall = counts.sum((0, 2))
    result = {'cldice': float(np.mean(cldice)), 'pixel_f1':
              float(2 * pixel_confusion[0] / max(2 * pixel_confusion[0] + pixel_confusion[1] + pixel_confusion[2], 1)),
              'pixel_iou': float(pixel_confusion[0] / max(sum(pixel_confusion), 1)),
              'overlap_sweep': [], 'size_buckets': []}
    per_image = counts.sum(2)
    for ti, threshold in enumerate(OVERLAPS):
        total, preserved, missed, fragmented = map(int, overall[ti])
        # Dataset has no empty GT images; checked before inference.
        macro = per_image[:, ti, 1] / per_image[:, ti, 0]
        result['overlap_sweep'].append(dict(overlap_threshold=threshold,
            total=total, preserved=preserved, missed=missed, fragmented=fragmented,
            micro=preserved / total, macro=float(macro.mean())))
    for bi, bucket in enumerate(BUCKETS):
        total, preserved, missed, fragmented = map(int, counts[:, 2, bi].sum(0))
        result['size_buckets'].append(dict(bucket=bucket, total=total,
            preserved=preserved, missed=missed, fragmented=fragmented,
            micro=preserved / total if total else 0.0))
    return result


def evaluate_masks(iterator, gts, parts, name):
    counts, cls, pix = [], [], np.zeros(3, np.int64)
    for i, pred in enumerate(iterator):
        gt = gts[i]
        assert pred.shape == gt.shape, f'{name}/{i}: mask shape mismatch'
        pred = pred.astype(bool)
        c = component_counts(pred, parts[i])
        if i == 0:
            reference = cm.connectivity_ratio_breakdown(pred, gt, .5)
            assert c[2].sum(0).tolist() == [reference[k] for k in ['n_total', 'n_preserved', 'n_missed', 'n_fragmented']]
        counts.append(c)
        cls.append(cm.cl_dice(pred, gt))
        pix += [(pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()]
        if (i + 1) % 50 == 0:
            print(f'{name}: {i + 1}/{len(gts)} images', flush=True)
    assert len(counts) == len(gts), name
    return summarize(np.array(counts), cls, pix)


def timed(fn, device, n=500, warm=15):
    times = []
    with torch.no_grad():
        for i in range(warm + n):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize(device)
            if i >= warm:
                times.append((time.perf_counter() - start) * 1000)
    return dict(**q(times), mean=float(np.mean(times)), sd=float(np.std(times, ddof=1)),
                timed_iterations=n, warmup=warm)


def report(out, results):
    lines = ['# Project pitch rerun', '',
        'Historical pitch checkpoints and 256x256 inference are evaluated separately from current five-seed node predictions.',
        'No retraining, primary threshold changes, or test-set threshold fitting.', '',
        '## Historical checkpoint benchmark', '',
        '| Model | clDice | Strict micro CR | Pixel F1 | Parameters | GFLOPs fvcore | Model median ms | Historical E2E median ms |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for key, r in results['historical'].items():
        a, e = r['accuracy'], r['efficiency']
        lines.append(f'| {key} | {a["cldice"]} | {a["overlap_sweep"][2]["micro"]} | {a["pixel_f1"]} | {e["parameters"]} | {e["gflops_fvcore_historical"]} | {e["model_only"]["latency_ms"]} | {e["historical_e2e"]["end_to_end_latency_ms"]} |')
    for label, models in [('Historical size connectivity', {k: v['accuracy'] for k, v in results['historical'].items()}),
                          ('Current five-seed mean size connectivity', results['current_five_seed_means'])]:
        lines += ['', f'## {label}', '', '| Model | Small | Medium | Large |', '| --- | --- | --- | --- |']
        for key, a in models.items():
            values = [b['micro'] for b in a['size_buckets']]
            lines.append(f'| {key} | {values[0]} | {values[1]} | {values[2]} |')
    lines += ['', '## Historical overlap sweep', '', '| Overlap | M4 micro | DeepCrack micro | Winner |', '| --- | --- | --- | --- |']
    a = results['historical']['M4']['accuracy']['overlap_sweep']
    b = results['historical']['DeepCrack']['accuracy']['overlap_sweep']
    for x, y in zip(a, b):
        winner = 'M4' if x['micro'] > y['micro'] else ('DeepCrack' if y['micro'] > x['micro'] else 'Tie')
        lines.append(f'| {x["overlap_threshold"]} | {x["micro"]} | {y["micro"]} | {winner} |')
    lines += ['', '## SLIC connectivity control', '', 'Fixed historical M4 checkpoint and tau=0.66; inference-only preprocessing comparison.', '',
              '| enforce_connectivity | clDice | Strict micro CR | SLIC median ms | Mean nodes | Mean edges |',
              '| --- | --- | --- | --- | --- | --- |']
    for flag, r in results['slic_control'].items():
        lines.append(f'| {flag} | {r["accuracy"]["cldice"]} | {r["accuracy"]["overlap_sweep"][2]["micro"]} | {r["slic_timing_ms"]["median"]} | {np.mean(r["nodes"])} | {np.mean(r["directed_edges"])} |')
    lines += ['', '## Positional encoding cost', '', '```json', json.dumps(results['pe_cost'], indent=2), '```',
              '', '## Protocol notes', ''] + ['- ' + n for n in results['notes']]
    lines += ['', '## Wall times in seconds', '', '```json', json.dumps(results['wall_seconds'], indent=2), '```', '']
    with (out / 'PITCH_RERUN.md').open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    assert out.is_relative_to(Path(common.OUT).resolve()), 'Output must be beneath results/paper_v2'
    out.mkdir(exist_ok=False)
    before = original_snapshot(out)
    save(out / 'existing_results_manifest.json', before)
    started = time.perf_counter()
    result = {'historical': {}, 'slic_control': {}, 'current_five_seed': {},
              'current_five_seed_means': {}, 'wall_seconds': {}, 'notes': [
        'Historical baseline: DeepSegmentor DeepCrackNet with batch normalization, seed-42 checkpoint. Current baseline: separate VGG16/HED DeepCrack-style reimplementation.',
        'Historical fixed thresholds M1=.60, M2=.70, M3=.74, M4=.66, DeepCrack=.36 are unchanged; current thresholds loaded unchanged from saved NPZ files.',
        'Strict historical CR requires sufficient GT overlap AND exactly one connected intersection; includes every GT component without a minimum-area cutoff. This differs from paper_v2 topology definition.',
        'Historical size-bin implementation is [0,50), [50,200), [200,infinity), despite its printed medium/large labels; preserved exactly for reproduction.',
        'Historical E2E function resizes outside timing; CNN input normalization/transfer and binary-mask reconstruction are excluded. These times are not full native-resolution RGB-to-mask latency.',
        'Historical model-only timing/FLOPs uses synthetic graph tensors, despite the pitch describing real-graph profiling. Real first-image graph FLOPs are additionally measured; unsupported fvcore ops are recorded.',
        'Connectivity switch is an inference-only sensitivity check using one fixed trained checkpoint; it does not establish a retraining-based causal improvement.',
        'Cross-sensor and morphology experiments appear as proposed future work in the pitch; no new sensor experiment or retraining is included.',
        'Python launcher could not locate Python inside sandbox; requested py -3.11 works with approved elevated access.',
        'Current five-seed supplementary analysis uses existing probabilities, recomputes all pixel masks and strict historical topology metrics, and leaves primary predictions intact.',
    ]}
    try:
        t = time.perf_counter()
        log = out / 'metric_unit_tests.log'
        with log.open('x', encoding='utf-8') as f:
            suite = unittest.defaultTestLoader.loadTestsFromName('test_crack_metrics')
            tests = unittest.TextTestRunner(stream=f, verbosity=2).run(suite)
        if not tests.wasSuccessful():
            raise RuntimeError(f'Metric tests failed; see {log}')
        result['unit_tests'] = dict(run=tests.testsRun, failures=len(tests.failures), errors=len(tests.errors))
        result['wall_seconds']['metric_tests'] = time.perf_counter() - t
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for pitch efficiency rerun')
        device = torch.device('cuda')
        result['gpu'] = torch.cuda.get_device_name(0)
        result['gpu_preflight'] = subprocess.check_output(['nvidia-smi'], text=True)
        result['torch'] = torch.__version__
        result['tf32'] = dict(matmul=torch.backends.cuda.matmul.allow_tf32, cudnn=torch.backends.cudnn.allow_tf32)
        np.random.seed(42)
        torch.manual_seed(42)
        # Validate all required files before spending time benchmarking.
        for _, ckpt, _ in SPECS.values():
            if not Path(common.BASE, ckpt).is_file():
                raise FileNotFoundError(ckpt)
        y, offsets, stems = load_labels()
        items = []
        for stem in stems:
            ip = common.find(str(Path(common.DC, 'test_img')), stem)
            gp = common.find(str(Path(common.DC, 'test_lab')), stem)
            image = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gp, cv2.IMREAD_GRAYSCALE) > 127
            assert image.shape[:2] == gt.shape, stem
            assert gt.any(), 'Empty GT case needs explicit macro handling'
            items.append((image, gt, {'image_id': stem}))
        assert len(items) == 237
        gts = [x[1] for x in items]
        parts = [image_components(gt) for gt in gts]
        result['test_images'] = len(items)
        result['gt_components'] = sum(map(len, parts))
        result['stems'] = stems
        sample = items[0][0]
        small = cv2.resize(sample, (256, 256), interpolation=cv2.INTER_LINEAR)
        for key in SPECS:
            t = time.perf_counter()
            print(f'HISTORICAL BENCHMARK {key}', flush=True)
            model, wrapper, cfg, ckpt = load_historical(key, device)
            efficiency = dict(parameters=sum(p.numel() for p in model.parameters()),
                gflops_fvcore_historical=cm.count_gflops(wrapper, device=device),
                model_only=cm.measure_fps(wrapper, device=device, n_runs=500),
                historical_e2e=cm.benchmark_end_to_end(wrapper, sample, key, device=device, n_runs=500))
            if cfg:
                seg = compute_superpixels(small, 300, 10., 1.)
                ft, ce = extract_node_features(small, seg)
                ei, _ = symmetrize_graph(*build_pirm_graph(ft, 8))
                inp = (torch.from_numpy(ft).to(device), torch.from_numpy(ei).long().to(device))
                if cfg['model']['type'] == 'hybrid':
                    inp += (torch.from_numpy(ce).to(device),)
                fv, unsupported = fvcore_count(model, inp)
                efficiency['real_graph'] = dict(nodes=len(ft), directed_edges=ei.shape[1],
                    gflops_fvcore=2 * fv if fv is not None else None, unsupported=unsupported,
                    gflops_analytic=2 * analytic_gmacs(cfg, len(ft), ei.shape[1]))
            predictions = (wrapper.predict_mask(image, meta) for image, _, meta in items)
            accuracy = evaluate_masks(predictions, gts, parts, key)
            r = dict(checkpoint=str(ckpt.relative_to(ROOT)), checkpoint_sha256=fingerprint(ckpt),
                     threshold=SPECS[key][2], efficiency=efficiency, accuracy=accuracy)
            result['historical'][key] = r
            save(out / f'historical_{key}.json', r)
            result['wall_seconds'][f'historical_{key}'] = time.perf_counter() - t
            print(f'{key} clDice={accuracy["cldice"]}, CR={accuracy["overlap_sweep"][2]["micro"]}', flush=True)
            del wrapper, model
        t = time.perf_counter()
        print('SLIC CONNECTIVITY CONTROL', flush=True)
        model, wrapper, cfg, _ = load_historical('M4', device)
        for flag in [False, True]:
            wrapper.enforce_connectivity = flag
            stats = {'nodes': [], 'directed_edges': []}
            def iterator():
                for image, _, meta in items:
                    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
                    seg = compute_superpixels(img, 300, 10., 1., enforce_connectivity=flag)
                    ft, _ = extract_node_features(img, seg)
                    ei, _ = symmetrize_graph(*build_pirm_graph(ft, 8))
                    stats['nodes'].append(len(ft))
                    stats['directed_edges'].append(ei.shape[1])
                    yield wrapper.predict_mask(image, meta)
            stats['accuracy'] = evaluate_masks(iterator(), gts, parts, f'M4 connectivity={flag}')
            stats['slic_timing_ms'] = timed(lambda: compute_superpixels(small, 300, 10., 1., enforce_connectivity=flag), device)
            result['slic_control'][str(flag)] = stats
        false = result['slic_control']['False']['accuracy']
        assert false == result['historical']['M4']['accuracy'], 'M4 false control changed predictions'
        save(out / 'slic_control.json', result['slic_control'])
        result['wall_seconds']['slic_control'] = time.perf_counter() - t
        t = time.perf_counter()
        seg = compute_superpixels(small, 300, 10., 1.)
        ft, ce = extract_node_features(small, seg)
        ei, _ = symmetrize_graph(*build_pirm_graph(ft, 8))
        x = torch.from_numpy(ft).to(device)
        e = torch.from_numpy(ei).long().to(device)
        p = torch.from_numpy(ce).to(device)
        with torch.no_grad():
            model(x, e, p)
            h = model.gnn_layer_outputs[-1]
        pe = timed(lambda: model.pe_fusion(h, p), device)
        e2e = result['historical']['M4']['efficiency']['historical_e2e']['end_to_end_latency_ms']
        result['pe_cost'] = dict(parameters=sum(p.numel() for p in model.pe_fusion.parameters()),
            real_graph_nodes=len(ft), pe_fusion_gpu_ms=pe, percent_of_historical_m4_e2e=100 * pe['median'] / e2e,
            note='Direct PE module timing includes sinusoidal construction, concatenation, and projection; M3/M4 whole-model timing difference is not a pure PE measurement.')
        save(out / 'pe_cost.json', result['pe_cost'])
        result['wall_seconds']['pe_cost'] = time.perf_counter() - t
        del model, wrapper
        t = time.perf_counter()
        segments = [common.get_segments(s, 'test') for s in stems]
        inverse = []
        for i, seg in enumerate(segments):
            labels, inv = np.unique(seg, return_inverse=True)
            assert seg.shape == gts[i].shape and len(labels) == offsets[i + 1] - offsets[i], stems[i]
            inverse.append(inv.reshape(seg.shape))
        for key in ['M1_focal', 'M2_focal', 'M3_focal', 'M4_focal', 'DeepCrack', 'M1_wce']:
            runs = []
            for seed in common.SEEDS:
                probs, tau = load_probs(key, seed)
                node_predictions = decide(probs, tau)
                predictions = (node_predictions[offsets[i]:offsets[i + 1]][inv] for i, inv in enumerate(inverse))
                a = evaluate_masks(predictions, gts, parts, f'current {key}/{seed}')
                a['seed'], a['threshold'] = seed, tau
                runs.append(a)
            result['current_five_seed'][key] = runs
            result['current_five_seed_means'][key] = dict(
                cldice=float(np.mean([r['cldice'] for r in runs])),
                strict_micro_cr=float(np.mean([r['overlap_sweep'][2]['micro'] for r in runs])),
                size_buckets=[dict(bucket=BUCKETS[b], micro=float(np.mean([r['size_buckets'][b]['micro'] for r in runs])),
                                   sd=float(np.std([r['size_buckets'][b]['micro'] for r in runs], ddof=1))) for b in range(3)])
            save(out / f'current_{key}.json', runs)
        result['wall_seconds']['current_five_seed'] = time.perf_counter() - t
        after = original_snapshot(out)
        assert before == after, 'An existing results artifact was changed during rerun'
        result['existing_results_unchanged'] = True
        result['wall_seconds']['total'] = time.perf_counter() - started
        save(out / 'pitch_rerun.json', result)
        report(out, result)
        print(f'COMPLETE: {out / "PITCH_RERUN.md"}', flush=True)
    except Exception as exc:
        save(out / 'FAILURE.json', dict(error=repr(exc), wall_seconds=time.perf_counter() - started,
                                       existing_results_unchanged=before == original_snapshot(out)))
        raise


if __name__ == '__main__':
    main()
