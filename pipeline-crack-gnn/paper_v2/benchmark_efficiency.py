import os, time, json, platform, subprocess
import numpy as np
import torch
import torch.nn as nn
from common import *
from stats_core import load_probs, load_labels
from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph
from src.models.hybrid_model import build_model
from src.train import load_config
from src.utils.dataset import CrackGraphDataset
from deepcrack_model import DeepCrackHED, MEAN, STD

NIT, WARM = 500, 30
STAGES = ['slic', 'features', 'graph', 'h2d_forward_d2h', 'reconstruct']

def smi():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw', '--format=csv,noheader'], text=True).strip()
    except Exception as e: return str(e)

def q(a):
    a = np.asarray(a)
    return dict(median=float(np.median(a)), q1=float(np.percentile(a, 25)), q3=float(np.percentile(a, 75)),
                iqr=float(np.percentile(a, 75) - np.percentile(a, 25)), p95=float(np.percentile(a, 95)))

def analytic_gmacs(cfg, N, E):
    m = cfg['model']; d = m.get('hidden_dim', 64); t = m['type']
    L = m.get('num_layers', 8 if t == 'deep_gnn' else 2) if t != 'hybrid' else m.get('gnn_layers', 2)
    macs, din = 0, m['in_dim']
    for _ in range(L):
        macs += N * 2 * din * d + E * din; din = d  # SAGE linears + mean aggregation
    if t == 'hybrid':
        if m.get('use_pe', True): macs += N * (d + m.get('pe_dim', 32)) * d
        macs += m.get('transformer_layers', 2) * (4 * N * d * d + 2 * N * N * d + 8 * N * d * d)
    return (macs + N * d * 2) / 1e9

def dc_analytic_gmacs(model, H, W):
    # Dense conv MACs including padding, all side heads and full-resolution fusion.
    h, w = H + (-H % 32), W + (-W % 32); macs = 0
    for block, side in zip(model.blocks, model.side):
        for layer in block:
            if isinstance(layer, nn.MaxPool2d): h //= 2; w //= 2
            elif isinstance(layer, nn.Conv2d):
                macs += h * w * layer.out_channels * (layer.in_channels // layer.groups) * np.prod(layer.kernel_size)
        macs += h * w * side.in_channels
    return float((macs + H * W * 5) / 1e9)

@torch.no_grad()
def graph_once(img, model, mt, tau, dev, image_stem):
    torch.cuda.synchronize(dev); t = [time.perf_counter()]
    seg = compute_superpixels(img, 300, 10.0, 1.0); t.append(time.perf_counter())
    assert seg.shape == img.shape[:2], f'image/segments shape mismatch on {image_stem}: {img.shape[:2]} != {seg.shape}'
    f, c = extract_node_features(img, seg); t.append(time.perf_counter())
    ei, ew = symmetrize_graph(*build_pirm_graph(f, k=8)); t.append(time.perf_counter())
    x = torch.from_numpy(f).to(dev); e = torch.from_numpy(ei).long().to(dev); p = torch.from_numpy(c).to(dev)
    o = model(x, e, p) if mt == 'hybrid' else model(x, e)
    pr = torch.softmax(o, 1)[:, 1].cpu().numpy()
    torch.cuda.synchronize(dev); t.append(time.perf_counter())
    lab, inv = np.unique(seg, return_inverse=True)
    assert pr.shape == (len(lab),), f'node probabilities/segments mismatch on {image_stem}: {pr.shape} != {(len(lab),)}'
    mask = (pr >= np.float32(tau))[inv.reshape(seg.shape)]
    assert mask.shape == seg.shape, f'mask/segments shape mismatch on {image_stem}: {mask.shape} != {seg.shape}'
    t.append(time.perf_counter())
    return np.diff(t) * 1e3

def dc_input(img, dev):
    return ((torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1) - MEAN) / STD)[None].to(dev)

@torch.no_grad()
def dc_once(img, model, tau, dev):
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    lg = model(dc_input(img, dev))[0]
    mask = (torch.sigmoid(lg)[0, 0] >= np.float32(tau)).cpu().numpy()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) * 1e3

@torch.no_grad()
def model_only(fn, prepare, n_inputs, dev):
    ts = []
    for it in range(WARM + NIT):
        inp = prepare(it % n_inputs)  # Input preparation/transfer excluded from model-only timing.
        torch.cuda.synchronize(dev); a = time.perf_counter(); fn(inp); torch.cuda.synchronize(dev)
        elapsed = (time.perf_counter() - a) * 1e3
        if it >= WARM: ts.append(elapsed)
    return q(ts)

@torch.no_grad()
def fvcore_count(model, inputs):
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, inputs)
        fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
        value = fa.total() / 1e9
        return value, dict(fa.unsupported_ops())
    except Exception as ex: return None, {'error': str(ex)}

class W1(nn.Module):
    def __init__(self, model): super().__init__(); self.m = model
    def forward(self, x): return self.m(x)[0]

def main():
    if not torch.cuda.is_available(): raise SystemExit('CUDA required for the specified efficiency protocol.')
    dev = torch.device('cuda')
    # FP32, including matrix multiplications; no autocast or TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, off, stems = load_labels()
    IMGS = [read_rgb(find(os.path.join(DC, 'test_img'), s)) for s in stems]
    if len(IMGS) != 237: raise ValueError(f'Expected 237 holdout images, found {len(IMGS)}')
    ds = CrackGraphDataset(rel('data/graphs/test'))
    if [stem(f) for f in ds._file_list] != stems: raise ValueError('Test graph/image order mismatch')
    graphs = [ds[i] for i in range(len(ds))]
    R = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cpu=platform.processor(),
             warmup=WARM, timed_iterations=NIT, n_images=len(IMGS), dtype='float32',
             telemetry={'start': smi()}, models={})
    try:
        import psutil
        freq = psutil.cpu_freq(); R['cpu_freq_mhz'] = freq.current if freq else None
    except ImportError: pass
    N_mean = float(np.mean(np.diff(off))); E_mean = float(np.mean([g.edge_index.shape[1] for g in graphs]))
    for key in ['M1_focal', 'M2_focal', 'M3_focal', 'M4_focal']:
        cfgp, _, root = GRAPH_CELLS[key]; cfg = load_config(rel(cfgp)); mt = cfg['model']['type']
        model = build_model(cfg).to(dev)
        model.load_state_dict(torch.load(os.path.join(BASE, root, 'seed_42', 'best_model.pt'), map_location=dev, weights_only=True))
        model.eval(); tau = load_probs(key, 42)[1]; stage = []
        R['telemetry'][f'start_{key}'] = smi()
        for it in range(WARM + NIT):
            s = graph_once(IMGS[it % len(IMGS)], model, mt, tau, dev, stems[it % len(IMGS)])
            if it >= WARM: stage.append(s)
            if it == WARM + NIT // 2: R['telemetry'][f'mid_{key}'] = smi()
        stage = np.array(stage); e2e = stage.sum(1)
        # Compare both counters on exactly the same real preprocessed graph.
        seg = compute_superpixels(IMGS[0], 300, 10.0, 1.0)
        assert seg.shape == IMGS[0].shape[:2], f'image/segments shape mismatch on {stems[0]}: {IMGS[0].shape[:2]} != {seg.shape}'
        ft, ce = extract_node_features(IMGS[0], seg); ei, _ = symmetrize_graph(*build_pirm_graph(ft, 8))
        X, EI, P = torch.from_numpy(ft).to(dev), torch.from_numpy(ei).long().to(dev), torch.from_numpy(ce).to(dev)
        fv, unsupported = fvcore_count(model, (X, EI, P) if mt == 'hybrid' else (X, EI))
        def prepare(i):
            g = graphs[i]
            return (g.x.to(dev), g.edge_index.to(dev), g.pos.to(dev)) if mt == 'hybrid' else (g.x.to(dev), g.edge_index.to(dev))
        mo = model_only(lambda inp: model(*inp), prepare, len(IMGS), dev)
        R['models'][key] = dict(params=sum(p.numel() for p in model.parameters()), gmacs_fvcore=fv,
            gmacs_analytic=analytic_gmacs(cfg, len(ft), ei.shape[1]),
            gmacs_analytic_mean_graph=analytic_gmacs(cfg, N_mean, E_mean),
            complexity_input=dict(n_nodes=len(ft), directed_edges=ei.shape[1]), fvcore_unsupported=unsupported,
            model_only_ms=mo, e2e_ms=q(e2e), e2e_fps=1000 / float(np.median(e2e)), model_fps=1000 / mo['median'],
            e2e_mean_ms=float(e2e.mean()), stage_mean_ms=dict(zip(STAGES, stage.mean(0).tolist())),
            stage_median_ms=dict(zip(STAGES, np.median(stage, 0).tolist())))
        R['telemetry'][f'end_{key}'] = smi()
        print(key, R['models'][key]['e2e_ms']['median'], flush=True)
        del model
    ck = os.path.join(OUT, 'runs', 'DeepCrack', 'seed_42', 'best_model.pt')
    dc = DeepCrackHED(pretrained=not os.path.exists(ck)).to(dev)
    if os.path.exists(ck): dc.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
    dc.eval(); H, W = IMGS[0].shape[:2]; xin = dc_input(IMGS[0], dev)
    dfv, unsupported = fvcore_count(W1(dc), xin)
    tau = load_probs('DeepCrack', 42)[1] if os.path.exists(f'{OUT}/probs/DeepCrack/seed_42.npz') else 0.5
    e2e = []; R['telemetry']['start_DeepCrack'] = smi()
    for it in range(WARM + NIT):
        elapsed = dc_once(IMGS[it % len(IMGS)], dc, tau, dev)
        if it >= WARM: e2e.append(elapsed)
        if it == WARM + NIT // 2: R['telemetry']['mid_DeepCrack'] = smi()
    mo = model_only(lambda x: dc(x), lambda i: dc_input(IMGS[i], dev), len(IMGS), dev)
    R['models']['DeepCrack'] = dict(params=sum(p.numel() for p in dc.parameters()), gmacs_fvcore=dfv,
        gmacs_analytic=dc_analytic_gmacs(dc, H, W), fvcore_unsupported=unsupported, model_only_ms=mo,
        e2e_ms=q(e2e), e2e_fps=1000 / float(np.median(e2e)), model_fps=1000 / mo['median'], input_hw=[H, W],
        initialization='trained' if os.path.exists(ck) else 'ImageNet VGG', tau=tau)
    R['telemetry']['end'] = smi(); R['N_mean_nodes'] = N_mean; R['E_mean_edges'] = E_mean
    with open(f'{OUT}/efficiency.json', 'w') as f: json.dump(R, f, indent=2)

if __name__ == '__main__':
    main()
