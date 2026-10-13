"""Format stored experiment outputs; never run inference or statistical analysis."""
import argparse
import json
import os
import re
from pathlib import Path
from common import BASE, OUT, SEEDS, GRAPH_CELLS

PRIMARY = [f'M{i}_focal' for i in range(1, 5)] + ['DeepCrack']

def read(name):
    with open(os.path.join(OUT, name), encoding='utf-8') as f: return json.load(f)

def num(v):
    return 'unavailable' if v is None else f'{v:.4f}'

def ms(v): return f'{num(v["mean"])} ± {num(v["sd"])}'
def ci(v): return f'[{num(v[0])}, {num(v[1])}]'
def vector(v): return '[' + ', '.join(num(x) for x in v) + ']'
def cell(v): return str(v).replace('|', '\\|').replace('\n', '<br>')

def table(lines, headers, rows):
    lines.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
    lines.extend('| ' + ' | '.join(cell(c) for c in row) + ' |' for row in rows)
    lines.append('')

def manuscript_claims(path):
    """Extract candidate sentences only; semantic claim review remains author work."""
    text = Path(path).read_text(encoding='utf-8')
    parts = re.findall(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.S)
    parts += re.findall(r'\\section\*?\{[^}]*[Cc]onclusions?[^}]*\}(.*?)(?=\\section|\\bibliography|\\begin\{thebibliography\}|\\end\{document\}|\Z)', text, re.S)
    claims = []
    for part in parts:
        part = re.sub(r'(?m)(?<!\\)%.*$', '', part)
        for sentence in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+', ' ', part).strip()):
            if re.search(r'\d', sentence): claims.append(sentence)
    return claims

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--paper', help='Optional manuscript .tex for claim inventory (read only)')
    args = ap.parse_args()
    data, analysis, topology, efficiency, check = [read(n) for n in
        ['dataset_stats.json', 'analysis_main.json', 'topology.json', 'efficiency.json', 'predict_check.json']]
    expected = list(GRAPH_CELLS) + ['DeepCrack']
    if any(k not in analysis['overall'] for k in expected):
        raise SystemExit('Final tables require all nine models with five seeds; rerun analysis after training.')
    if any(f'{k}/{s}' not in check or not check[f'{k}/{s}']['ok'] for k in GRAPH_CELLS for s in SEEDS):
        raise SystemExit('All 40 graph runs must pass predict_all.py before final tables.')
    if any(k not in topology['models'] or k not in efficiency['models'] for k in PRIMARY):
        raise SystemExit('Final tables require topology and efficiency for every primary model.')
    if len(analysis['loss_grid']) != 4 or len(analysis['pairs']) != 5:
        raise SystemExit('Final tables require all pre-specified loss/pairwise comparisons.')
    runs = {}
    for key in expected:
        root = os.path.join(BASE, GRAPH_CELLS[key][2]) if key in GRAPH_CELLS else os.path.join(OUT, 'runs', key)
        runs[key] = []
        for seed in SEEDS:
            with open(os.path.join(root, f'seed_{seed}', 'result.json'), encoding='utf-8') as f: runs[key].append(json.load(f))
    lines = ['# Paper v2 numbers', '', 'Primary comparison: focal loss for M1–M4 plus DeepCrack-style (re-implemented).',
             'Weighted CE is secondary. Metrics are node-level; SD uses ddof=1 over the five fixed seeds.',
             'Test PR-AUC is average precision. Morphology bootstrap resamples images, jointly for all models.',
             f'Graph prediction verification: {len(check)} runs; all required runs passed within 0.0020 F1.', '']
    lines += ['## 1. Data & splits', '']
    table(lines, ['Seed', 'Train', 'Val', 'Unused', 'Train nodes', 'Crack nodes', 'Crack fraction', 'Focal α [bg, crack]', 'WCE weights'],
          [[s, v['n_train'], v['n_val'], v['n_unused'], v['train_nodes'], v['train_crack_nodes'], num(v['train_crack_frac']), vector(v['focal_alpha']), vector(v['wce_weights'])]
           for s in SEEDS for v in [data['seeds'][str(s)]]])
    t = data['test']
    table(lines, ['Test images', 'Nodes', 'Crack-node fraction', 'Mean nodes', 'Mean directed edges', 'Image sizes [H,W]', 'Mean crack-pixel fraction'],
          [[t['n_images'], t['n_nodes'], num(t['crack_node_frac']), num(t['mean_nodes']), num(t['mean_directed_edges']), t['image_sizes'], num(t['crack_pixel_frac'])]])
    lines += ['## 2. Main table', '']
    metrics = ['f1', 'iou', 'precision', 'recall', 'roc_auc', 'pr_auc']
    table(lines, ['Model', 'F1', 'IoU', 'Precision', 'Recall', 'ROC-AUC', 'PR-AUC (AP)'],
          [[k] + [ms(analysis['overall'][k][m]) for m in metrics] for k in PRIMARY])
    lines += ['## 3. Loss grid (secondary)', '', 'Paired seed-level t-tests; Holm correction over four model comparisons.', '']
    table(lines, ['Loss'] + [f'M{i}' for i in range(1, 5)],
          [[loss] + [ms(analysis['loss_grid'][f'M{i}'][loss]) for i in range(1, 5)] for loss in ['wce', 'focal']])
    table(lines, ['Model', 'Focal − WCE F1', 'Paired p', 'Holm p'],
          [[k, num(v['diff_mean']), num(v['p']), num(v['p_holm'])] for k, v in analysis['loss_grid'].items()])
    lines += ['## 4. Pairwise family', '', 'Differences use seed-averaged per-image F1; Holm adjusts Wilcoxon p over the five pre-specified pairs.', '']
    table(lines, ['A − B', 'Mean image F1 difference', 'd_z', 'Image t p', 'Wilcoxon p', 'Wilcoxon Holm p', 'Seed t p'],
          [[f'{v["a"]} − {v["b"]}'] + [num(v[m]) for m in ['mean_diff_img', 'cohen_dz', 'p_t_img', 'p_wilcoxon_img', 'p_wilcoxon_holm', 'p_seed_t']] for v in analysis['pairs']])
    lines += ['## 5. Morphology', '', 'Thin first: area fraction <0.025; otherwise Long if bbox aspect ratio ≥2; otherwise Branched.', '']
    for bucket, v in analysis['buckets'].items():
        lines += [f'### {bucket} (N={v["n"]})', '']
        table(lines, ['Model', 'F1', 'Image-bootstrap 95% CI'], [[k, num(m['f1']), ci(m['ci95'])] for k, m in v['models'].items()])
        table(lines, ['Paired difference', 'Bootstrap mean', '95% CI', 'Bootstrap p'],
              [[k, num(d['mean']), ci(d['ci95']), num(d['p_boot'])] for k, d in v['diffs'].items()])
    lines += ['## 6. Topology', '', f'clDice images: {topology["n_images_cldice"]}; retained GT components: {topology["n_gt_components"]:.0f}.',
              'All predictions reconstructed from the identical superpixel nodes. Empty GT skeletons are excluded.',
              'Connectivity uses 8-connected GT components of ≥20 pixels; one predicted component must cover ≥50%.', '']
    table(lines, ['Model', 'clDice seed mean ± sd', 'Image-bootstrap CI', 'Connectivity micro mean ± sd', 'Connectivity macro mean ± sd'],
          [[k, f'{num(v["cldice_seed_mean"])} ± {num(v["cldice_seed_sd"])}', ci(v['cldice_img_ci95']),
            f'{num(v["conn_micro_seed_mean"])} ± {num(v["conn_micro_seed_sd"])}', f'{num(v["conn_macro_seed_mean"])} ± {num(v["conn_macro_seed_sd"])}'] for k, v in topology['models'].items()])
    table(lines, ['clDice paired difference', 'Bootstrap mean', '95% CI'], [[k, num(v['mean']), ci(v['ci95'])] for k, v in topology['diffs'].items()])
    lines += ['## 7. Efficiency', '', f'Device: {efficiency["device"]}; CPU: {efficiency["cpu"]}; PyTorch: {efficiency["torch"]}.',
              'Protocol: 237 holdout RGB arrays preloaded in RAM, FP32, no_grad, 30 warm-up + 500 timed cyclic inputs, GPU synchronization.',
              'Model-only excludes input preparation/transfer. End-to-end starts with RGB in RAM and ends with a full-resolution binary mask.',
              'Both complexity columns are GMACs. Analytic counts use dense linears/attention, SAGE aggregation, or dense CNN convolutions.',
              'fvcore counts one multiply-add as one op; unsupported ops and normalization/non-MAC operations explain gaps.', '']
    table(lines, ['Model', 'Parameters', 'GMACs fvcore', 'GMACs analytic', 'Model ms median', 'Model FPS', 'E2E ms median', 'E2E IQR', 'E2E FPS'],
          [[k, v['params'], num(v['gmacs_fvcore']), num(v.get('gmacs_analytic')), num(v['model_only_ms']['median']), num(v['model_fps']),
            num(v['e2e_ms']['median']), num(v['e2e_ms']['iqr']), num(v['e2e_fps'])] for k, v in efficiency['models'].items()])
    for k, v in efficiency['models'].items():
        lines += [f'- {k}: complexity input {v.get("complexity_input", v.get("input_hw"))}; fvcore unsupported: `{v.get("fvcore_unsupported", {})}`.']
        if 'gmacs_analytic_mean_graph' in v: lines += [f'  Analytic mean-graph GMACs: {num(v["gmacs_analytic_mean_graph"])}.']
    lines += ['', 'Stage medians do not generally add to the median total. Stage means add to mean E2E; each iteration uses an exact stage sum.', '']
    stages = ['slic', 'features', 'graph', 'h2d_forward_d2h', 'reconstruct']
    for name in ['stage_median_ms', 'stage_mean_ms']:
        table(lines, ['Model (' + name + ')'] + stages + ['E2E mean ms'],
              [[k] + [num(v[name][s]) for s in stages] + [num(v.get('e2e_mean_ms'))] for k, v in efficiency['models'].items() if name in v])
    lines += ['Telemetry (SM clock, memory clock, GPU temperature, power):', '']
    table(lines, ['Sample', 'nvidia-smi'], [[k, v] for k, v in efficiency['telemetry'].items()])
    lines += ['## 8. Per-run τ*', '', 'All thresholds selected on validation nodes only, 0.10–0.90 in 0.02 steps.', '']
    table(lines, ['Model', 'Seed', 'τ*', 'DeepCrack epochs', 'DeepCrack parameters'],
          [[k, s, num(r['best_tau']), r.get('epochs', '—'), r.get('params', '—')] for k, rs in runs.items() for s, r in zip(SEEDS, rs)])
    table(lines, ['Model', 'Minimum τ*', 'Maximum τ*'], [[k, num(min(r['best_tau'] for r in rs)), num(max(r['best_tau'] for r in rs))] for k, rs in runs.items()])
    lines += ['## 9. Claims to re-check', '', 'These checks compare stored summaries; they do not replace manuscript review.', '']
    for a, b, claim in [('DeepCrack', 'M4_focal', 'DeepCrack ahead of M4'), ('M4_focal', 'M2_focal', 'M4 ahead of M2')]:
        av, bv = analysis['overall'][a]['f1']['mean'], analysis['overall'][b]['f1']['mean']
        lines += [f'- {claim}: {"supported numerically" if av > bv else "contradicted numerically"}; {num(av)} vs {num(bv)}. See pairwise tests for significance.']
    for k, v in analysis['loss_grid'].items():
        lines += [f'- Focal improves {k}: {"supported numerically" if v["diff_mean"] > 0 else "contradicted numerically"}; ΔF1={num(v["diff_mean"])}; Holm p={num(v["p_holm"])}.']
    e = efficiency['models']; ratio = e['DeepCrack']['params'] / e['M4_focal']['params']
    latency = e['M4_focal']['e2e_ms']['median'] / e['DeepCrack']['e2e_ms']['median']
    lines += [f'- Historical 126× parameter claim: actual DeepCrack/M4 parameter ratio = {num(ratio)}×.',
              f'- Historical ≈24× slower claim: actual M4/DeepCrack median end-to-end ratio = {num(latency)}×.',
              '- Topology superiority claims must be checked against section 6 using the revised component/empty-GT definitions.']
    if args.paper:
        claims = manuscript_claims(args.paper)
        lines += ['', 'Quantitative abstract/conclusion sentences (author must map each to tables and label support/contradiction):', '']
        lines += [f'- {c}' for c in claims] or ['- No quantitative sentences found; inspect manuscript section markup.']
    else:
        lines += ['- Exact abstract/conclusion text unavailable. Rerun with --paper /path/to/manuscript.tex to inventory every quantitative sentence; author review remains required.']
    text = '\n'.join(lines) + '\n'
    Path(OUT, 'PAPER_NUMBERS.md').write_text(text, encoding='utf-8')
    print(text)

if __name__ == '__main__':
    main()
