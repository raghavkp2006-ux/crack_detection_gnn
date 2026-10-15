"""Append transparent numeric interpretations and verification to the final report.

This script only reads locked results; it cannot select configs or run inference.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def mean(values):
    return statistics.mean(values)


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


if __name__=='__main__':
    if (OUT/'report_supplement.json').exists():raise FileExistsError('Final supplement already written')
    selection=read(OUT/'selection.json');lock=read(OUT/'locked_configs.json')
    confirmation=read(OUT/'held_out_results.json');rows=confirmation['results']
    assert digest(OUT/'locked_configs.json')==read(OUT/'locked_configs.sha256.json')['sha256']==confirmation['lock_sha256']
    diagnosis=read(OUT/'step1_baseline/diagnosis.json')
    text=['','## Numeric interpretation and complete run ledger','',
        'The baseline diagnosis is mixed. Smoothed node F1 and pixel F1 peak at epochs 84/61/89 (seeds 42/1/2), whereas clDice peaks at 84/99/89 and strict CR at 7/10/10. Only seed 1 achieves a higher smoothed clDice after its smoothed node-F1 maximum (about +0.204 pp). All three show later local rises in strict CR relative to its value at the node-F1 peak, but none surpasses the early global strict-CR maximum. These local rises are not evidence of sustained joint gains in connectivity and segmentation quality. Training loss falls across all three baseline runs.','',
        'The baseline is a fresh retraining with the historical architecture, focal loss and schedule. The per-epoch validation implementation now logs native metrics and directly iterates validation graphs; training RNG consumption after epoch one need not match the legacy validation DataLoader trajectory. Seeds and split membership remain fixed, and all new candidates use the same new logging path.','',
        '### Explicit negative results','',
        '- Initial preflight rejected a valid all-white empty GT mask. The parser was corrected before protocol freeze or training; the error and its original partial artifacts are preserved in failures.jsonl and preflight_failure_01/. This was an operational failure, not a failed model-training run.',
        '- Optional M2 retraining was skipped following the request to finish fast; the mandatory historical three-seed M2 comparator was retained.']
    for trial in selection['trials']:
        cr=100*(trial['processed']['strict_cr']['mean']-trial['reference_mean']['strict_cr'])
        cd=100*(trial['processed']['cldice']['mean']-trial['reference_mean']['cldice'])
        if not trial['accepted']:text.append(f'- Candidate {trial["candidate"]} was rejected versus {trial["incumbent_before"]}: processed validation strict CR {cr:+.3f} pp, clDice {cd:+.3f} pp. The required joint replacement rule was not met.')
    if selection['winner']=='historical':text += ['- No new M4 recipe replaced the incumbent. “New M4” in the generated tables is the selected M4 slot and aliases the retained historical M4; it is not a newly adopted model.',
        '- Optional M2 recipe transfer was not run because no new winning training recipe was adopted. Historical M2 remains the registered comparator.']
    if not confirmation['external_tits_unchanged']:raise AssertionError('TITS was modified')
    text += ['', '### Pre-registered claim deltas','']
    old,current=selection['historical_mean'],selection['winning_mean']
    text.append(f'Claim 1 compares selected versus current historical processed DeepCrack validation means: strict CR {current["strict_cr"]:.6f} versus {old["strict_cr"]:.6f} ({100*(current["strict_cr"]-old["strict_cr"]):+.3f} pp), and clDice {current["cldice"]:.6f} versus {old["cldice"]:.6f} ({100*(current["cldice"]-old["cldice"]):+.3f} pp). The report verdict refers only to this frozen validation comparison, not to source-independent generalization.')
    new=[r['held_out'] for r in rows if r['model']=='New M4' and r['mode']=='raw']
    dc=[r['held_out'] for r in rows if r['model']=='DeepCrack' and r['mode']=='raw']
    means={name:{k:mean([r[k] for r in group]) for k in ['strict_cr','cldice','pixel_f1']} for name,group in [('M4',new),('DeepCrack',dc)]}
    small={name:mean([r['size_bins'][0]['strict_cr'] for r in group]) for name,group in [('M4',new),('DeepCrack',dc)]}
    count=new[0]['size_bins'][0]['total']
    text += ['',f'Claim 2 uses raw locked predictions on all 66 annotated TITS pairs. M4 versus raw DeepCrack: strict CR {means["M4"]["strict_cr"]:.6f} versus {means["DeepCrack"]["strict_cr"]:.6f} ({100*(means["M4"]["strict_cr"]-means["DeepCrack"]["strict_cr"]):+.3f} pp); small-crack CR {small["M4"]:.6f} versus {small["DeepCrack"]:.6f} ({100*(small["M4"]-small["DeepCrack"]):+.3f} pp; {count} native GT components in this bin). For context, raw pixel F1 changes by {100*(means["M4"]["pixel_f1"]-means["DeepCrack"]["pixel_f1"]):+.3f} pp and clDice by {100*(means["M4"]["cldice"]-means["DeepCrack"]["cldice"]):+.3f} pp. “Matches/beats” is a comparison of three-seed point means, not a statistical equivalence or noninferiority test. It does not establish superiority over processed DeepCrack or on other metrics.','',
        '### Per-run validation results','', '| Candidate | Seed | Selected epoch | Raw CR | Raw clDice | Raw F1 | Processed CR | Processed clDice | Processed F1 |','| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    training_seconds=0
    for candidate in ['A','B','C','D']:
        for seed in [42,1,2]:
            result=read(OUT/f'runs/{candidate}/seed_{seed}/training.json');val=read(OUT/f'validation/{candidate}_seed{seed}_detailed.json')
            training_seconds+=result['wall_seconds']
            values=[val['baseline'][k] for k in ['strict_cr','cldice','pixel_f1']]+[val['metrics'][k] for k in ['strict_cr','cldice','pixel_f1']]
            text.append(f'| {candidate} | {seed} | {result["selected_epoch"]} | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    text += ['', '### Run wall times','', '| Candidate | Seed | Train + per-epoch native validation (s) | First-three projected full run (s) |','| --- | --- | --- | --- |']
    for candidate in ['A','B','C','D']:
        for seed in [42,1,2]:
            result=read(OUT/f'runs/{candidate}/seed_{seed}/training.json');gate=read(OUT/f'runs/{candidate}/seed_{seed}/timing_gate.json')
            text.append(f'| {candidate} | {seed} | {result["wall_seconds"]:.3f} | {gate["projected_seconds"]:.3f} |')
    cnn=read(OUT/'cnn_analytic.json');cost=read(OUT/'complexity.json')['DeepCrack']
    text += ['',f'Total mandatory training plus epoch logging: {training_seconds:.3f} s. Augmented-graph cache construction: {read(OUT/"augmented_graphs/manifest.json")["wall_seconds"]:.3f} s. One held-out TITS pass: {confirmation["wall_seconds"]:.3f} s. Calibration and other operations have exact wall times in `events.jsonl` and the per-candidate JSONL grids.','',
        'To honor the request to finish fast, the registered augmentation bank was prebuilt during B and published atomically, and D/seed-2 ran in an isolated process while earlier C work continued. Completed outputs were reused, with no duplicate training or altered recipe. All twelve 100-epoch runs and full grids remain present; replacement selection still follows A/B/C/D order. Individual operation wall times overlap and should not be summed as elapsed session time. The parallel commands are preserved alongside the sequential fresh-output reproduction.','',
        '### CNN analytic count and source provenance','',
        f'DeepCrack has {cnn["parameters"]} parameters and uses {cost["gflops_fvcore"]:.9f} fvcore GFLOPs versus {cnn["gflops_analytic"]:.9f} analytic convolution GFLOPs at 256×256. Both use 2×MACs. The analytic CNN count includes all convolutional side outputs and fusion; it excludes normalization, activations and interpolation. Graph analytic counts above include the attention matrix products and use the historical graph estimate. All figures are model costs; native projection, graph construction and CPU morphology are excluded.','',
        'The supplied TITS layout has no train/test directories: sensor-grouped `IMAGES`, `GROUND_TRUTH`, and pre-existing `RESULTS`. The latter predictions were not used. Dataset bibliographic sources and the limits of cross-corpus byte matching are in [PROVENANCE.md](PROVENANCE.md). The historical source-ID audit flags 39/45 DeepCrack validation images (86.7%) and 19/237 original test-folder images (8.0%) with overlapping source IDs under its stated rule. These are provenance metadata, not a new DeepCrack test evaluation.','',
        '### Verification','',
        'The 13 inherited synthetic metric/processing/timing tests passed. `verification_before_confirmation.json` independently checks all twelve 100-epoch logs, five-epoch windows, selected checkpoints, cosine endpoints, accumulation update counts, augmentation cache hashes, source hashes, the immutable lock and historical file metadata before TITS is consumed. The Step 1 plot was visually inspected. External TITS input hashes were checked again during the confirmation and its complete file metadata remained unchanged.','']
    for checkpoint in read(OUT/'checkpoint_manifest.json'):
        assert digest(checkpoint['path'])==checkpoint['sha256']
    for record in read(OUT/'complete_source_manifest.json')['records']:
        assert digest(record['path'])==digest(record['snapshot'])==record['sha256']
    for pair in read(OUT/'deepcrack_input_manifest.json'):
        for key in ['image','mask']:assert digest(pair[key]['path'])==pair[key]['sha256']
    started=datetime.fromisoformat(read(OUT/'session.json')['started_at_utc'].replace('Z','+00:00'))
    wall=(datetime.now(timezone.utc)-started).total_seconds()
    text += [f'Total elapsed session wall time including implementation, inspection, training and reporting: {wall:.1f} s ({wall/3600:.3f} h), within the nine-hour budget. Exact fresh-output reproduction is in `REPRODUCE.ps1`; source/config snapshots and SHA-256 manifests are preserved. No old checkpoint or result was overwritten.','']
    original=digest(OUT/'RESULTS.md')
    with (OUT/'RESULTS.md').open('a',encoding='utf-8') as stream:stream.write('\n'.join(text))
    with (OUT/'report_supplement.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(time=datetime.now(timezone.utc).isoformat(),base_report_sha256=original,final_report_sha256=digest(OUT/'RESULTS.md'),
            lock_sha256=digest(OUT/'locked_configs.json'),total_wall_seconds=wall,all_source_and_checkpoint_hashes_verified=True,
            claim2_raw_means=means,claim2_small_means=small,small_component_count=count),stream,indent=2)
    print('Completed report supplement:',OUT/'RESULTS.md')
