"""Audit saved experiment artifacts and append reporting detail; no inference.

This command reads saved results/configurations/checkpoints only. It never reads
dataset images, labels, or prediction NPZ files and cannot evaluate test again.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def audit(directory):
    directory=Path(directory).resolve()
    if (directory/'artifact_audit.json').exists(): raise FileExistsError('Audit already exists')
    lock=read(directory/'locked_configs.json'); final=read(directory/'final_results.json')
    consumed=read(directory/'test_consumed.json')
    lock_hash=sha(directory/'locked_configs.json')
    assert lock_hash==final['lock_sha256']==consumed['lock_sha256']==read(directory/'locked_configs.sha256.json')['sha256']
    assert lock['protocol_sha256']==sha(directory/'protocol.json')
    assert lock['original_processing_sha256']==sha(directory/'original_processing_locked.json')
    assert final['original_artifacts_unchanged']
    assert all(value['matched'] for value in final['reproduction'].values())
    assert consumed['retry_allowed'] is False
    groups={name:[r for r in final['cases'] if r['model']==name] for name in dict.fromkeys(r['model'] for r in final['cases'])}
    assert set(groups)=={'M2','M4','M2+M4','DeepCrack raw','DeepCrack processed'}
    expected={42,1,2} if lock['comparator_mode']=='three_seeds' else {42}
    for name,rows in groups.items():
        assert {r['seed'] for r in rows}==({42,1,2} if name in ('M4','M2+M4') else expected)
        for row in rows:
            score=row['test']
            assert score['n_total']==992
            assert score['n_total']==score['n_preserved']+score['n_missed']+score['n_fragmented']
            assert abs(score['strict_cr']-score['n_preserved']/score['n_total'])<1e-12
            assert len(score['cldice_per_image'])==237
            assert [b['bin'] for b in score['size_bins']]==['[0,50)','[50,200)','[200,inf)']
            assert [b['total'] for b in score['size_bins']]==[167,274,551]
            sweep=[s['strict_cr'] for s in score['overlap_sweep']]
            assert sweep==sorted(sweep,reverse=True)
            for key in ('strict_cr','cldice','pixel_f1'):
                assert abs(row['val_minus_test_pp'][key]-100*(row['validation'][key]-score[key]))<1e-10
    locked_original=read(directory/'original_processing_locked.json')['models']
    for case in lock['cases']:
        if case['model']=='M2': assert case['recipe']==locked_original['M2']['recipe']
        if case['model']=='DeepCrack processed': assert case['recipe']==locked_original['DeepCrack']['recipe']
        if case['model']=='DeepCrack raw': assert case['recipe']==dict(low=.36,high=.36,min_area=0,max_gap=0)
        for descriptor in [case['descriptor']]+([case['m2_descriptor']] if 'm2_descriptor' in case else []):
            assert sha(descriptor['checkpoint'])==descriptor['checkpoint_sha256']
    manifest=read(directory/'checkpoint_manifest.json')
    checkpoint_count=0
    for run in manifest:
        assert Path(run['config']).is_file()
        assert Path(run['best_checkpoint']).is_file()
        for checkpoint in run['checkpoints']:
            assert sha(checkpoint['path'])==checkpoint['sha256']
            checkpoint_count+=1
    def mean(name,key,kind='test'):
        return statistics.mean(r[kind][key] for r in groups[name])
    negatives=[]
    for path in sorted((directory/'validation').glob('*.json')):
        result=read(path)
        declines=[f'{key} {100*(result["metrics"][key]-result["baseline"][key]):+.3f} pp'
                  for key in ('strict_cr','cldice','pixel_f1')
                  if result['metrics'][key]<result['baseline'][key]-1e-12]
        if declines:
            negatives.append(f'Validation {path.stem}: accepted/retained processing has these declines versus its stated baseline: '+', '.join(declines)+'.')
    for name,historical in [('M2','M2'),('M4','M4'),('DeepCrack processed','DeepCrack')]:
        old=final['reproduction'][historical]['metrics']
        for key in ('strict_cr','cldice','pixel_f1'):
            delta=mean(name,key)-old[key]
            if delta<=0: negatives.append(f'{name} final mean {key} did not improve over the historical seed-42 control ({100*delta:+.3f} pp).')
        for index,label in enumerate(('[0,50)','[50,200)','[200,inf)')):
            delta=statistics.mean(r['test']['size_bins'][index]['strict_cr'] for r in groups[name])-old['size_bins'][index]['strict_cr']
            if delta<=0: negatives.append(f'{name} final mean connectivity in {label} did not improve over the historical seed-42 control ({100*delta:+.3f} pp).')
    for comparator in ('DeepCrack raw','DeepCrack processed'):
        for key in ('strict_cr','cldice','pixel_f1'):
            delta=mean('M4',key)-mean(comparator,key)
            if delta<=0: negatives.append(f'M4 final mean {key} did not exceed {comparator} ({100*delta:+.3f} pp).')
    for key in ('strict_cr','cldice','pixel_f1'):
        delta=mean('DeepCrack processed',key)-mean('DeepCrack raw',key)
        if delta<=0: negatives.append(f'DeepCrack processing did not improve final mean {key} ({100*delta:+.3f} pp).')
    dc_raw={r['seed']:r for r in groups['DeepCrack raw']}
    transfer_guard_failures=[]
    for row in groups['DeepCrack processed']:
        raw=dc_raw[row['seed']]
        if row['validation']['cldice']<raw['validation']['cldice']-1e-12:
            transfer_guard_failures.append(row['seed'])
            negatives.append(f'DeepCrack seed {row["seed"]}: transferred processing lowers validation clDice relative to that seed\'s raw output; the recipe was guarded and selected on seed 42 only.')
        for key in ('strict_cr','cldice','pixel_f1'):
            delta=row['test'][key]-raw['test'][key]
            if delta<0: negatives.append(f'DeepCrack seed {row["seed"]}: processing lowers test {key} versus that seed\'s raw output ({100*delta:+.3f} pp).')
    for case in lock['cases']:
        if case.get('alpha') in (0.,1.):
            negatives.append(f'M2+M4 seed {case["seed"]}: selected M2 weight {case["alpha"]}; this is a single-constituent control, not an averaging gain. Reported ensemble cost conservatively includes both networks.')
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    actual_invocations=[dict(argv=e['argv'],cwd=e['cwd']) for e in events if e['kind']=='invocation']
    started=read(directory/'protocol_started.json')['unix_time']
    now=datetime.now(timezone.utc)
    audit_record=dict(time=now.isoformat(),passed=True,lock_sha256=lock_hash,cases=len(final['cases']),
                      new_runs=len(manifest),new_checkpoint_files=checkpoint_count,negative_test_results=negatives,
                      transferred_deepcrack_validation_guard_failures=transfer_guard_failures,
                      actual_invocations=actual_invocations,total_elapsed_seconds_including_implementation=now.timestamp()-started,
                      auditor_sha256=sha(__file__),additional_test_evaluations=0)
    with (directory/'artifact_audit.json').open('x',encoding='utf-8') as f: json.dump(audit_record,f,indent=2,allow_nan=False)
    lines=['','## Loss and SLIC before/after against the incumbent','',
           '| Candidate | Before CR | Candidate CR | Before clDice | Candidate clDice | Before F1 | Candidate F1 | Replaced incumbent |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for path in sorted((directory/'selection').glob('*.json')):
        selection=read(path); before=selection['incumbent_before']; after=selection['validation']['metrics']
        lines.append(f'| {selection["cell"]} | {before["strict_cr"]:.6f} | {after["strict_cr"]:.6f} | {before["cldice"]:.6f} | {after["cldice"]:.6f} | {before["pixel_f1"]:.6f} | {after["pixel_f1"]:.6f} | {selection["accepted"]} |')
    lines+=['','## Per-seed final results','',
           '| Model | Seed | Test strict CR | Test clDice | Test pixel F1 | Val−test CR pp | Val−test clDice pp | Val−test F1 pp |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in final['cases']:
        lines.append(f'| {row["model"]} | {row["seed"]} | '+
                     ' | '.join(f'{row["test"][k]:.6f}' for k in ('strict_cr','cldice','pixel_f1'))+' | '+
                     ' | '.join(f'{row["val_minus_test_pp"][k]:+.3f}' for k in ('strict_cr','cldice','pixel_f1'))+' |')
    lines+=['','## Locked final operating points','',
            '| Model | Seed | Low | High | Minimum area px | Maximum gap px | M2 weight | Calibration |',
            '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for case in lock['cases']:
        recipe=case['recipe']
        lines.append(f'| {case["model"]} | {case["seed"]} | {recipe["low"]} | {recipe["high"]} | {recipe["min_area"]} | {recipe["max_gap"]} | {case.get("alpha","—")} | {case["calibration"]} |')
    lines+=['','## Additional negative test results','']+(['- '+n for n in negatives] or ['- None.'])
    lines+=['','## Complexity counter detail','',
            '| Model | Actual graph nodes | Actual directed edges | GFLOPs fvcore | GFLOPs analytic graph estimate |',
            '| --- | --- | --- | --- | --- |']
    for name,rows in groups.items():
        cost=rows[0]['complexity']
        if 'constituent_costs' in cost:
            parts=cost['constituent_costs']
            analytic=sum(p['gflops_analytic'] for p in parts)
            nodes='+'.join(str(p['nodes']) for p in parts)
            edges='+'.join(str(p['directed_edges']) for p in parts)
        else:
            analytic=cost.get('gflops_analytic')
            nodes=cost['nodes'] if cost['nodes'] is not None else 'CNN 256×256'
            edges=cost['directed_edges'] if cost['directed_edges'] is not None else '—'
        lines.append(f'| {name} | {nodes} | {edges} | {cost["gflops_fvcore"]:.9f} | {f"{analytic:.9f}" if analytic is not None else "not applicable"} |')
    lines+=['','The graph estimate includes the attention matrix multiplications analytically. fvcore may omit the fused native attention operator; unsupported-operation dictionaries are retained in final_results.json. Both columns use 2×MACs. These are model costs; probability mixing and CPU morphology are not included.','']
    names=['M2','M4','M2+M4','DeepCrack raw','DeepCrack processed']
    def formatted(values):
        return f'{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}' if len(values)>1 else f'{values[0]:.6f} (n=1)'
    lines+=['','## Overlap sweep in PITCH_RERUN layout','',
            '| Overlap | '+' | '.join(names)+' |','| --- | '+' | '.join('---' for _ in names)+' |']
    for index,threshold in enumerate((.3,.4,.5,.6,.7)):
        lines.append(f'| {threshold} | '+' | '.join(formatted([r['test']['overlap_sweep'][index]['strict_cr'] for r in groups[name]]) for name in names)+' |')
    lines+=['','## Artifact verification and execution','',
            f'Artifact audit passed: {len(final["cases"])} final cases, {len(manifest)} new runs, {checkpoint_count} immutable checkpoint files. Original results are unchanged. This audit reads saved results only and performs no further test evaluation.', '',
            'Standard deviations describe training-seed variability on fixed images; they do not measure uncertainty over independent sources or new data splits.', '',
            'The folder holdout is preserved for PITCH_RERUN comparability; it is not a source-disjoint split. Package versions are saved in environment_packages.json.', '',
            f'Total elapsed time including implementation and this audit: {(now.timestamp()-started)/60:.2f} minutes.', '',
            'The master driver dispatched training in process. The per-run commands above are equivalent reproduction commands; the exact master invocation actually used is:', '', '```json',json.dumps(actual_invocations,indent=2),'```','']
    lines+=['The nested legacy architecture config retains its original output_dir for provenance. The isolated runner ignores that field and writes exclusively inside this experiment\'s runs directory. Use the recorded runner commands rather than passing that nested config to a legacy trainer.','']
    with (directory/'RESULTS.md').open('a',encoding='utf-8') as f: f.write('\n'.join(lines))
    print(json.dumps(audit_record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    audit(parser.parse_args().output)
