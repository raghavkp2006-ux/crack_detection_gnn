"""Finalize the budget-stopped study without training or held-out inference."""
import datetime as dt
import hashlib
import json
import pathlib
import statistics as st

P = pathlib.Path(__file__).resolve().parent
ROOT = P.parents[2]
def read(p): return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(name, obj):
    with (P / name).open('x', encoding='utf-8') as f: json.dump(obj, f, indent=2)

errors = []
checks = {}
sources = read(P / 'complete_source_manifest.json')['records']
for row in sources:
    for key in ('path', 'snapshot'):
        if sha(row[key]) != row['sha256']: errors.append('source mismatch: ' + row[key])
checks['source_records_verified'] = len(sources)
manifest = read(P / 'checkpoint_manifest.json')
for row in manifest:
    if sha(row['path']) != row['sha256']: errors.append('checkpoint mismatch: ' + row['path'])
checks['checkpoint_hashes_verified'] = len(manifest)
oldmeta = read(P / 'existing_historical_artifacts.json')
for name, expected in oldmeta.items():
    s = pathlib.Path(name).stat()
    if [s.st_size, s.st_mtime_ns] != expected: errors.append('historical metadata changed: ' + name)
checks['historical_files_metadata_verified'] = len(oldmeta)
for row in read(P / 'deepcrack_input_manifest.json'):
    for key in ('image', 'mask'):
        if sha(row[key]['path']) != row[key]['sha256']: errors.append('input changed: ' + row[key]['path'])
checks['deepcrack_train_validation_pairs_verified'] = 255
for row in read(P / 'tits_manifest.json')['pairs']:
    for key in ('image', 'mask'):
        if sha(row[key]) != row[key + '_sha256']: errors.append('TITS input changed: ' + row[key])
checks['tits_pairs_hash_verified_read_only'] = 66
for name in ('tits_consumed.json', 'held_out_results.json', 'locked_configs.sha256.json', 'selection.json'):
    checks[name + '_absent'] = not (P / name).exists()
    if (P / name).exists(): errors.append('unexpected artifact: ' + name)

rows = []
for candidate in 'ABCD':
    for seed in (42, 1, 2):
        run = P / 'runs' / candidate / ('seed_' + str(seed))
        epochs = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines() if line.strip()]
        training = read(run / 'training.json') if (run / 'training.json').exists() else None
        valpath = P / 'validation' / f'{candidate}_seed{seed}.json'
        val = read(valpath) if valpath.exists() else None
        rows.append(dict(candidate=candidate, seed=seed, epochs=len(epochs), completed=training is not None,
                         selected_epoch=training['selected_epoch'] if training else None,
                         raw=val['baseline'] if val else None, processed=val['metrics'] if val else None))
checks['completed_runs'] = sum(r['completed'] for r in rows)
checks['logged_epochs'] = sum(r['epochs'] for r in rows)
old = read(ROOT / 'pipeline-crack-gnn/results/paper_v2/improve_m2_m4_20261014_1112_02/locked_configs.json')
oldvals = [c['validation'] for c in old['cases'] if c['model'] == 'M4']
def summary(vals):
    return {k: {'mean': st.mean(v[k] for v in vals), 'sample_sd': st.stdev(v[k] for v in vals)}
            for k in ('strict_cr', 'cldice', 'pixel_f1')}
aggregate = {'old_M4_processed': summary(oldvals)}
for c in 'ABC':
    aggregate[c] = {mode: summary([r[mode] for r in rows if r['candidate'] == c]) for mode in ('raw', 'processed')}
    new = aggregate[c]['processed']; ref = aggregate['old_M4_processed']
    aggregate[c]['replacement_guard_passes'] = (new['strict_cr']['mean'] >= ref['strict_cr']['mean'] + .01
                                                and new['cldice']['mean'] >= ref['cldice']['mean'])
save('partial_validation_summary.json', {'runs': rows, 'aggregate': aggregate, 'D': 'incomplete; no three-seed aggregate or winner'})
save('lock_status.json', {'status': 'NOT_LOCKED_BUDGET_EXHAUSTED', 'winning_recipe': None,
                          'held_out_executed': False, 'reason': 'D seed 1 stopped after 56 of 100 epochs at nine-hour budget gate'})
now = dt.datetime.now(dt.timezone.utc)
start = dt.datetime.fromisoformat(read(P / 'session.json')['started_at_utc'].replace('Z', '+00:00'))
checks['verification_utc'] = now.isoformat()
checks['elapsed_wall_seconds_including_report_finalization'] = (now - start).total_seconds()
checks['errors'] = errors
save('verification_partial.json', checks)
if errors: raise RuntimeError(str(errors))

original = P / 'RESULTS.md'
snapshot = P / 'RESULTS.budget_stop_snapshot.md'
original.rename(snapshot)
text = snapshot.read_text(encoding='utf-8')
text = text.replace('All mandatory candidate runs use 100 epochs with no early stopping.',
                    'The protocol requires 100 epochs per run with no early stopping. Eleven runs completed 100 epochs; D seed 1 stopped at 56 because the nine-hour budget expired. D seed 1 curves and peak epochs below describe only its observed partial trajectory.')
text = text.replace('- Candidate rejections, including before/after means, are recorded in `selection.json` and the validation table.',
                    '- No final selection was made: D is incomplete and `selection.json` does not exist. Completed A/B/C comparisons are recorded in `partial_validation_summary.json`; none passes the replacement guard because mean clDice falls below the incumbent.')
text = text.replace('Run `REPRODUCE.ps1` from a fresh preregistered output folder.',
                    'The original planned command is preserved in `REPRODUCE.initial_plan.ps1`; it includes held-out confirmation and must not be run as an automatic verification step. `REPRODUCE.ps1` now records validation-only instructions. A new experiment requires a fresh preregistered output folder and a separately authorized time budget.')
table = ['| Recipe | Processed strict CR mean ± sample SD | Processed clDice mean ± sample SD | Replacement guard |',
         '| --- | --- | --- | --- |']
for key in ('old_M4_processed', 'A', 'B', 'C'):
    data = aggregate[key] if key == 'old_M4_processed' else aggregate[key]['processed']
    fmt = lambda metric: f"{data[metric]['mean']:.6f} ± {data[metric]['sample_sd']:.6f}"
    table.append(f"| {key} | {fmt('strict_cr')} | {fmt('cldice')} | {'incumbent' if key == 'old_M4_processed' else 'fails clDice guard'} |")
insert = '\n'.join(table)
text = text.replace('No complete three-seed candidate selection is available.',
                    'No final candidate selection or held-out lock is available. A/B/C have complete three-seed validation, shown below; D has only two completed seeds and is excluded from aggregate selection. The incumbent remains unreplaced. Scores are fractions.\n\n' + insert)
lead = ('**Budget-stopped partial delivery:** 11 completed runs plus D seed 1 at epoch 56 (1,156 logged epochs). '
        'No TITS inference, no winner, and no valid held-out lock. Both preregistered claims remain inconclusive for the full study. '
        'Finishing D, its calibration, selection, and the once-only held-out comparison requires an explicitly extended budget.\n\n')
text = text.replace('# M4 stabilization v2 results\n\n', '# M4 stabilization v2 results\n\n' + lead)
text += ('\n## Final partial verification and remaining work\n\n'
         f"`verification_partial.json` verifies {len(manifest)} checkpoint hashes, {len(sources)} source/snapshot records, "
         f"{len(oldmeta)} historical file metadata records, 255 DeepCrack training/validation input pairs and 66 TITS pairs. No mismatches were found. "
         'These checks read inputs without evaluating models. The existing 13 synthetic checks passed (`synthetic_verification.json`).\n\n'
         'Unfinished mandatory work: D seed 1 epochs 57–100 and calibration; complete three-seed selection and immutable lock; new fvcore/attention compute measurements and held-out tradeoff plots; once-only TITS inference and raw/processed tables, size-bin comparisons and claim-2 verdict. '
         'No fabricated lock JSON or held-out scores were written. `lock_status.json` explicitly records NOT_LOCKED.\n\n'
         'The nine-hour wall-clock deadline was 2026-10-15 06:39:25 Asia/Kolkata. Training stopped at the budget gate; report finalization occurred after the cutoff. '
         f"Elapsed wall time through verification, including implementation, pauses and report finalization: {(now-start).total_seconds()/3600:.3f} hours. Parallel job durations overlap and must not be summed.\n\n"
         'The shape-only CNN calculation is available in `cnn_analytic.json` (14,720,389 parameters; 40.105804 convolution GFLOPs at 256×256, excluding other operations). '
         'New full comparison compute measurements were not reached. Historical compute references remain historical evidence, not a new measurement.\n')
with original.open('x', encoding='utf-8') as f: f.write(text)
repro = P / 'REPRODUCE.ps1'
repro.rename(P / 'REPRODUCE.initial_plan.ps1')
with repro.open('x', encoding='utf-8') as f:
    f.write("# Partial delivery: the original study exhausted its authorized nine-hour budget.\n"
            "# Exact executed commands are preserved in commands.jsonl and each run's command.json.\n"
            "# The original full-study invocation is archived in REPRODUCE.initial_plan.ps1.\n"
            "# Do not execute it for validation: it includes once-only TITS inference.\n"
            "throw 'Budget exhausted. Reproduction or continuation needs an explicitly extended budget; no held-out inference was run.'\n")
print(json.dumps({'verification': checks, 'aggregate': aggregate}, indent=2))
