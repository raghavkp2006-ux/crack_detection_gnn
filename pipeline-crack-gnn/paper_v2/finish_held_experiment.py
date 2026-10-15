"""Resume the explicit validation-review hold, then consume test exactly once."""
import argparse
from pathlib import Path
import sys
import time
import traceback

from improve_m2_m4 import Experiment, read, save, sha, utc
from review_blend_locks import review
from audit_improvement import audit


def finish(output):
    experiment=Experiment(output); directory=experiment.out
    if (directory/'test_consumed.json').exists(): raise RuntimeError('Test already consumed')
    if not (directory/'RESULTS.md').exists(): raise RuntimeError('Wait until the held worker has exited and saved its partial report')
    failures=sorted(directory.glob('failure_*.json'))
    if not failures or any('FileExistsError' not in read(p)['error'] or 'locked_configs.sha256.json' not in read(p)['error'] for p in failures):
        raise RuntimeError('Only the deliberate pre-test hash-file hold can be resumed')
    experiment.event('invocation',argv=sys.argv,cwd=str(Path.cwd()),purpose='review_hold_then_single_test')
    started=time.perf_counter()
    review(output)
    history=directory/'audit_history/pretest_hold'
    history.mkdir(parents=True,exist_ok=False)
    for path in [directory/'RESULTS.md',directory/'checkpoint_manifest.json',*failures]:
        if not path.resolve().is_relative_to(directory): raise ValueError('Archive source leaves experiment directory')
        target=history/path.name
        if not target.resolve().is_relative_to(directory) or target.exists(): raise ValueError('Invalid archive target')
        path.rename(target)
    archive=[]
    for name in ('review_blend_locks.py','finish_held_experiment.py'):
        source=Path(__file__).with_name(name)
        target=directory/'source_snapshot/pipeline-crack-gnn/paper_v2'/name
        with target.open('xb') as f: f.write(source.read_bytes())
        archive.append(dict(path=str(target.relative_to(directory/'source_snapshot')),sha256=sha(source)))
    save(directory/'source_snapshot_review_manifest.json',dict(base_manifest_sha256=sha(directory/'source_snapshot_manifest.json'),files=archive))
    try:
        experiment.final_test()
        experiment.report()
        audit(directory)
        events=[__import__('json').loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        budget_elapsed={}
        for event in events:
            if event['kind']=='stage_end': budget_elapsed[event['stage']]=max(budget_elapsed.get(event['stage'],0.),event['wall_seconds'])
        save(directory/'consolidated_timing.json',dict(time=utc(),budget_elapsed_seconds=budget_elapsed,
                                                      continuation_wall_seconds=time.perf_counter()-started,
                                                      note='Resumed stage wall times are cumulative from the original stage start; use the maximum per stage rather than summing repeated stage-end entries.'))
        reviewed=read(directory/'pretest_review.json')
        with (directory/'RESULTS.md').open('a',encoding='utf-8') as f:
            f.write('\n## Pre-test selection review\n\n')
            f.write('An explicit hash-file hold stopped automatic test access so the saved validation blend grids could be independently replayed. The temporary file-exists exit and draft report are preserved in audit_history/pretest_hold/. No test evaluation occurred during this hold.\n\n')
            f.write(f'Reviewed blend choices changed for seeds {[r["seed"] for r in reviewed["reviews"] if r["changed"]]}. Checkpoints, data membership, search grids, and comparator recipes were unchanged. The reviewed lock was sealed before the single test job. Full details are in pretest_review.json.\n\n')
            f.write('Repeated stage-end timings include resumed stages and are cumulative; consolidated_timing.json records the maximum per stage. Do not sum repeated cumulative entries. The source archive includes the exact independent reviewer and continuation code.\n\n')
            original=read(directory/'validation/blend_original_seed42.json')
            if original['alpha'] in (0.,1.):
                f.write(f'- Negative inference-only averaging result: the original-model blend chose M2 weight {original["alpha"]}; this is a single-constituent control rather than an averaging gain.\n')
    except Exception as ex:
        save(directory/'continuation_failure.json',dict(time=utc(),error=f'{type(ex).__name__}: {ex}',traceback=traceback.format_exc()))
        if not (directory/'RESULTS.md').exists(): experiment.report()
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    finish(parser.parse_args().output)
