"""Complete the held pre-test lock by replaying saved validation blend grids.

No test data or inference is used. This independently checks the alpha=0
incumbent edge case and preserves the original draft and all original trials.
Only frozen-grid refinements missing from the saved trials are scored, using
already cached validation probabilities and the original validation GT.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from improvement_core import GRIDS, MetricSet, Recipe, process, qualifies, rank
from improve_m2_m4 import Experiment, read, save, sha, utc


def operations(recipe):
    return int(recipe['low']!=recipe['high'])+int(recipe['min_area']>0)+int(recipe['max_gap']>0)


def select_pair(rows, baseline, guard, raw_recipe):
    eligible=[r for r in rows if 'family' not in r and qualifies(r['metrics'],baseline,guard)]
    if not eligible:
        return dict(alpha=0.,recipe=dict(raw_recipe),metrics=dict(baseline))
    best=max(eligible,key=lambda r:(rank(r['metrics']),-operations(r['recipe'])))
    return {k:best[k] for k in ('alpha','recipe','metrics')}


def review(output):
    experiment=Experiment(output); directory=experiment.out
    if (directory/'test_consumed.json').exists(): raise RuntimeError('Cannot review after test consumption')
    if not read(directory/'locked_configs.sha256.json').get('pretest_hold'):
        raise RuntimeError('An explicit pre-test hold is required')
    lock=read(directory/'locked_configs.json')
    if (directory/'pretest_review.json').exists(): raise FileExistsError('Already reviewed')
    experiment.verify_sources()
    original_draft_sha=sha(directory/'locked_configs.json')
    reviews=[]; started=time.perf_counter()
    with experiment.stage('train'):
        items=None; metrics=None
        for case in lock['cases']:
            if case['model']!='M2+M4': continue
            seed=case['seed']; experiment.check()
            m4=next(c for c in lock['cases'] if c['seed']==seed and c['model']=='M4')
            old=read(directory/f'validation/blend_final_seed{seed}.json')
            rows=read(old['trials']); guard=old['guard']
            selected=select_pair(rows,old['baseline'],guard,m4['recipe'])
            cached={(float(r['alpha']),tuple(Recipe(**r['recipe']).json().values())):r['metrics'] for r in rows}
            extra=[]; probabilities=None
            if selected['alpha']!=0:
                for family,field in [('min_areas','min_area'),('max_gaps','max_gap')]:
                    incumbent=dict(selected); choices=[]
                    for value in GRIDS[family]:
                        recipe=replace(Recipe(**incumbent['recipe']),**{field:value})
                        key=float(selected['alpha']),tuple(recipe.json().values())
                        if key in cached:
                            score=cached[key]; reused=True
                        else:
                            if probabilities is None:
                                if items is None:
                                    items=experiment.items(experiment.protocol['split']['val'])
                                    metrics=MetricSet([i['gt'] for i in items])
                                p2tag='M2_original_seed42' if seed==42 else f'M2_final_seed{seed}'
                                p4tag=f'M4_final_seed{seed}'
                                def load(tag,descriptor):
                                    path=directory/'probabilities/val'/tag
                                    manifest=read(path/'manifest.json')
                                    if manifest['checkpoint_sha256']!=descriptor['checkpoint_sha256']:
                                        raise ValueError('Validation cache/checkpoint mismatch')
                                    if manifest['stems']!=experiment.protocol['split']['val']:
                                        raise ValueError('Validation cache ordering mismatch')
                                    return [np.load(path/f'{i:03d}.npz')['probability'] for i in range(45)]
                                p2=load(p2tag,case['m2_descriptor']); p4=load(p4tag,case['descriptor'])
                                probabilities=[np.float32(selected['alpha'])*a+np.float32(1-selected['alpha'])*b for a,b in zip(p2,p4)]
                            experiment.check()
                            score=metrics.score(process(p,recipe) for p in probabilities); reused=False
                        extra.append(dict(alpha=selected['alpha'],recipe=recipe.json(),metrics=score,family=family,reused_saved_score=reused))
                        if qualifies(score,incumbent['metrics'],guard):
                            choices.append(dict(alpha=selected['alpha'],recipe=recipe.json(),metrics=score))
                    if choices:
                        selected=max(choices,key=lambda c:(rank(c['metrics']),-operations(c['recipe'])))
            changed=(selected['alpha']!=case['alpha'] or selected['recipe']!=case['recipe'] or selected['metrics']!=case['validation'])
            result=dict(tag=f'blend_reviewed_seed{seed}',baseline=old['baseline'],guard=guard,**selected,
                        improved=selected['alpha']!=0 or selected['recipe']!=m4['recipe'],
                        original_trials=old['trials'],reviewed_before_test=True)
            save(directory/f'validation/blend_reviewed_seed{seed}.json',result)
            save(directory/f'validation_trials/blend_reviewed_seed{seed}.json',extra)
            reviews.append(dict(seed=seed,changed=changed,original=dict(alpha=case['alpha'],recipe=case['recipe'],metrics=case['validation']),
                                selected=selected,original_trials_sha256=sha(old['trials'])))
            case.update(alpha=selected['alpha'],recipe=selected['recipe'],validation=selected['metrics'],
                        calibration='per_seed_validation_blend_reviewed')
            experiment.event('blend_grid_review',seed=seed,changed=changed,selected=selected)
        review_record=dict(time=utc(),passed=True,reviewed_before_test=True,test_evaluations=0,
                           original_draft_sha256=original_draft_sha,selector_sha256=sha(__file__),
                           reviews=reviews,wall_seconds=time.perf_counter()-started,
                           command=dict(argv=sys.argv,cwd=str(Path.cwd())),
                           reason='Independent replay of frozen validation grids; alpha-zero cannot be displaced by a lower-ranked eligible candidate')
        # The old JSON had no finalized hash because of the explicit hold.
        # Preserve it as a draft; only the reviewed version is sealed for test.
        for filename in ('locked_configs.json','locked_configs.sha256.json'):
            source=directory/filename
            target=directory/filename.replace('.json','.pre_review.json')
            if target.exists(): raise FileExistsError(target)
            source.rename(target)
        lock['pretest_review']=dict(path='pretest_review.json',changed_seeds=[r['seed'] for r in reviews if r['changed']],
                                   original_draft_sha256=original_draft_sha)
        save(directory/'pretest_review.json',review_record)
        save(directory/'locked_configs.json',lock)
        save(directory/'locked_configs.sha256.json',dict(sha256=sha(directory/'locked_configs.json')))
        experiment.event('pretest_hold_released',changed_seeds=lock['pretest_review']['changed_seeds'],test_evaluations=0)
    print(json.dumps(review_record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    review(parser.parse_args().output)
