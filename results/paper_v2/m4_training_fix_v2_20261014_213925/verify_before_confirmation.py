"""Independent ledger verification before consuming the one TITS attempt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify():
    assert not (OUT/'tits_consumed.json').exists(), 'Confirmation has already been consumed'
    protocol = read(OUT/'protocol.json')
    assert [len(protocol['split'][k]) for k in ['train','val','unused']] == [210,45,45]
    assert len(set(sum(protocol['split'].values(),[]))) == 300
    assert protocol['split_seed'] == 42
    for path,digest in protocol['sources'].items():
        assert sha(path) == digest, f'Frozen source changed: {path}'
    locked = read(OUT/'locked_configs.json')
    assert sha(OUT/'locked_configs.json') == read(OUT/'locked_configs.sha256.json')['sha256']
    assert sha(OUT/'protocol.json') == locked['protocol_sha256']
    assert sha(OUT/'tits_manifest.json') == locked['tits_manifest_sha256']
    assert len(locked['cases']) == 24
    assert {(c['model'],c['mode'],c['seed']) for c in locked['cases']} == {
        (m,mode,seed) for m in ['Old M4','New M4','M2','DeepCrack']
        for mode in ['raw','processed'] for seed in [42,1,2]}
    verified = []
    for candidate in ['A','B','C','D']:
        for seed in [42,1,2]:
            directory = OUT/f'runs/{candidate}/seed_{seed}'
            result,config = read(directory/'training.json'),read(directory/'config.json')
            rows = [json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows] == list(range(1,101))
            assert result['epochs'] == 100 and config['seed'] == seed and config['split_seed'] == 42
            assert config['config']['model'] == protocol['baseline_config']['model']
            assert config['config']['training']['early_stopping_patience'] is None
            for i,row in enumerate(rows):
                assert row['optimizer_steps'] == (210 if candidate=='A' else 27)
                assert row['n_total'] == row['n_preserved']+row['n_missed']+row['n_fragmented'] == 157
                assert all(0 <= row[k] <= 1 for k in ['node_f1','cldice','pixel_f1','strict_cr'])
                if i<4:
                    assert row['smoothed_connectivity_score'] is None
                else:
                    expected = sum((r['cldice']+r['strict_cr'])/2 for r in rows[i-4:i+1])/5
                    assert abs(row['smoothed_connectivity_score']-expected)<1e-12
            best = max(rows[4:],key=lambda r:r['smoothed_connectivity_score'])
            assert result['selected_epoch'] == best['epoch']
            assert sha(result['selected_checkpoint']) == result['selected_checkpoint_sha256']
            assert all((directory/f'checkpoint_epoch_{epoch:03d}.pt').is_file() for epoch in range(10,101,10))
            timing = read(directory/'timing_gate.json')
            assert timing['passed'] and timing['projected_seconds'] <= 2400
            if candidate!='A':
                assert abs(rows[0]['lr']-.001)<1e-12
                assert abs(rows[-1]['lr']-.00001)<1e-12
                assert all(rows[i]['lr']>=rows[i+1]['lr'] for i in range(99))
            assert all(r['evaluated_weights']==('EMA' if candidate=='D' else 'current') for r in rows)
            validation = read(OUT/f'validation/{candidate}_seed{seed}_detailed.json')
            recipe = validation['recipe']
            assert recipe['low'] in protocol['grids']['thresholds'] and recipe['high'] in protocol['grids']['thresholds']
            assert recipe['min_area'] in protocol['grids']['min_areas'] and recipe['max_gap'] in protocol['grids']['max_gaps']
            verified.append(dict(candidate=candidate,seed=seed,selected_epoch=result['selected_epoch'],
                training_wall_seconds=result['wall_seconds'],timing_projection_seconds=timing['projected_seconds']))
    selection = read(OUT/'selection.json')
    incumbent = dict(selection['historical_mean']);winner='historical'
    for row in selection['trials']:
        means = {k:row['processed'][k]['mean'] for k in ['strict_cr','cldice','pixel_f1']}
        qualifies = means['strict_cr']>=incumbent['strict_cr']+.01-1e-12 and means['cldice']>=incumbent['cldice']-1e-12
        assert qualifies == row['accepted'] and winner==row['incumbent_before']
        if qualifies:incumbent=means;winner=row['candidate']
    assert winner==selection['winner']==locked['winner']
    for case in locked['cases']:
        assert sha(case['descriptor']['checkpoint'])==case['descriptor']['checkpoint_sha256']
    augmentation = read(OUT/'augmented_graphs/manifest.json')
    assert len(augmentation['records'])==210*8
    assert all(sha(r['path'])==r['sha256'] for r in augmentation['records'])
    existing = read(OUT/'existing_historical_artifacts.json')
    assert all(Path(p).is_file() and [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==meta for p,meta in existing.items())
    tits = read(OUT/'tits_manifest.json')
    assert tits['annotated_pairs']==66 and tits['no_deepcrack_training_image_duplicates']
    assert all([Path(tits['root'],p).stat().st_size,Path(tits['root'],p).stat().st_mtime_ns]==meta for p,meta in tits['external_file_metadata'].items())
    output=dict(status='passed',time=datetime.now(timezone.utc).isoformat(),runs=verified,
        full_100_epoch_runs=len(verified),training_epochs=1200,checkpoint_selection_verified=True,
        frozen_sources_unchanged=True,historical_files_unchanged=len(existing),
        external_tits_unchanged=True,lock_sha256=sha(OUT/'locked_configs.json'),
        held_out_not_yet_consumed=True,winning_recipe=winner)
    with (OUT/'verification_before_confirmation.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,indent=2)
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    verify()
