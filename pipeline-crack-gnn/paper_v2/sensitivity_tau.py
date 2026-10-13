"""Secondary validation-only threshold sensitivity; primary artifacts stay unchanged."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from common import BASE, OUT, SEEDS, GRAPH_CELLS, rel, stem
from src.models.hybrid_model import build_model
from src.train import load_config
from src.utils.dataset import CrackGraphDataset, get_splits
from predict_all import node_probs as graph_node_probs
from stats_core import load_labels, load_probs, decide
from deepcrack_model import DeepCrackHED
import train_deepcrack as dc


def main():
    destination = Path(OUT, 'sensitivity_tau.json')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    train_ds = CrackGraphDataset(rel('data/graphs/train'))
    test_ds = CrackGraphDataset(rel('data/graphs/test'))
    yt, offsets, test_stems = load_labels()
    assert test_stems == [stem(p) for p in test_ds._file_list], 'Test stem order differs from primary labels'
    assert offsets[-1] == len(yt) and len(offsets) == len(test_stems) + 1
    assert np.isin(yt, [0, 1]).all(), 'Test labels must be binary'
    validation = {}
    balance = {}
    dc.VAL_Y = {stem(p): train_ds[i].y.numpy() for i, p in enumerate(train_ds._file_list)}
    for seed in SEEDS:
        _, indices, _ = get_splits(train_ds, 0.7, 0.15, seed)
        assert len(indices) == 45, f'Expected 45 validation graphs for seed {seed}, found {len(indices)}'
        stems = [stem(train_ds._file_list[i]) for i in indices]
        yv = np.concatenate([dc.VAL_Y[s] for s in stems])
        assert np.isin(yv, [0, 1]).all(), f'Validation labels must be binary for seed {seed}'
        validation[seed] = (indices, stems, yv)
        balance[str(seed)] = dict(validation_n_graphs=len(indices), validation_n_nodes=len(yv),
                                  validation_crack_node_fraction=float(yv.mean()),
                                  test_n_nodes=len(yt), test_crack_node_fraction=float(yt.mean()))
    roots = {key: Path(BASE, root) for key, (_, _, root) in GRAPH_CELLS.items()}
    roots['DeepCrack'] = Path(OUT, 'runs', 'DeepCrack')
    # Require every run before doing any sensitivity inference. No partial output.
    for key, root in roots.items():
        for seed in SEEDS:
            for p in [root / f'seed_{seed}' / 'best_model.pt', root / f'seed_{seed}' / 'result.json',
                      Path(OUT, 'probs', key, f'seed_{seed}.npz')]:
                if not p.is_file(): raise FileNotFoundError(p)
    grid = np.arange(0.10, 1.00, 0.02)  # 0.10 through 0.98, inclusive.
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result = dict(analysis='secondary validation-only threshold sensitivity',
                  threshold_grid=grid.tolist(), tie_rule='first maximum on ascending validation grid',
                  primary_artifacts_unchanged=True, class_balance_by_seed=balance,
                  old_tau_at_0_90_count=0, runs=[])
    for key, root in roots.items():
        cfg = load_config(rel(GRAPH_CELLS[key][0])) if key in GRAPH_CELLS else None
        for seed in SEEDS:
            run = root / f'seed_{seed}'
            with (run / 'result.json').open(encoding='utf-8') as f: old = json.load(f)
            old_tau = float(old['best_tau'])
            pt, cached_tau = load_probs(key, seed)
            assert cached_tau == old_tau, f'Primary threshold mismatch for {key}/{seed}'
            old_test_recomputed = float(f1_score(yt, decide(pt, old_tau), zero_division=0))
            assert abs(old_test_recomputed - old['holdout_f1']) <= 2e-3, f'Primary test F1 mismatch for {key}/{seed}'
            indices, stems, yv = validation[seed]
            model = (build_model(cfg) if cfg is not None else DeepCrackHED(pretrained=False)).to(dev)
            model.load_state_dict(torch.load(run / 'best_model.pt', map_location=dev, weights_only=True))
            if cfg is not None:
                loader = DataLoader(Subset(train_ds, indices), batch_size=1, shuffle=False)
                pv = np.concatenate(graph_node_probs(model, cfg, loader, dev)).astype(np.float32)
            else:
                pv = np.concatenate(dc.predict(model, stems, 'train', dev)).astype(np.float32)
            assert pv.shape == yv.shape, f'Validation probability/label mismatch for {key}/{seed}'
            assert np.isfinite(pv).all() and np.all((pv >= 0) & (pv <= 1)), f'Invalid validation probabilities for {key}/{seed}'
            scores = [float(f1_score(yv, decide(pv, tau), zero_division=0)) for tau in grid]
            best = int(np.argmax(scores))
            new_tau = float(grid[best])
            boundary = bool(np.isclose(old_tau, 0.90, rtol=0.0, atol=1e-8))
            row = dict(model=key, seed=seed, old_tau=old_tau, new_tau=new_tau,
                       old_test_f1=float(old['holdout_f1']),
                       new_test_f1=float(f1_score(yt, decide(pt, new_tau), zero_division=0)),
                       old_test_f1_recomputed=old_test_recomputed,
                       new_validation_f1=scores[best], old_tau_at_0_90=boundary,
                       validation_stems=stems, validation_f1_by_tau=scores)
            result['old_tau_at_0_90_count'] += int(boundary)
            result['runs'].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ('validation_stems', 'validation_f1_by_tau')}), flush=True)
            del model
    # Exclusive creation guarantees existing sensitivity outputs are never replaced.
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(f'Saved {destination}', flush=True)


if __name__ == '__main__':
    main()
