import os, sys, glob, torch
from common import *
import run_multiseed_experiment as rm
from torch_geometric.loader import DataLoader
from src.utils.dataset import CrackGraphDataset

rm.RESULTS_DIR = os.path.join(OUT, 'runs')            # train_single_seed writes RESULTS_DIR/<id>/seed_<s>/
NEW = [('M1_focal', GRAPH_CELLS['M1_focal']), ('M2_wce', GRAPH_CELLS['M2_wce']),
       ('M3_wce', GRAPH_CELLS['M3_wce']), ('M4_wce', GRAPH_CELLS['M4_wce'])]

def main():
    os.chdir(BASE)  # Existing trainer resolves config data paths relative to cwd.
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hold = DataLoader(CrackGraphDataset('data/graphs/test'), batch_size=1, shuffle=False)
    test_files = sorted(glob.glob('data/graphs/test/*.pt'))
    mask_dir = os.path.join(DC, 'test_lab')
    for seed in SEEDS:                                  # seed-major so partial results stay balanced
        for key, (cfg, loss, _) in NEW:
            done = os.path.join(OUT, 'runs', key, f'seed_{seed}', 'result.json')
            if os.path.exists(done) and os.path.exists(os.path.join(os.path.dirname(done), 'best_model.pt')): continue
            info = dict(id=key, name=key, config_path=rel(cfg), loss_type=loss)
            r = rm.train_single_seed(info, seed, dev, hold, mask_dir, test_files)
            print(f'{key} seed {seed}: tau={r["best_tau"]:.2f} F1={r["holdout_f1"]:.4f} ({r["train_time_sec"]:.0f}s)', flush=True)

if __name__ == '__main__':
    main()
