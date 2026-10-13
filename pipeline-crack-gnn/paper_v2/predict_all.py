import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score
from common import *
from src.utils.dataset import CrackGraphDataset
from src.models.hybrid_model import build_model
from src.train import load_config

LAB = os.path.join(OUT, 'test_labels.npz')

def build_labels(ds):
    ys = [ds[i].y.numpy() for i in range(len(ds))]
    off = np.concatenate([[0], np.cumsum([len(y) for y in ys])])
    labels = dict(y=np.concatenate(ys), offsets=off, stems=np.array([stem(f) for f in ds._file_list]))
    if os.path.exists(LAB):
        with np.load(LAB) as old:
            if any(not np.array_equal(old[k], v) for k, v in labels.items()):
                raise RuntimeError('Cached test labels/order differ from current graphs; review cached predictions.')
    else:
        np.savez(LAB, **labels)
    return labels

@torch.no_grad()
def node_probs(model, cfg, loader, dev):
    mt = cfg['model']['type']; model.eval(); out = []
    for d in loader:
        d = d.to(dev)
        o = model(d.x, d.edge_index, d.pos) if mt == 'hybrid' else model(d.x, d.edge_index)
        out.append(torch.softmax(o, 1)[:, 1].cpu().numpy())
    return out

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CrackGraphDataset(rel('data/graphs/test'))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    y = build_labels(ds)['y']; check = {}; missing = []
    for key, (cfgp, loss, root) in GRAPH_CELLS.items():
        cfg = load_config(rel(cfgp))
        for s in SEEDS:
            run = os.path.join(BASE, root, f'seed_{s}')
            rp = os.path.join(run, 'result.json'); ckpt = os.path.join(run, 'best_model.pt')
            out = os.path.join(OUT, 'probs', key, f'seed_{s}.npz')
            if not (os.path.exists(rp) and os.path.exists(ckpt)):
                if root.startswith('results/multiseed/') or os.path.exists(out):
                    raise FileNotFoundError(f'Required checkpoint/result missing: {run}')
                print('MISSING (not trained yet)', key, s); missing.append(f'{key}/{s}'); continue
            with open(rp) as f: r = json.load(f)
            tau = float(r['best_tau'])
            # Re-infer changed sources; always verify cached probabilities on resume.
            cached = os.path.exists(out) and os.path.getmtime(out) >= max(os.path.getmtime(rp), os.path.getmtime(ckpt))
            if cached:
                with np.load(out) as z:
                    p = z['probs'].astype(np.float32)
                    if float(z['tau']) != tau: cached = False
            if not cached:
                model = build_model(cfg).to(dev)
                model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
                p = np.concatenate(node_probs(model, cfg, loader, dev)).astype(np.float32)
                del model
            if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                raise ValueError(f'Invalid node probabilities: {key}/{s}')
            f1 = f1_score(y, (p >= np.float32(tau)).astype(int), zero_division=0)
            ok = abs(f1 - r['holdout_f1']) <= 2e-3
            check[f'{key}/{s}'] = dict(recomputed=float(f1), stored=r['holdout_f1'], ok=bool(ok))
            with open(os.path.join(OUT, 'predict_check.json'), 'w') as f: json.dump(check, f, indent=2)
            print(f'{key} seed {s}: recomputed {f1:.4f} stored {r["holdout_f1"]:.4f} {"OK" if ok else "MISMATCH"}', flush=True)
            if not ok:
                message = f'VERIFICATION FAILED: {key}/{s}: recomputed {f1:.8f}, stored {r["holdout_f1"]:.8f}.'
                with open(os.path.join(BASE, 'paper_v2', 'NOTES.md'), 'a', encoding='utf-8') as f: f.write('\n' + message + '\n')
                raise SystemExit(message)
            if not cached:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                np.savez_compressed(out, probs=p, tau=tau)
    print(f'Verified {sum(v["ok"] for v in check.values())}/{len(check)} available runs; {len(missing)} new runs pending.')

if __name__ == '__main__':
    main()
