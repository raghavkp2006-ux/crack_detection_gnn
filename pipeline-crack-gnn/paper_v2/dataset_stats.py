import json, numpy as np, torch
from common import *
from src.utils.dataset import CrackGraphDataset, get_splits

def main():
    tr_ds = CrackGraphDataset(rel('data/graphs/train')); te_ds = CrackGraphDataset(rel('data/graphs/test'))
    S = {'n_train_dir_graphs': len(tr_ds), 'seeds': {}}
    for s in SEEDS:
        tr, va, un = get_splits(tr_ds, 0.7, 0.15, s)
        y = torch.cat([tr_ds[i].y for i in tr]); N = int(y.numel()); P = int(y.sum())
        S['seeds'][s] = dict(n_train=len(tr), n_val=len(va), n_unused=len(un), train_nodes=N, train_crack_nodes=P,
            train_crack_frac=P / N, focal_alpha=[P / N, (N - P) / N],                  # [bg, crack] as used by get_inverted_class_weights
            wce_weights=[N / (2 * (N - P)), N / (2 * P)],                              # as used by get_class_weights
            val_stems=[stem(tr_ds._file_list[i]) for i in va])
    ys = [te_ds[i].y for i in range(len(te_ds))]; tot = sum(len(y) for y in ys)
    S['test'] = dict(n_images=len(te_ds), n_nodes=int(tot), crack_node_frac=float(sum(int(y.sum()) for y in ys) / tot),
        mean_nodes=float(tot / len(ys)), mean_directed_edges=float(np.mean([te_ds[i].edge_index.shape[1] for i in range(len(te_ds))])))
    sizes, fr = set(), []
    for p in te_ds._file_list:
        m = read_mask(find(os.path.join(DC, 'test_lab'), stem(p))); sizes.add(m.shape); fr.append(float(m.mean()))
    S['test'].update(image_sizes=[list(x) for x in sorted(sizes)], crack_pixel_frac=float(np.mean(fr)))
    json.dump(S, open(os.path.join(OUT, 'dataset_stats.json'), 'w'), indent=2)
    print(json.dumps(S['test'], indent=2)); print(S['seeds'][42]['n_train'], S['seeds'][42]['n_val'], S['seeds'][42]['focal_alpha'])

if __name__ == '__main__':
    main()
