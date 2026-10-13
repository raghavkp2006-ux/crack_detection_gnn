import os, sys, json, time, random, argparse, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import f1_score
from common import *
from src.utils.dataset import CrackGraphDataset, get_splits
from deepcrack_model import DeepCrackHED, cb_bce, MEAN, STD

# Populated only by main(), after the graph prediction gate has created labels.
TRAIN_DS = None
VAL_Y = {}
LAB = None
TEST_STEMS = []
OFF = None

def node_probs(prob, seg):
    lab, inv = np.unique(seg, return_inverse=True); inv = inv.reshape(-1)
    s = np.bincount(inv, weights=prob.reshape(-1), minlength=len(lab)); c = np.bincount(inv, minlength=len(lab))
    return (s / c).astype(np.float32)

class DS(torch.utils.data.Dataset):
    def __init__(s, stems, split, train): s.stems, s.split, s.train = stems, split, train
    def __len__(s): return len(s.stems)
    def __getitem__(s, i):
        st = s.stems[i]
        img = read_rgb(find(os.path.join(DC, f'{s.split}_img'), st)).astype(np.float32) / 255.
        m = read_mask(find(os.path.join(DC, f'{s.split}_lab'), st))
        if s.train:
            if random.random() < .5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < .5: img, m = img[::-1], m[::-1]
        x = (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1) - MEAN) / STD
        return x, torch.from_numpy(np.ascontiguousarray(m))[None]

@torch.no_grad()
def predict(model, stems, split, dev):
    model.eval(); out = []
    for st in stems:
        x, _ = DS([st], split, False)[0]
        with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
            fused, _ = model(x[None].to(dev))
        p = node_probs(torch.sigmoid(fused.float())[0, 0].cpu().numpy(), get_segments(st, split))
        if split == 'train' and len(p) != len(VAL_Y[st]):
            raise ValueError(f'validation node-count mismatch on {st}')
        out.append(p)
    return out

def run_seed(seed, a, dev):
    out = os.path.join(OUT, 'runs', 'DeepCrack', f'seed_{seed}')
    prob_path = os.path.join(OUT, 'probs', 'DeepCrack', f'seed_{seed}.npz')
    if all(os.path.exists(p) for p in [os.path.join(out, 'result.json'), os.path.join(out, 'best_model.pt'), prob_path]): return
    os.makedirs(out, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tr, va, _ = get_splits(TRAIN_DS, 0.7, 0.15, seed)
    trs = [stem(TRAIN_DS._file_list[i]) for i in tr]; vas = [stem(TRAIN_DS._file_list[i]) for i in va]
    dl = torch.utils.data.DataLoader(DS(trs, 'train', True), batch_size=a.bs, shuffle=True, num_workers=0)
    model = DeepCrackHED(True).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=3)
    scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
    ckpt = os.path.join(out, 'best_model.pt'); best, bad, t0 = -1, 0, time.time()
    yv = np.concatenate([VAL_Y[s] for s in vas])
    for ep in range(a.max_epochs):
        epoch_start = time.time()
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev); opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                fused, sides = model(x)
            loss = cb_bce(fused.float(), y) + sum(cb_bce(s.float(), y) for s in sides)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        f1 = f1_score(yv, np.concatenate(predict(model, vas, 'train', dev)) >= 0.5, zero_division=0)
        if ep == 0: print(f'seed {seed} first epoch (including validation): {time.time() - epoch_start:.1f}s', flush=True)
        sched.step(f1); print(f'seed {seed} ep {ep+1} val node-F1@0.5 {f1:.4f}', flush=True)
        if f1 > best: best, bad = f1, 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    pv = np.concatenate(predict(model, vas, 'train', dev))
    tau, bvf = 0.5, -1
    for t in np.arange(0.10, 0.92, 0.02):
        f = f1_score(yv, pv >= np.float32(t), zero_division=0)
        if f > bvf: bvf, tau = f, float(t)
    pt = predict(model, TEST_STEMS, 'test', dev)
    for i, p in enumerate(pt): assert len(p) == OFF[i + 1] - OFF[i], f'node-count mismatch on {TEST_STEMS[i]}'
    pt = np.concatenate(pt)
    f1 = f1_score(LAB['y'], pt >= np.float32(tau), zero_division=0)
    os.makedirs(os.path.join(OUT, 'probs', 'DeepCrack'), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, 'probs', 'DeepCrack', f'seed_{seed}.npz'), probs=pt, tau=tau)
    json.dump(dict(seed=seed, best_tau=tau, best_val_f1=bvf, holdout_f1=float(f1), epochs=ep + 1, params=nparams,
                   train_time_sec=time.time() - t0, val_stems=vas), open(os.path.join(out, 'result.json'), 'w'), indent=2)
    print(f'DONE seed {seed}: tau={tau:.2f} test node-F1={f1:.4f}', flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--max_epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=10); ap.add_argument('--bs', type=int, default=4); a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.max_epochs < 1 or a.patience < 1 or a.bs < 1:
        ap.error('max_epochs, patience, and bs must be positive')
    TRAIN_DS = CrackGraphDataset(rel('data/graphs/train'))
    VAL_Y = {stem(f): TRAIN_DS[i].y.numpy() for i, f in enumerate(TRAIN_DS._file_list)}
    LAB = np.load(os.path.join(OUT, 'test_labels.npz'))
    TEST_STEMS = [str(s) for s in LAB['stems']]; OFF = LAB['offsets']
    from PIL import Image
    sizes = {}
    for part, stems_ in [('train', list(VAL_Y)), ('test', TEST_STEMS)]:
        sizes[part] = set()
        for st in stems_:
            with Image.open(find(os.path.join(DC, f'{part}_img'), st)) as im:
                sizes[part].add(im.size)
    print(f'Image sizes (W, H): {sizes}', flush=True)
    if any(len(v) != 1 for v in sizes.values()) and a.bs != 1:
        print('Mixed image sizes: using batch size 1.', flush=True); a.bs = 1
    for s in SEEDS: run_seed(s, a, dev)
