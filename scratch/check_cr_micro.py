import os, glob, cv2, torch, numpy as np, pandas as pd
import sys
sys.path.insert(0, 'DeepSegmentor')
sys.path.insert(0, 'pipeline-crack-gnn')
from models.deepcrack_networks import DeepCrackNet
from src.models.hybrid_model import build_model
from src.train import load_config
from src.evaluate import reconstruct_mask
import skimage.measure
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = {
    'M1': ('pipeline-crack-gnn/configs/baseline_deep_gnn.yaml', 'pipeline-crack-gnn/results/M1_deep_gnn/best_model.pt', 0.60, 'deep_gnn'),
    'M2': ('pipeline-crack-gnn/configs/shallow_gnn.yaml', 'pipeline-crack-gnn/results/M2_shallow_gnn/best_model.pt', 0.70, 'shallow_gnn'),
    'M3': ('pipeline-crack-gnn/configs/hybrid_no_pe.yaml', 'pipeline-crack-gnn/results/M3_hybrid_no_pe/best_model.pt', 0.74, 'hybrid'),
    'M4': ('pipeline-crack-gnn/configs/step2_loss/hybrid_full_focal.yaml', 'pipeline-crack-gnn/results/step2_loss/M4_hybrid_full/best_model.pt', 0.66, 'hybrid'),
}
gnn_models = {}
for k, (cfg_p, ckpt_p, tau, mtype) in configs.items():
    cfg = load_config(cfg_p)
    m = build_model(cfg).to(device).eval()
    m.load_state_dict(torch.load(ckpt_p, map_location=device))
    gnn_models[k] = (m, tau, mtype)

dc = DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch').to(device).eval()
dc.load_state_dict(torch.load('pipeline-crack-gnn/results/multiseed/deepcrack/seed_42/best_model.pt', map_location=device))

img_files = sorted(glob.glob('DeepCrack/test_img/*'))

gts = []
preds = {k: [] for k in ['M1', 'M2', 'M3', 'M4', 'DeepCrack']}

for img_p in img_files:
    bname = os.path.splitext(os.path.basename(img_p))[0]
    lab_p = f'DeepCrack/test_lab/{bname}.png'
    if not os.path.exists(lab_p): lab_p = f'DeepCrack/test_lab/{bname}.jpg'
    gt = (cv2.imread(lab_p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    gts.append(gt)
    H, W = gt.shape

    gdata = torch.load(f'pipeline-crack-gnn/data/graphs/test/{bname}.pt', weights_only=False).to(device)
    seg = np.load(f'pipeline-crack-gnn/data/superpixels/test/{bname}.npy')
    with torch.no_grad():
        for k, (m, tau, mtype) in gnn_models.items():
            out = m(gdata.x, gdata.edge_index, gdata.pos) if mtype == 'hybrid' else m(gdata.x, gdata.edge_index)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            n_preds = (probs >= tau).astype(int)
            mask = reconstruct_mask(seg, n_preds, seg.shape)
            preds[k].append(mask)

    img_bgr = cv2.imread(img_p)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (256, 256))
    t = torch.from_numpy(img_res).permute(2,0,1).float()/255.0
    t = ((t - 0.5)/0.5).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, _, _, fused = dc(t)
        fused_up = F.interpolate(fused, size=(H, W), mode='bilinear', align_corners=True)
        prob_dc = torch.sigmoid(fused_up)[0,0].cpu().numpy()
        mask_dc = (prob_dc >= 0.36).astype(np.uint8)
        preds['DeepCrack'].append(mask_dc)

def connectivity_ratio_dataset(model_preds, gts, overlap_thresh=0.5):
    per_image_crs = []
    tot_segs, pres_segs, miss_segs, frag_segs = 0, 0, 0, 0
    for p, g in zip(model_preds, gts):
        gt_bin = (g > 0).astype(bool)
        p_bin = (p > 0).astype(bool)
        gt_labels, n_tot_img = skimage.measure.label(gt_bin, connectivity=2, return_num=True)
        if n_tot_img == 0:
            per_image_crs.append(1.0 if not p_bin.any() else 0.0)
            continue
        pred_labels = skimage.measure.label(p_bin, connectivity=2)
        n_pres_img = 0
        for i in range(1, n_tot_img + 1):
            tot_segs += 1
            g_i = (gt_labels == i)
            ov = g_i & p_bin
            if ov.sum() / g_i.sum() < overlap_thresh:
                miss_segs += 1
            else:
                _, n_ov = skimage.measure.label(ov, connectivity=2, return_num=True)
                up = np.unique(pred_labels[ov])
                if n_ov == 1 and len(up) == 1:
                    pres_segs += 1
                    n_pres_img += 1
                else:
                    frag_segs += 1
        per_image_crs.append(n_pres_img / n_tot_img)
    return {
        'cr_macro': float(np.mean(per_image_crs)),
        'cr_micro': float(pres_segs / tot_segs) if tot_segs > 0 else 1.0,
        'n_total': tot_segs,
        'n_preserved': pres_segs,
        'n_missed': miss_segs,
        'n_fragmented': frag_segs
    }

print('=== CR SUMMARY (tau=0.5) ===')
for k in ['M1', 'M2', 'M3', 'M4', 'DeepCrack']:
    r = connectivity_ratio_dataset(preds[k], gts, 0.5)
    print(f"{k:10s} | Micro CR: {r['cr_micro']:.4f} ({r['n_preserved']}/{r['n_total']}) | Macro CR: {r['cr_macro']:.4f} | Missed: {r['n_missed']} | Frag: {r['n_fragmented']}")

print('\n=== MICRO CR THRESHOLD SWEEP (M4 vs DeepCrack) ===')
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    r_m4 = connectivity_ratio_dataset(preds['M4'], gts, th)
    r_dc = connectivity_ratio_dataset(preds['DeepCrack'], gts, th)
    diff = r_dc['cr_micro'] - r_m4['cr_micro']
    print(f"tau={th:.1f}: M4 Micro={r_m4['cr_micro']:.4f}, DC Micro={r_dc['cr_micro']:.4f}, Gap={diff:+.4f}")
