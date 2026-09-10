"""
Multi-Seed Training, Validation Calibration, and Holdout Evaluation for DeepCrack (Liu et al. 2019).

Integrates Liu et al. 2019 DeepCrack architecture into the DCS GNN evaluation pipeline:
- Exact 5-seed training (42, 123, 456, 789, 2026).
- Exact 300-train (210 train / 45 val) and 237 holdout test split parity.
- DeepSegmentor DeepCrackNet + BinaryFocalLoss.
- Non-circular validation threshold calibration (tau*).
- Apples-to-apples superpixel node pooling (67,998 test nodes) and full-resolution pixel evaluation.
"""

import os
import sys
import glob
import json
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DCS_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEEPSEG_DIR = os.path.join(DCS_ROOT, "DeepSegmentor")
sys.path.insert(0, DEEPSEG_DIR)
sys.path.insert(0, BASE_DIR)

from models.deepcrack_networks import DeepCrackNet, BinaryFocalLoss
from src.utils.dataset import CrackGraphDataset, get_splits
from src.utils.metrics import compute_iou

SEEDS = [42, 123, 456, 789, 2026]
RESULTS_DIR = os.path.join(BASE_DIR, "results", "multiseed", "deepcrack")
SAVE_PREDS_DIR = os.path.join(BASE_DIR, "results", "deepcrack", "test_latest", "images")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVE_PREDS_DIR, exist_ok=True)

# Image and mask paths
TRAIN_IMG_DIR = os.path.join(DCS_ROOT, "DeepCrack", "train_img")
TRAIN_LAB_DIR = os.path.join(DCS_ROOT, "DeepCrack", "train_lab")
TEST_IMG_DIR = os.path.join(DCS_ROOT, "DeepCrack", "test_img")
TEST_LAB_DIR = os.path.join(DCS_ROOT, "DeepCrack", "test_lab")
TEST_SP_DIR = os.path.join(BASE_DIR, "data", "superpixels", "test")
TEST_GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs", "test")


def classify_crack_geometry(mask_path):
    if not os.path.exists(mask_path):
        return 'Unknown'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 'Unknown'
    pts = np.argwhere(mask > 127)
    if len(pts) == 0:
        return 'Clean_Negative'
    H, W = mask.shape
    y_min, x_min = pts.min(axis=0)
    y_max, x_max = pts.max(axis=0)
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    area_frac = len(pts) / (H * W)
    if area_frac < 0.025:
        return 'Thin_Fine_Fissure'
    elif aspect_ratio >= 2.0:
        return 'Long_Elongated'
    else:
        return 'Branched_Complex'


class DeepCrackImageDataset(Dataset):
    """
    Dataset loader for raw DeepCrack images and ground truth binary masks.
    """
    def __init__(self, img_dir, lab_dir, img_size=(256, 256), augment=False):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_size = img_size
        self.augment = augment
        
        exts = ('*.jpg', '*.png')
        self.img_files = []
        for ext in exts:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_files = sorted(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lab_path = os.path.join(self.lab_dir, f"{basename}.png")
        if not os.path.exists(lab_path):
            lab_path = os.path.join(self.lab_dir, f"{basename}.jpg")

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        if os.path.exists(lab_path):
            mask = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((orig_h, orig_w), dtype=np.float32)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                img_rgb = np.fliplr(img_rgb).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                img_rgb = np.flipud(img_rgb).copy()
                mask = np.flipud(mask).copy()

        # Resize to fixed training resolution
        w, h = self.img_size
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Normalize to [-1, 1] as in DeepSegmentor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - 0.5) / 0.5
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return {
            'image': img_tensor,
            'label': mask_tensor,
            'orig_shape': (orig_h, orig_w),
            'id': basename,
            'img_path': img_path,
            'lab_path': lab_path
        }


def pool_dense_prob_to_superpixels(prob_map, segments):
    """
    Pools dense 2D probability map into superpixel node probabilities.
    """
    H, W = prob_map.shape
    if segments.shape != (H, W):
        segments = cv2.resize(segments, (W, H), interpolation=cv2.INTER_NEAREST)

    labels = np.unique(segments)
    node_probs = np.zeros(len(labels), dtype=np.float32)
    for i, seg_id in enumerate(labels):
        mask = (segments == seg_id)
        if mask.any():
            node_probs[i] = prob_map[mask].mean()
    return node_probs


def train_single_seed(seed, device, epochs=30, batch_size=4, lr=0.001):
    print(f"\n==================================================================")
    print(f"  Training DeepCrack (Liu et al. 2019) on Seed {seed}")
    print(f"==================================================================")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Dataset splits matching M1-M4 (210 train / 45 val)
    full_train_ds = DeepCrackImageDataset(TRAIN_IMG_DIR, TRAIN_LAB_DIR, augment=True)
    graph_ds = CrackGraphDataset('data/graphs/train')
    train_idx, val_idx, test_idx = get_splits(graph_ds, 0.7, 0.15, seed)

    train_sub = Subset(full_train_ds, train_idx)
    # Val dataset without augmentation
    val_raw_ds = DeepCrackImageDataset(TRAIN_IMG_DIR, TRAIN_LAB_DIR, augment=False)
    val_sub = Subset(val_raw_ds, val_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=0)

    # 2. Build DeepCrack Net
    model = DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch').to(device)
    criterion = BinaryFocalLoss(alpha=1, gamma=2, logits=True).to(device)
    weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]
    lambda_fused = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    out_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt_path = os.path.join(out_dir, "best_model.pt")

    best_val_f1 = -1.0
    best_tau = 0.50
    patience = 12
    patience_cnt = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            s1, s2, s3, s4, s5, fused = model(imgs)

            loss = (
                weight_side[0] * criterion(s1, labels) +
                weight_side[1] * criterion(s2, labels) +
                weight_side[2] * criterion(s3, labels) +
                weight_side[3] * criterion(s4, labels) +
                weight_side[4] * criterion(s5, labels) +
                lambda_fused * criterion(fused, labels)
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_sub)

        # Validation sweep
        model.eval()
        val_probs_all, val_labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                labels = batch['label'].cpu().numpy().flatten()
                _, _, _, _, _, fused = model(imgs)
                probs = torch.sigmoid(fused).cpu().numpy().flatten()
                val_probs_all.append(probs)
                val_labels_all.append(labels)

        val_probs_all = np.concatenate(val_probs_all)
        val_labels_all = np.concatenate(val_labels_all)

        # Quick sweep for validation monitoring
        ep_best_f1, ep_best_tau = 0.0, 0.5
        for tau in np.arange(0.20, 0.85, 0.05):
            preds = (val_probs_all >= tau).astype(int)
            f1 = f1_score(val_labels_all, preds, zero_division=0)
            if f1 > ep_best_f1:
                ep_best_f1 = f1
                ep_best_tau = tau

        scheduler.step(ep_best_f1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val F1: {ep_best_f1:.4f} (tau={ep_best_tau:.2f})")

        if ep_best_f1 > best_val_f1:
            best_val_f1 = ep_best_f1
            best_tau = ep_best_tau
            torch.save(model.state_dict(), best_ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    train_time = time.time() - t0
    print(f"Training finished in {train_time:.1f}s. Best Val F1: {best_val_f1:.4f}")

    # Load best weights
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()

    # 3. Fine-grained threshold calibration on validation set (non-circular)
    val_probs_all, val_labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            labels = batch['label'].cpu().numpy().flatten()
            _, _, _, _, _, fused = model(imgs)
            probs = torch.sigmoid(fused).cpu().numpy().flatten()
            val_probs_all.append(probs)
            val_labels_all.append(labels)

    val_probs_all = np.concatenate(val_probs_all)
    val_labels_all = np.concatenate(val_labels_all)

    calib_tau, calib_f1 = 0.50, 0.0
    for tau in np.arange(0.10, 0.92, 0.02):
        preds = (val_probs_all >= tau).astype(int)
        f1 = f1_score(val_labels_all, preds, zero_division=0)
        if f1 > calib_f1:
            calib_f1 = f1
            calib_tau = float(tau)

    print(f"Calibrated Optimal Tau*: {calib_tau:.2f} (Val F1 = {calib_f1:.4f})")

    # 4. Holdout Benchmark Evaluation on 237 images
    test_ds = DeepCrackImageDataset(TEST_IMG_DIR, TEST_LAB_DIR, augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    test_files = sorted(glob.glob(os.path.join(TEST_GRAPH_DIR, "*.pt")))
    test_mask_dir = TEST_LAB_DIR

    all_node_preds = []
    all_node_labels = []
    all_node_probs = []
    per_graph_f1s = []
    per_image_results = {}

    pixel_f1s, pixel_ious = [], []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            imgs = batch['image'].to(device)
            orig_h, orig_w = int(batch['orig_shape'][0]), int(batch['orig_shape'][1])
            img_id = batch['id'][0]

            _, _, _, _, _, fused = model(imgs)
            # Upsample prediction to full original resolution
            fused_up = F.interpolate(fused, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
            prob_map = torch.sigmoid(fused_up)[0, 0].cpu().numpy()

            # Save latest prediction images if seed 42
            if seed == 42:
                vis_prob = (prob_map * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(SAVE_PREDS_DIR, f"{img_id}_fused.png"), vis_prob)

            # Load ground truth graph
            graph_path = os.path.join(TEST_GRAPH_DIR, f"{img_id}.pt")
            graph_data = torch.load(graph_path, weights_only=False)
            y_true = graph_data.y.cpu().numpy()

            # Superpixel pooling
            sp_path = os.path.join(TEST_SP_DIR, f"{img_id}.npy")
            if os.path.exists(sp_path):
                segments = np.load(sp_path)
            else:
                # Fallback compute if missing
                img_orig = cv2.imread(batch['img_path'][0])
                from src.preprocessing.superpixel import compute_superpixels
                segments = compute_superpixels(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))

            node_probs = pool_dense_prob_to_superpixels(prob_map, segments)
            node_preds = (node_probs >= calib_tau).astype(int)

            all_node_preds.append(node_preds)
            all_node_labels.append(y_true)
            all_node_probs.append(node_probs)

            gf1 = float(f1_score(y_true, node_preds, zero_division=0))
            per_graph_f1s.append(gf1)

            # Pixel metrics
            gt_mask_path = os.path.join(TEST_LAB_DIR, f"{img_id}.png")
            if os.path.exists(gt_mask_path):
                gt_mask = (cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(int)
                pixel_pred = (prob_map >= calib_tau).astype(int)
                px_f1 = f1_score(gt_mask.flatten(), pixel_pred.flatten(), zero_division=0)
                px_iou = compute_iou(pixel_pred.flatten(), gt_mask.flatten())
                pixel_f1s.append(px_f1)
                pixel_ious.append(px_iou)

            cat = classify_crack_geometry(gt_mask_path)
            per_image_results[img_id] = {
                'f1': round(gf1, 4),
                'precision': round(float(precision_score(y_true, node_preds, zero_division=0)), 4),
                'recall': round(float(recall_score(y_true, node_preds, zero_division=0)), 4),
                'category': cat,
                'nodes': len(y_true),
                'defects': int((node_preds == 1).sum())
            }

    flat_preds = np.concatenate(all_node_preds)
    flat_labels = np.concatenate(all_node_labels)
    flat_probs = np.concatenate(all_node_probs)

    holdout_f1 = float(f1_score(flat_labels, flat_preds, zero_division=0))
    holdout_iou = float(compute_iou(flat_preds, flat_labels))
    holdout_prec = float(precision_score(flat_labels, flat_preds, zero_division=0))
    holdout_rec = float(recall_score(flat_labels, flat_preds, zero_division=0))
    holdout_auc = float(roc_auc_score(flat_labels, flat_probs))

    # Morphology buckets
    buckets = ['Thin_Fine_Fissure', 'Long_Elongated', 'Branched_Complex']
    b_metrics = {}
    for b in buckets:
        b_idxs = [i for i, f in enumerate(test_files) if classify_crack_geometry(os.path.join(test_mask_dir, f"{os.path.splitext(os.path.basename(f))[0]}.png")) == b]
        if b_idxs:
            b_p = np.concatenate([all_node_preds[i] for i in b_idxs])
            b_l = np.concatenate([all_node_labels[i] for i in b_idxs])
            b_f1 = float(f1_score(b_l, b_p, zero_division=0))
            b_iou = float(compute_iou(b_p, b_l))
            b_rec = float(recall_score(b_l, b_p, zero_division=0))
            b_prec = float(precision_score(b_l, b_p, zero_division=0))
            b_metrics[b] = {'f1': b_f1, 'iou': b_iou, 'recall': b_rec, 'precision': b_prec, 'n_images': len(b_idxs)}

    res = {
        'model_id': 'DeepCrack_Liu2019',
        'seed': seed,
        'train_time_sec': round(train_time, 1),
        'best_tau': calib_tau,
        'best_val_f1': round(calib_f1, 4),
        'holdout_f1': round(holdout_f1, 4),
        'holdout_iou': round(holdout_iou, 4),
        'holdout_precision': round(holdout_prec, 4),
        'holdout_recall': round(holdout_rec, 4),
        'holdout_roc_auc': round(holdout_auc, 4),
        'pixel_f1_mean': round(float(np.mean(pixel_f1s)), 4) if pixel_f1s else None,
        'pixel_iou_mean': round(float(np.mean(pixel_ious)), 4) if pixel_ious else None,
        'per_graph_f1s': per_graph_f1s,
        'bucket_metrics': b_metrics,
        'per_image_results': per_image_results
    }

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=2)

    print(f"Seed {seed} Result: Node F1 = {holdout_f1:.4f} | IoU = {holdout_iou:.4f} | Prec = {holdout_prec:.4f} | Rec = {holdout_rec:.4f} | AUC = {holdout_auc:.4f}")
    if pixel_f1s:
        print(f"  Pixel-Level: F1 = {np.mean(pixel_f1s):.4f} | IoU = {np.mean(pixel_ious):.4f}")
    return res


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Multi-Seed DeepCrack Training & Benchmark on: {device}")

    all_results = []
    t_start = time.time()

    for seed in SEEDS:
        res = train_single_seed(seed, device, epochs=30, batch_size=4, lr=0.001)
        all_results.append(res)

    total_time = time.time() - t_start
    print(f"\nAll 5 seeds finished in {total_time/60.0:.2f} minutes.")

    # Multi-seed aggregate stats
    f1s = [r['holdout_f1'] for r in all_results]
    ious = [r['holdout_iou'] for r in all_results]
    precs = [r['holdout_precision'] for r in all_results]
    recs = [r['holdout_recall'] for r in all_results]
    aucs = [r['holdout_roc_auc'] for r in all_results]

    summary = {
        'model_name': 'DeepCrack (Liu et al. 2019, Pixel CNN)',
        'f1_mean': float(np.mean(f1s)),
        'f1_std': float(np.std(f1s, ddof=1)),
        'iou_mean': float(np.mean(ious)),
        'iou_std': float(np.std(ious, ddof=1)),
        'precision_mean': float(np.mean(precs)),
        'precision_std': float(np.std(precs, ddof=1)),
        'recall_mean': float(np.mean(recs)),
        'recall_std': float(np.std(recs, ddof=1)),
        'roc_auc_mean': float(np.mean(aucs)),
        'roc_auc_std': float(np.std(aucs, ddof=1)),
        'seed_runs': all_results
    }

    summary_file = os.path.join(RESULTS_DIR, "deepcrack_multiseed_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("DEEPCRACK (LIU ET AL. 2019) 5-SEED MASTER SUMMARY")
    print("="*80)
    print(f"Test F1:        {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=1):.4f}")
    print(f"Test IoU:       {np.mean(ious):.4f} +/- {np.std(ious, ddof=1):.4f}")
    print(f"Precision:      {np.mean(precs):.4f} +/- {np.std(precs, ddof=1):.4f}")
    print(f"Recall:         {np.mean(recs):.4f} +/- {np.std(recs, ddof=1):.4f}")
    print(f"ROC-AUC:        {np.mean(aucs):.4f} +/- {np.std(aucs, ddof=1):.4f}")
    print(f"Saved summary to {summary_file}")


if __name__ == '__main__':
    main()
