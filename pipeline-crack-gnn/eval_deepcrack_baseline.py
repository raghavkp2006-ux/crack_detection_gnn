"""
Standalone Reproducible Evaluation of DeepCrack (Liu et al. 2019) Baseline.

WHY THIS SCRIPT EXISTS
======================
The DeepCrack CNN baseline was originally trained AND evaluated in a single
script (train_deepcrack_multiseed.py). That design bundles re-evaluation with
retraining, making it impossible to reproduce the reported numbers without a
full GPU training run. This script extracts the evaluation-only path so that
anyone can:

  1. Load a pre-trained DeepCrack checkpoint.
  2. Run inference on the 237 holdout test images.
  3. Produce node-level F1/IoU/precision/recall/AUC that are directly
     comparable to M1-M4's numbers.
  4. Cross-validate the output against the committed multiseed results.

CRITICAL METHODOLOGICAL NOTE — SUPERPIXEL AGGREGATION
=====================================================
M1-M4 (GNN models) operate on superpixel graphs: each image is partitioned
into ~300 SLIC superpixels, and predictions are made per-node (per-superpixel).
DeepCrack is a dense pixel-level CNN that outputs a probability map at full
image resolution (H x W). To produce an apples-to-apples comparison, this
script performs **superpixel mean-pooling**: for each superpixel segment, the
mean of DeepCrack's predicted pixel probabilities within that segment becomes
the node-level probability. This pooled probability is then thresholded at
tau* (calibrated on the validation set, not the test set) to produce a binary
node prediction.

Without this aggregation step, DeepCrack's metrics would be computed in a
fundamentally different prediction space (65,536 pixels vs ~300 nodes), making
any comparison with M1-M4 invalid.

The aggregation was already correctly implemented in the original
train_deepcrack_multiseed.py via pool_dense_prob_to_superpixels(). This script
reuses the identical logic and verifies numerical consistency.

USAGE
=====
  # Evaluate all 5 seeds (default, for full reproducibility):
  python eval_deepcrack_baseline.py

  # Evaluate a single seed:
  python eval_deepcrack_baseline.py --seeds 42

  # Skip cross-validation against committed results:
  python eval_deepcrack_baseline.py --no-verify

OUTPUT
======
  results/deepcrack_baseline_evaluation.json
"""

import os
import sys
import glob
import json
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DCS_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEEPSEG_DIR = os.path.join(DCS_ROOT, "DeepSegmentor")
sys.path.insert(0, DEEPSEG_DIR)
sys.path.insert(0, BASE_DIR)

from models.deepcrack_networks import DeepCrackNet
from src.utils.metrics import compute_iou

# Paths
MULTISEED_DIR = os.path.join(BASE_DIR, "results", "multiseed", "deepcrack")
TEST_IMG_DIR = os.path.join(DCS_ROOT, "DeepCrack", "test_img")
TEST_LAB_DIR = os.path.join(DCS_ROOT, "DeepCrack", "test_lab")
TEST_SP_DIR = os.path.join(BASE_DIR, "data", "superpixels", "test")
TEST_GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs", "test")
TRAIN_IMG_DIR = os.path.join(DCS_ROOT, "DeepCrack", "train_img")
TRAIN_LAB_DIR = os.path.join(DCS_ROOT, "DeepCrack", "train_lab")

OUTPUT_PATH = os.path.join(BASE_DIR, "results", "deepcrack_baseline_evaluation.json")


def pool_dense_prob_to_superpixels(prob_map, segments):
    """
    Mean-pool a dense 2D probability map into per-superpixel-node probabilities.

    This is the critical aggregation step that converts DeepCrack's pixel-level
    output into the same prediction space used by M1-M4 (superpixel graph nodes).

    For each superpixel segment i, the node probability is:
        p_i = mean( prob_map[pixels in segment i] )

    This matches the implementation in train_deepcrack_multiseed.py exactly.

    Args:
        prob_map: Dense probability map of shape (H, W), values in [0, 1].
        segments: SLIC superpixel label map of shape (H, W), integer-valued.

    Returns:
        node_probs: Array of shape (n_superpixels,) with one aggregated
                    probability per node.
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


def classify_crack_geometry(mask_path):
    """Classify crack morphology from ground truth mask into one of three buckets."""
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


def calibrate_threshold_on_val(model, device, seed):
    """
    Non-circular threshold calibration on the validation split.

    Uses the same get_splits() function and seed as M1-M4, so the validation
    images are identical across all models. Threshold is swept in [0.10, 0.90]
    with step 0.02, and the tau* that maximizes pixel-level F1 on the
    validation set is selected.
    """
    from src.utils.dataset import CrackGraphDataset, get_splits

    # Get validation image indices (from the 300-image training pool)
    graph_ds = CrackGraphDataset('data/graphs/train')
    _, val_idx, _ = get_splits(graph_ds, 0.7, 0.15, seed)

    # Map val indices to image files
    train_img_files = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")) +
                              glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))

    model.eval()
    val_probs_all, val_labels_all = [], []

    with torch.no_grad():
        for idx in val_idx:
            img_path = train_img_files[idx]
            basename = os.path.splitext(os.path.basename(img_path))[0]
            lab_path = os.path.join(TRAIN_LAB_DIR, f"{basename}.png")
            if not os.path.exists(lab_path):
                lab_path = os.path.join(TRAIN_LAB_DIR, f"{basename}.jpg")

            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0).to(device)

            if os.path.exists(lab_path):
                mask = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 127).astype(np.float32)
                mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = np.zeros((256, 256), dtype=np.float32)

            _, _, _, _, _, fused = model(img_tensor)
            probs = torch.sigmoid(fused).cpu().numpy().flatten()
            labels = mask_resized.flatten()

            val_probs_all.append(probs)
            val_labels_all.append(labels)

    val_probs_all = np.concatenate(val_probs_all)
    val_labels_all = np.concatenate(val_labels_all)

    best_tau, best_f1 = 0.50, 0.0
    for tau in np.arange(0.10, 0.92, 0.02):
        preds = (val_probs_all >= tau).astype(int)
        f1 = f1_score(val_labels_all, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    return best_tau, best_f1


def evaluate_single_seed(seed, device):
    """
    Evaluate a single DeepCrack checkpoint on the 237 holdout test images.

    Returns a dict with holdout metrics (node-level and pixel-level),
    per-graph F1s, and morphology-bucketed metrics.
    """
    ckpt_path = os.path.join(MULTISEED_DIR, f"seed_{seed}", "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return None

    print(f"\n{'='*70}")
    print(f"  Evaluating DeepCrack seed {seed}")
    print(f"{'='*70}")

    # Load model
    model = DeepCrackNet(in_nc=3, num_classes=1, ngf=64, norm='batch').to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Calibrate threshold on validation set
    calib_tau, calib_f1 = calibrate_threshold_on_val(model, device, seed)
    print(f"  Calibrated tau*: {calib_tau:.2f} (Val F1 = {calib_f1:.4f})")

    # Collect test image files
    test_img_files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")) +
                             glob.glob(os.path.join(TEST_IMG_DIR, "*.png")))

    all_node_preds = []
    all_node_labels = []
    all_node_probs = []
    per_graph_f1s = []
    pixel_f1s = []
    pixel_ious = []

    with torch.no_grad():
        for img_path in test_img_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]

            # Read and preprocess image
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            img_resized = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Forward pass
            _, _, _, _, _, fused = model(img_tensor)
            fused_up = F.interpolate(fused, size=(orig_h, orig_w),
                                     mode='bilinear', align_corners=True)
            prob_map = torch.sigmoid(fused_up)[0, 0].cpu().numpy()

            # Load ground truth graph for node-level labels
            graph_path = os.path.join(TEST_GRAPH_DIR, f"{basename}.pt")
            if not os.path.exists(graph_path):
                continue
            graph_data = torch.load(graph_path, weights_only=False)
            y_true = graph_data.y.cpu().numpy()

            # Superpixel aggregation (the critical step for fair comparison)
            sp_path = os.path.join(TEST_SP_DIR, f"{basename}.npy")
            if os.path.exists(sp_path):
                segments = np.load(sp_path)
            else:
                from src.preprocessing.superpixel import compute_superpixels
                segments = compute_superpixels(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            node_probs = pool_dense_prob_to_superpixels(prob_map, segments)
            node_preds = (node_probs >= calib_tau).astype(int)

            all_node_preds.append(node_preds)
            all_node_labels.append(y_true)
            all_node_probs.append(node_probs)

            gf1 = float(f1_score(y_true, node_preds, zero_division=0))
            per_graph_f1s.append(gf1)

            # Pixel-level metrics (secondary, for reference)
            gt_mask_path = os.path.join(TEST_LAB_DIR, f"{basename}.png")
            if os.path.exists(gt_mask_path):
                gt_mask = (cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(int)
                pixel_pred = (prob_map >= calib_tau).astype(int)
                px_f1 = f1_score(gt_mask.flatten(), pixel_pred.flatten(), zero_division=0)
                px_iou = compute_iou(pixel_pred.flatten(), gt_mask.flatten())
                pixel_f1s.append(px_f1)
                pixel_ious.append(px_iou)

    # Aggregate node-level metrics (primary — same as M1-M4)
    flat_preds = np.concatenate(all_node_preds)
    flat_labels = np.concatenate(all_node_labels)
    flat_probs = np.concatenate(all_node_probs)

    holdout_f1 = float(f1_score(flat_labels, flat_preds, zero_division=0))
    holdout_iou = float(compute_iou(flat_preds, flat_labels))
    holdout_prec = float(precision_score(flat_labels, flat_preds, zero_division=0))
    holdout_rec = float(recall_score(flat_labels, flat_preds, zero_division=0))
    holdout_auc = float(roc_auc_score(flat_labels, flat_probs))

    print(f"  Node-Level:  F1={holdout_f1:.4f}  IoU={holdout_iou:.4f}  "
          f"Prec={holdout_prec:.4f}  Rec={holdout_rec:.4f}  AUC={holdout_auc:.4f}")
    if pixel_f1s:
        print(f"  Pixel-Level: F1={np.mean(pixel_f1s):.4f}  IoU={np.mean(pixel_ious):.4f}")
    print(f"  Test images: {len(per_graph_f1s)}  |  "
          f"Total nodes: {len(flat_labels)}  |  "
          f"Crack nodes: {int(flat_labels.sum())}")

    # Morphology-bucketed metrics
    test_graph_files = sorted(glob.glob(os.path.join(TEST_GRAPH_DIR, "*.pt")))
    buckets = ['Thin_Fine_Fissure', 'Long_Elongated', 'Branched_Complex']
    bucket_metrics = {}
    for b in buckets:
        b_idxs = [i for i, f in enumerate(test_graph_files)
                  if classify_crack_geometry(
                      os.path.join(TEST_LAB_DIR,
                                   f"{os.path.splitext(os.path.basename(f))[0]}.png")) == b]
        if b_idxs:
            b_p = np.concatenate([all_node_preds[i] for i in b_idxs])
            b_l = np.concatenate([all_node_labels[i] for i in b_idxs])
            bucket_metrics[b] = {
                'f1': round(float(f1_score(b_l, b_p, zero_division=0)), 4),
                'iou': round(float(compute_iou(b_p, b_l)), 4),
                'precision': round(float(precision_score(b_l, b_p, zero_division=0)), 4),
                'recall': round(float(recall_score(b_l, b_p, zero_division=0)), 4),
                'n_images': len(b_idxs)
            }

    return {
        'seed': seed,
        'best_tau': calib_tau,
        'best_val_f1': round(calib_f1, 4),
        'holdout_f1': round(holdout_f1, 4),
        'holdout_iou': round(holdout_iou, 4),
        'holdout_precision': round(holdout_prec, 4),
        'holdout_recall': round(holdout_rec, 4),
        'holdout_roc_auc': round(holdout_auc, 4),
        'pixel_f1_mean': round(float(np.mean(pixel_f1s)), 4) if pixel_f1s else None,
        'pixel_iou_mean': round(float(np.mean(pixel_ious)), 4) if pixel_ious else None,
        'n_test_images': len(per_graph_f1s),
        'n_test_nodes': len(flat_labels),
        'per_graph_f1s': per_graph_f1s,
        'bucket_metrics': bucket_metrics
    }


def verify_against_committed(seed_results, tolerance=1e-3):
    """
    Cross-check evaluation results against the committed multiseed JSONs.

    Returns True if all metrics match within tolerance, False otherwise.
    Prints detailed diagnostics for any mismatches.
    """
    print("\n" + "=" * 70)
    print("  CROSS-VALIDATION AGAINST COMMITTED RESULTS")
    print("=" * 70)

    all_match = True
    for res in seed_results:
        seed = res['seed']
        committed_path = os.path.join(MULTISEED_DIR, f"seed_{seed}", "result.json")
        if not os.path.exists(committed_path):
            print(f"  [WARN] No committed result for seed {seed}, skipping verification.")
            continue

        with open(committed_path, 'r') as f:
            committed = json.load(f)

        metrics = ['holdout_f1', 'holdout_iou', 'holdout_precision',
                    'holdout_recall', 'holdout_roc_auc']
        seed_ok = True
        for m in metrics:
            expected = committed.get(m)
            actual = res.get(m)
            if expected is not None and actual is not None:
                diff = abs(expected - actual)
                status = "OK" if diff <= tolerance else "MISMATCH"
                if diff > tolerance:
                    seed_ok = False
                    all_match = False
                print(f"  Seed {seed} | {m:25s}: expected={expected:.4f}  "
                      f"actual={actual:.4f}  diff={diff:.6f}  {status}")

        if seed_ok:
            print(f"  Seed {seed}: ALL METRICS MATCH (tolerance={tolerance})")

    if all_match:
        print("\n  [PASS] ALL SEEDS VERIFIED -- output is consistent with committed results.")
    else:
        print("\n  [FAIL] VERIFICATION FAILED -- some metrics differ beyond tolerance.")
        print("    This may indicate environment differences (GPU vs CPU, library versions).")
        print("    Review the diffs above carefully.")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Standalone reproducible evaluation of DeepCrack baseline.",
        epilog="See script docstring for full methodology notes."
    )
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 2026],
                        help='Seeds to evaluate (default: all 5)')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip cross-validation against committed results')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH,
                        help='Output JSON path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DeepCrack Baseline Evaluation -- Device: {device}")
    print(f"Seeds: {args.seeds}")

    # Evaluate each seed
    seed_results = []
    for seed in args.seeds:
        res = evaluate_single_seed(seed, device)
        if res is not None:
            seed_results.append(res)

    if not seed_results:
        print("ERROR: No seeds were evaluated successfully.")
        sys.exit(1)

    # Compute aggregate statistics
    f1s = [r['holdout_f1'] for r in seed_results]
    ious = [r['holdout_iou'] for r in seed_results]
    precs = [r['holdout_precision'] for r in seed_results]
    recs = [r['holdout_recall'] for r in seed_results]
    aucs = [r['holdout_roc_auc'] for r in seed_results]

    aggregate = {
        'f1_mean': round(float(np.mean(f1s)), 5),
        'f1_std': round(float(np.std(f1s, ddof=1)), 5) if len(f1s) > 1 else 0.0,
        'iou_mean': round(float(np.mean(ious)), 5),
        'iou_std': round(float(np.std(ious, ddof=1)), 5) if len(ious) > 1 else 0.0,
        'precision_mean': round(float(np.mean(precs)), 5),
        'precision_std': round(float(np.std(precs, ddof=1)), 5) if len(precs) > 1 else 0.0,
        'recall_mean': round(float(np.mean(recs)), 5),
        'recall_std': round(float(np.std(recs, ddof=1)), 5) if len(recs) > 1 else 0.0,
        'roc_auc_mean': round(float(np.mean(aucs)), 5),
        'roc_auc_std': round(float(np.std(aucs, ddof=1)), 5) if len(aucs) > 1 else 0.0,
    }

    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS")
    print("=" * 70)
    print(f"  F1:        {aggregate['f1_mean']:.5f} ± {aggregate['f1_std']:.5f}")
    print(f"  IoU:       {aggregate['iou_mean']:.5f} ± {aggregate['iou_std']:.5f}")
    print(f"  Precision: {aggregate['precision_mean']:.5f} ± {aggregate['precision_std']:.5f}")
    print(f"  Recall:    {aggregate['recall_mean']:.5f} ± {aggregate['recall_std']:.5f}")
    print(f"  ROC-AUC:   {aggregate['roc_auc_mean']:.5f} ± {aggregate['roc_auc_std']:.5f}")

    # Build output
    output = {
        'script': 'eval_deepcrack_baseline.py',
        'description': (
            'Standalone reproducible evaluation of DeepCrack (Liu et al. 2019) '
            'baseline. Predictions are mean-pooled from dense pixel probabilities '
            'onto the same SLIC superpixel graph used by M1-M4, ensuring '
            'apples-to-apples metric comparison at node (superpixel) granularity.'
        ),
        'methodology': {
            'aggregation': 'mean-pooling of pixel probabilities within each SLIC superpixel segment',
            'threshold_calibration': 'non-circular sweep on validation split (same split as M1-M4)',
            'metric_functions': 'src/utils/metrics.py (same as M1-M4)',
            'test_set': '237 holdout images, 67,998 superpixel nodes',
            'superpixel_params': {
                'method': 'SLIC (skimage.segmentation.slic)',
                'n_segments': 300,
                'compactness': 10.0,
                'sigma': 1.0
            }
        },
        'aggregate': aggregate,
        'seed_results': seed_results
    }

    # Cross-validate
    verified = True
    if not args.no_verify:
        verified = verify_against_committed(seed_results)
    output['verification'] = {
        'cross_validated': not args.no_verify,
        'all_match': verified
    }

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved evaluation results to {args.output}")


if __name__ == '__main__':
    main()
