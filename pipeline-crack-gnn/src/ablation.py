"""
Ablation study script for pipeline crack detection.

Trains all 4 model variants (M1–M4) sequentially, collects metrics,
performs bucketed analysis, and generates comparison outputs.

Usage:
    python -m src.ablation --configs_dir configs --output_dir results/ablation
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


def classify_crack_bucket(mask: np.ndarray) -> str:
    """
    Classify a crack into geometric buckets based on its bounding box.

    Args:
        mask: Binary mask of a single image's crack ground truth.

    Returns:
        One of: 'short/compact', 'long/diagonal', 'lighting-variable'
    """
    crack_pixels = np.argwhere(mask > 0)
    if len(crack_pixels) == 0:
        return 'no_crack'

    y_min, x_min = crack_pixels.min(axis=0)
    y_max, x_max = crack_pixels.max(axis=0)

    height = y_max - y_min + 1
    width = x_max - x_min + 1
    diagonal = np.sqrt(height ** 2 + width ** 2)
    aspect_ratio = max(height, width) / max(min(height, width), 1)

    # Intensity variability along crack
    # (in a real implementation, overlay crack pixels on original image)
    # For now, use geometric heuristics only
    if diagonal < 50 and aspect_ratio < 3.0:
        return 'short/compact'
    elif aspect_ratio > 3.0:
        return 'long/diagonal'
    else:
        return 'short/compact'


MODEL_CONFIGS = {
    'M1_deep_gnn': 'baseline_deep_gnn.yaml',
    'M2_shallow_gnn': 'shallow_gnn.yaml',
    'M3_hybrid_no_pe': 'hybrid_no_pe.yaml',
    'M4_hybrid_full': 'hybrid_full.yaml',
}


def run_ablation(configs_dir: str, output_dir: str):
    """
    Train all 4 model variants sequentially and collect results.

    Reads each model's results.json (produced by train.py) after training.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for model_id, cfg_name in MODEL_CONFIGS.items():
        cfg_path = os.path.join(configs_dir, cfg_name)
        if not os.path.exists(cfg_path):
            print(f"[SKIP] Config not found: {cfg_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Training {model_id} ({cfg_name})")
        print(f"{'=' * 60}")

        # Launch training as subprocess
        cmd = [sys.executable, '-m', 'src.train', '--config', cfg_path]
        ret = subprocess.run(cmd, capture_output=False)

        if ret.returncode != 0:
            print(f"[ERROR] Training failed for {model_id}")
            continue

        # Load results
        import yaml
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        results_path = os.path.join(cfg['output_dir'], 'results.json')

        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                model_results = json.load(f)
            results.append({
                'Model': model_id,
                'F1': model_results.get('test_f1', 0.0),
                'IoU': model_results.get('test_iou', 0.0),
                'ROC-AUC': model_results.get('test_roc_auc', 0.0),
            })
        else:
            print(f"[WARN] No results.json found for {model_id}")

    # --- Summary Table ---
    if results:
        df = pd.DataFrame(results)
        print(f"\n{'=' * 60}")
        print("  ABLATION RESULTS SUMMARY")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))

        csv_path = os.path.join(output_dir, 'ablation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    else:
        print("\n[WARN] No results collected. Check that data is available and training completes.")


def bucketed_analysis(
    test_graph_dir: str,
    superpixel_dir: str,
    mask_dir: str,
    configs_dir: str,
    output_dir: str,
):
    """
    Run per-bucket (short/compact, long/diagonal) F1/IoU analysis for each model.

    This requires trained models and ground-truth masks.
    """
    import yaml
    from src.utils.dataset import CrackGraphDataset
    from src.models.hybrid_model import build_model
    from src.utils.metrics import compute_f1, compute_iou

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import torch, glob

    # Classify each test image into a bucket
    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.*')))
    image_buckets = {}
    for mf in mask_files:
        mask = np.load(mf) if mf.endswith('.npy') else (
            __import__('skimage.io', fromlist=['imread']).imread(mf)
        )
        image_id = os.path.splitext(os.path.basename(mf))[0]
        bucket = classify_crack_bucket(mask)
        image_buckets[image_id] = bucket

    bucket_results = []

    for model_id, cfg_name in MODEL_CONFIGS.items():
        cfg_path = os.path.join(configs_dir, cfg_name)
        if not os.path.exists(cfg_path):
            continue

        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        model_path = os.path.join(cfg['output_dir'], 'best_model.pt')
        if not os.path.exists(model_path):
            print(f"[SKIP] No trained model for {model_id}")
            continue

        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        model_type = cfg['model']['type']

        # Evaluate per-bucket
        for bucket_name in ['short/compact', 'long/diagonal']:
            bucket_preds, bucket_labels = [], []
            dataset = CrackGraphDataset(test_graph_dir)

            for idx in range(len(dataset)):
                data = dataset[idx]
                # Determine image_id from graph file
                graph_files = sorted(os.listdir(test_graph_dir))
                graph_files = [f for f in graph_files if f.endswith('.pt')]
                if idx >= len(graph_files):
                    continue
                image_id = os.path.splitext(graph_files[idx])[0]

                if image_buckets.get(image_id) != bucket_name:
                    continue

                data = data.to(device)
                with torch.no_grad():
                    if model_type == 'hybrid':
                        out = model(data.x, data.edge_index, data.pos)
                    else:
                        out = model(data.x, data.edge_index)
                preds = out.argmax(dim=1).cpu().numpy()
                labels = data.y.cpu().numpy()

                bucket_preds.extend(preds)
                bucket_labels.extend(labels)

            if bucket_preds:
                f1 = compute_f1(np.array(bucket_preds), np.array(bucket_labels))
                iou = compute_iou(np.array(bucket_preds), np.array(bucket_labels))
                bucket_results.append({
                    'Model': model_id,
                    'Bucket': bucket_name,
                    'F1': f1,
                    'IoU': iou,
                    'N_nodes': len(bucket_preds),
                })

    if bucket_results:
        df = pd.DataFrame(bucket_results)
        print(f"\n{'=' * 60}")
        print("  BUCKETED ANALYSIS")
        print(f"{'=' * 60}")
        print(df.to_string(index=False))

        csv_path = os.path.join(output_dir, 'bucketed_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nBucketed results saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation study for crack detection models')
    parser.add_argument('--configs_dir', type=str, default='configs',
                        help='Directory containing YAML config files')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                        help='Directory to save ablation results')
    parser.add_argument('--run_bucketed', action='store_true',
                        help='Also run bucketed analysis (requires masks)')
    parser.add_argument('--test_graph_dir', type=str, default='data/graphs')
    parser.add_argument('--superpixel_dir', type=str, default='data/superpixels')
    parser.add_argument('--mask_dir', type=str, default='data/masks')
    args = parser.parse_args()

    run_ablation(args.configs_dir, args.output_dir)

    if args.run_bucketed:
        bucketed_analysis(
            args.test_graph_dir,
            args.superpixel_dir,
            args.mask_dir,
            args.configs_dir,
            args.output_dir,
        )
