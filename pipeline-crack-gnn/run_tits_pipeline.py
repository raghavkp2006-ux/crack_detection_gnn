"""
End-to-end preprocessing, fine-tuning, and evaluation script for the TITS Crack dataset.

Path: C:\\Users\\ragha\\Downloads\\CrackDataset\\TITS

Steps:
  1. Matches all 66 crack images with ground-truth masks across 5 sensors
     (AIGLE_RN, ESAR, LCMS, LRIS, TEMPEST2) + negative clean surface controls.
  2. Parallel multi-worker preprocessing into SLIC superpixels and PIRM graphs.
  3. Fine-tunes/trains the models further, starting from the best DeepCrack checkpoints.
  4. Evaluates and reports performance on the new dataset.

Usage:
    python run_tits_pipeline.py
"""

import os
import sys
import glob
import time
import json
import yaml
import subprocess
import concurrent.futures
import numpy as np
from skimage.io import imread

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TITS_BASE = r"C:\Users\ragha\Downloads\CrackDataset\TITS"

TITS_SP_DIR = os.path.join(BASE_DIR, "data", "superpixels", "tits")
TITS_GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs", "tits")

CONFIG_PATH = os.path.join(BASE_DIR, "configs", "tits_hybrid_full.yaml")

sys.path.insert(0, BASE_DIR)
from src.preprocessing.build_dataset import process_single_image


def banner(msg: str):
    width = 65
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def collect_tits_pairs(tits_base, num_negative=34):
    """
    Collects image-mask pairs from the 5 sensor folders in TITS.
    Includes all 66 crack images + num_negative clean background images.
    """
    sensors = ['AIGLE_RN', 'ESAR', 'LCMS', 'LRIS', 'TEMPEST2']
    pairs = []
    
    for s in sensors:
        gt_files = glob.glob(os.path.join(tits_base, 'GROUND_TRUTH', s, '*.*'))
        img_files = glob.glob(os.path.join(tits_base, 'IMAGES', s, '*.*'))
        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
        
        for gt in gt_files:
            gt_base = os.path.splitext(os.path.basename(gt))[0]
            core_id = gt_base[3:] if gt_base.startswith('GT_') else gt_base
            for k, v in img_map.items():
                if core_id in k:
                    pairs.append((v, gt))
                    break
                    
    print(f"Matched {len(pairs)} crack image-mask pairs across {len(sensors)} sensors.")
    
    # Collect negative clean images (noGT)
    if num_negative > 0:
        negative_imgs = []
        for s in sensors:
            img_files = glob.glob(os.path.join(tits_base, 'IMAGES', s, '*.*'))
            for img in img_files:
                if 'noGT' in os.path.basename(img):
                    negative_imgs.append((img, None)) # None mask indicates crack-free
                    
        # Select evenly
        np.random.seed(42)
        np.random.shuffle(negative_imgs)
        selected_neg = negative_imgs[:num_negative]
        print(f"Added {len(selected_neg)} clean crack-free negative control images.")
        pairs.extend(selected_neg)
        
    return pairs


def _process_worker(args):
    img_path, mask_path, sp_dir, graph_dir, config, idx, total = args
    img_name = os.path.basename(img_path)
    out_id = os.path.splitext(img_name)[0]
    out_file = os.path.join(graph_dir, f"{out_id}.pt")
    
    if os.path.exists(out_file):
        return out_file
        
    print(f"[{idx+1}/{total}] Preprocessing {img_name}...")
    try:
        process_single_image(img_path, mask_path, sp_dir, graph_dir, config)
        return out_file
    except Exception as e:
        print(f"[ERROR] Failed {img_name}: {e}")
        return None


def preprocess_tits(pairs, config, num_workers=6):
    """Parallel multi-worker graph construction."""
    os.makedirs(TITS_SP_DIR, exist_ok=True)
    os.makedirs(TITS_GRAPH_DIR, exist_ok=True)
    
    tasks = []
    for idx, (img_path, mask_path) in enumerate(pairs):
        img_name = os.path.basename(img_path)
        out_id = os.path.splitext(img_name)[0]
        out_file = os.path.join(TITS_GRAPH_DIR, f"{out_id}.pt")
        if not os.path.exists(out_file):
            tasks.append((img_path, mask_path, TITS_SP_DIR, TITS_GRAPH_DIR, config, idx, len(pairs)))
            
    print(f"Remaining images to process: {len(tasks)} / {len(pairs)} ({len(pairs) - len(tasks)} cached)")
    
    if tasks:
        t0 = time.time()
        print(f"Processing with {num_workers} parallel CPU workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(_process_worker, tasks))
        elapsed = time.time() - t0
        print(f"Preprocessing completed in {elapsed:.1f}s.")
    else:
        print("All TITS graphs already cached.")
        
    total_graphs = len([f for f in os.listdir(TITS_GRAPH_DIR) if f.endswith('.pt')])
    print(f"Total TITS graphs available in {TITS_GRAPH_DIR}: {total_graphs}")
    return total_graphs


def create_tits_configs():
    """Create dedicated YAML configs for TITS dataset."""
    os.makedirs(os.path.join(BASE_DIR, "configs"), exist_ok=True)
    
    # M4: Hybrid Full (Proposed)
    m4_cfg = {
        'model': {
            'type': 'hybrid',
            'in_dim': 15,
            'hidden_dim': 64,
            'gnn_layers': 2,
            'transformer_layers': 2,
            'heads': 4,
            'pe_dim': 32,
            'num_freqs': 8,
            'gnn_dropout': 0.3,
            'transformer_dropout': 0.1,
            'use_pe': True
        },
        'superpixels': {
            'n_segments': 300,
            'compactness': 10.0,
            'sigma': 1.0
        },
        'graph': {
            'k': 8,
            'sigma': 0.1
        },
        'labels': {
            'threshold': 0.05
        },
        'data': {
            'graph_dir': 'data/graphs/tits',
            'train_ratio': 0.70,
            'val_ratio': 0.15
        },
        'training': {
            'epochs': 100,
            'lr': 0.0005,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'grad_clip_norm': 1.0,
            'batch_size': 1,
            'early_stopping_patience': 20,
            'seed': 42
        },
        'output_dir': 'results/tits_M4_hybrid_full'
    }
    
    # M1: Deep GNN Baseline
    m1_cfg = {
        'model': {
            'type': 'deep_gnn',
            'in_dim': 15,
            'hidden_dim': 64,
            'num_layers': 8,
            'dropout': 0.3
        },
        'superpixels': {'n_segments': 300, 'compactness': 10.0, 'sigma': 1.0},
        'graph': {'k': 8, 'sigma': 0.1},
        'labels': {'threshold': 0.05},
        'data': {'graph_dir': 'data/graphs/tits', 'train_ratio': 0.70, 'val_ratio': 0.15},
        'training': {
            'epochs': 100,
            'lr': 0.0005,
            'weight_decay': 0.0005,
            'batch_size': 1,
            'early_stopping_patience': 20,
            'seed': 42
        },
        'output_dir': 'results/tits_M1_deep_gnn'
    }

    with open(os.path.join(BASE_DIR, "configs", "tits_hybrid_full.yaml"), 'w') as f:
        yaml.dump(m4_cfg, f, default_flow_style=False)
        
    with open(os.path.join(BASE_DIR, "configs", "tits_deep_gnn.yaml"), 'w') as f:
        yaml.dump(m1_cfg, f, default_flow_style=False)
        
    return m4_cfg


def train_tits_further(config_path, pretrained_weights=None):
    """Runs training with the pretrained weights."""
    cmd = [sys.executable, "-m", "src.train", "--config", config_path]
    if pretrained_weights and os.path.exists(pretrained_weights):
        cmd.extend(["--weights", pretrained_weights])
        print(f"Continuing training from checkpoint: {pretrained_weights}")
    else:
        print("Training model from scratch...")
        
    t0 = time.time()
    res = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    
    if res.returncode != 0:
        print(f"[ERROR] Training failed with code {res.returncode}")
        return None
    print(f"Training completed successfully in {elapsed:.1f}s.")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    res_path = os.path.join(BASE_DIR, cfg['output_dir'], 'results.json')
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            return json.load(f)
    return None


def main():
    banner("TITS CRACK DATASET — PREPROCESSING & CONTINUED TRAINING")
    print(f"Dataset path: {TITS_BASE}")
    
    # 1. Configs
    config = create_tits_configs()
    
    # 2. Collect dataset pairs
    pairs = collect_tits_pairs(TITS_BASE, num_negative=34)
    
    # 3. Preprocess to graphs
    total_graphs = preprocess_tits(pairs, config, num_workers=6)
    if total_graphs < 10:
        print("[ERROR] Insufficient graphs generated. Check paths.")
        sys.exit(1)
        
    # 4. Train M4 Further (fine-tuning from DeepCrack checkpoint)
    banner("TRAINING M4 (HYBRID FULL WITH PE) ON TITS DATASET")
    m4_cfg_path = os.path.join(BASE_DIR, "configs", "tits_hybrid_full.yaml")
    deepcrack_m4_weights = os.path.join(BASE_DIR, "results", "M4_hybrid_full", "best_model.pt")
    
    m4_results = train_tits_further(m4_cfg_path, pretrained_weights=deepcrack_m4_weights)
    
    # 5. Also train M1 (Deep GNN) for baseline comparison on TITS
    banner("TRAINING M1 (DEEP GNN BASELINE) ON TITS DATASET")
    m1_cfg_path = os.path.join(BASE_DIR, "configs", "tits_deep_gnn.yaml")
    deepcrack_m1_weights = os.path.join(BASE_DIR, "results", "M1_deep_gnn", "best_model.pt")
    m1_results = train_tits_further(m1_cfg_path, pretrained_weights=deepcrack_m1_weights)
    
    # 6. Report summary
    banner("TITS DATASET TRAINING RESULTS SUMMARY")
    print(f"  {'Model':<30} {'Test F1':>10} {'Test IoU':>10} {'Test AUC':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    if m4_results:
        print(f"  {'M4 (Hybrid + PE - Proposed)':<30} {m4_results.get('test_f1', 0):>10.4f} {m4_results.get('test_iou', 0):>10.4f} {m4_results.get('test_roc_auc', 0):>10.4f}")
    if m1_results:
        print(f"  {'M1 (Deep GNN Baseline)':<30} {m1_results.get('test_f1', 0):>10.4f} {m1_results.get('test_iou', 0):>10.4f} {m1_results.get('test_roc_auc', 0):>10.4f}")
        
    # Save comparison
    summary = {}
    if m4_results:
        summary['tits_M4_hybrid_full'] = m4_results
    if m1_results:
        summary['tits_M1_deep_gnn'] = m1_results
    with open(os.path.join(BASE_DIR, "results", "tits_comparison.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to results/tits_comparison.json")


if __name__ == '__main__':
    main()
