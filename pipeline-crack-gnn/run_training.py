"""
Master training script for DeepCrack dataset.

Runs the full pipeline:
  1. Preprocess images -> superpixels -> features -> PIRM graphs
  2. Train all 4 model variants (M1-M4) sequentially

Usage:
    python run_training.py
"""

import os
import sys
import time
import json
import yaml
import subprocess


# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEEPCRACK_DIR = os.path.join(os.path.dirname(BASE_DIR), "DeepCrack")

TRAIN_IMG_DIR = os.path.join(DEEPCRACK_DIR, "train_img")
TRAIN_LAB_DIR = os.path.join(DEEPCRACK_DIR, "train_lab")
TEST_IMG_DIR = os.path.join(DEEPCRACK_DIR, "test_img")
TEST_LAB_DIR = os.path.join(DEEPCRACK_DIR, "test_lab")

TRAIN_SP_DIR = os.path.join(BASE_DIR, "data", "superpixels", "train")
TEST_SP_DIR = os.path.join(BASE_DIR, "data", "superpixels", "test")
TRAIN_GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs", "train")
TEST_GRAPH_DIR = os.path.join(BASE_DIR, "data", "graphs", "test")

CONFIG_DIR = os.path.join(BASE_DIR, "configs")

MODELS_TO_TRAIN = [
    ("M1_deep_gnn", "baseline_deep_gnn.yaml"),
    ("M2_shallow_gnn", "shallow_gnn.yaml"),
    ("M3_hybrid_no_pe", "hybrid_no_pe.yaml"),
    ("M4_hybrid_full", "hybrid_full.yaml"),
]


def banner(msg: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def preprocess_split(img_dir, mask_dir, sp_dir, graph_dir, config_path, num_workers=6):
    """Preprocess one split (train or test) into PyG graphs."""
    banner(f"Preprocessing: {os.path.basename(img_dir)}")
    cmd = [
        sys.executable, "-m", "src.preprocessing.build_dataset",
        "--raw_dir", img_dir,
        "--mask_dir", mask_dir,
        "--superpixel_dir", sp_dir,
        "--graph_dir", graph_dir,
        "--config", config_path,
        "--num_workers", str(num_workers),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] Preprocessing failed (exit code {result.returncode})")
        return False
    print(f"  Completed in {elapsed:.1f}s")
    return True


def count_graphs(graph_dir):
    """Count .pt files in a directory."""
    if not os.path.exists(graph_dir):
        return 0
    return len([f for f in os.listdir(graph_dir) if f.endswith('.pt')])


def train_model(config_name, config_path):
    """Train a single model variant."""
    cmd = [sys.executable, "-m", "src.train", "--config", config_path]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] Training failed (exit code {result.returncode})")
        return None
    print(f"  Training completed in {elapsed:.1f}s")

    # Read results
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    results_path = os.path.join(BASE_DIR, cfg['output_dir'], 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def main():
    banner("PIPELINE CRACK DETECTION - FULL TRAINING RUN")
    print(f"  Dataset:  DeepCrack")
    print(f"  Train:    {TRAIN_IMG_DIR}")
    print(f"  Test:     {TEST_IMG_DIR}")

    # Use hybrid_full config for preprocessing (all configs share the same preprocessing params)
    preprocess_config = os.path.join(CONFIG_DIR, "hybrid_full.yaml")

    # ---- Step 1: Preprocess ----
    train_count = count_graphs(TRAIN_GRAPH_DIR)
    test_count = count_graphs(TEST_GRAPH_DIR)

    if train_count >= 280:  # allow for some tolerance
        print(f"\n  Train graphs already exist ({train_count} files), skipping preprocessing.")
    else:
        ok = preprocess_split(TRAIN_IMG_DIR, TRAIN_LAB_DIR, TRAIN_SP_DIR, TRAIN_GRAPH_DIR, preprocess_config)
        if not ok:
            print("Aborting: preprocessing failed.")
            sys.exit(1)
        train_count = count_graphs(TRAIN_GRAPH_DIR)

    if test_count >= 220:
        print(f"  Test graphs already exist ({test_count} files), skipping preprocessing.")
    else:
        ok = preprocess_split(TEST_IMG_DIR, TEST_LAB_DIR, TEST_SP_DIR, TEST_GRAPH_DIR, preprocess_config)
        if not ok:
            print("Aborting: preprocessing failed.")
            sys.exit(1)
        test_count = count_graphs(TEST_GRAPH_DIR)

    print(f"\n  Graph dataset ready: {train_count} train, {test_count} test")

    # ---- Step 2: Train all models ----
    all_results = {}

    for model_id, config_name in MODELS_TO_TRAIN:
        config_path = os.path.join(CONFIG_DIR, config_name)
        if not os.path.exists(config_path):
            print(f"\n  [SKIP] Config not found: {config_name}")
            continue

        banner(f"Training {model_id}")
        results = train_model(model_id, config_path)
        if results:
            all_results[model_id] = results
            print(f"  F1: {results.get('test_f1', 0):.4f}  |  "
                  f"IoU: {results.get('test_iou', 0):.4f}  |  "
                  f"AUC: {results.get('test_roc_auc', 0):.4f}")

    # ---- Step 3: Summary ----
    if all_results:
        banner("FINAL RESULTS COMPARISON")
        print(f"  {'Model':<25} {'F1':>8} {'IoU':>8} {'AUC':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
        for model_id, r in all_results.items():
            print(f"  {model_id:<25} {r.get('test_f1',0):>8.4f} "
                  f"{r.get('test_iou',0):>8.4f} {r.get('test_roc_auc',0):>8.4f}")

        # Save combined results
        os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
        with open(os.path.join(BASE_DIR, "results", "comparison.json"), 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined results saved to results/comparison.json")
    else:
        print("\n  No results collected. Check errors above.")


if __name__ == '__main__':
    main()
