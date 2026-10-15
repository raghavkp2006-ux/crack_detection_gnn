import os
import glob
import yaml
import argparse
import numpy as np
import skimage.io
import torch
from typing import Optional

from src.preprocessing.superpixel import compute_and_cache_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data

def assign_node_labels(segments: np.ndarray, mask: np.ndarray, threshold: float = 0.05, min_pixels: int = 5) -> np.ndarray:
    """
    Assign binary label to each superpixel based on ground truth mask.
    Dynamically detects whether crack pixels are encoded as 1/255 or 0/black
    by checking the minority class.
    
    Args:
        segments (np.ndarray): Superpixel labels.
        mask (np.ndarray): Binary ground truth mask.
        threshold (float): Fraction of crack pixels required to label superpixel as crack.
        min_pixels (int): Minimum number of crack pixels to avoid single-pixel noise.
        
    Returns:
        np.ndarray: Binary array of node labels.
    """
    labels = np.unique(segments)
    node_labels = np.zeros(len(labels), dtype=np.int64)
    
    # In crack inspection, defects are always the minority class (< 50% of pixels)
    if (mask > 127).mean() > 0.5:
        # 255 is background, 0 is crack (e.g. TITS dataset)
        crack_mask = (mask < 127)
    else:
        # 0 is background, 255 is crack (e.g. DeepCrack dataset)
        crack_mask = (mask > 127)
        
    for i, label in enumerate(labels):
        segment_mask = (segments == label)
        crack_pixels = np.logical_and(segment_mask, crack_mask).sum()
        total_pixels = segment_mask.sum()
        
        if total_pixels > 0 and crack_pixels >= min_pixels and (crack_pixels / total_pixels) >= threshold:
            node_labels[i] = 1
            
    return node_labels

def process_single_image(image_path: str, mask_path: Optional[str], superpixel_dir: str, graph_dir: str, config: dict):
    """
    Full pipeline to process one image into a graph Data object.
    
    Args:
        image_path (str): Path to image.
        mask_path (Optional[str]): Path to mask (if available).
        superpixel_dir (str): Directory to cache superpixels.
        graph_dir (str): Directory to save PyG graphs.
        config (dict): Configuration dictionary.
        
    Returns:
        Data: PyG Data object.
    """
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Superpixels
    sp_params = config.get('superpixels', {})
    segments = compute_and_cache_superpixels(image_path, superpixel_dir, **sp_params)
    
    image = skimage.io.imread(image_path)
    
    # 2. Features
    features, centroids = extract_node_features(image, segments)
    
    # 3. Build Graph
    graph_params = config.get('graph', {})
    k = graph_params.get('k', 8)
    sigma = graph_params.get('sigma', 0.1)
    edge_index, edge_weight = build_pirm_graph(features, k=k, sigma=sigma)
    edge_index, edge_weight = symmetrize_graph(edge_index, edge_weight)
    
    # 4. Labels
    if mask_path is not None and os.path.exists(mask_path):
        mask = skimage.io.imread(mask_path)
        threshold = config.get('labels', {}).get('threshold', 0.05)
        node_labels = assign_node_labels(segments, mask, threshold=threshold)
    else:
        # If no mask provided (crack-free negative surface image), all nodes are background (0)
        node_labels = np.zeros(len(np.unique(segments)), dtype=np.int64)
        
    # 5. Build PyG Data
    data = build_pyg_data(features, edge_index, edge_weight, centroids, node_labels)
    
    # 6. Save Graph
    os.makedirs(graph_dir, exist_ok=True)
    out_path = os.path.join(graph_dir, f"{image_id}.pt")
    torch.save(data, out_path)
    
    return data

def _worker_wrapper(args):
    img_path, mask_path, superpixel_dir, graph_dir, config, idx, total = args
    filename = os.path.basename(img_path)
    print(f"[{idx+1}/{total}] Processing {filename}...")
    process_single_image(img_path, mask_path, superpixel_dir, graph_dir, config)

def build_dataset(raw_dir: str, mask_dir: Optional[str], superpixel_dir: str, graph_dir: str, config: dict, num_workers: int = 4):
    """
    Processes all images in raw_dir.
    
    Args:
        raw_dir (str): Directory with raw images.
        mask_dir (Optional[str]): Directory with ground truth masks.
        superpixel_dir (str): Directory to cache superpixels.
        graph_dir (str): Directory to output PyG graphs.
        config (dict): Configuration dictionary.
        num_workers (int): Number of parallel workers to use.
    """
    import concurrent.futures

    image_paths = glob.glob(os.path.join(raw_dir, "*.*"))
    # Filter common image extensions
    image_paths = sorted([p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(image_paths)} images in {raw_dir}")
    
    tasks = []
    for idx, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        image_id = os.path.splitext(filename)[0]
        out_path = os.path.join(graph_dir, f"{image_id}.pt")
        if os.path.exists(out_path):
            continue
            
        mask_path = None
        if mask_dir is not None:
            # Try multiple extensions since images and masks may differ
            # (e.g. DeepCrack uses .jpg images and .png masks)
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                candidate = os.path.join(mask_dir, image_id + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
                    
        tasks.append((img_path, mask_path, superpixel_dir, graph_dir, config, idx, len(image_paths)))
    
    print(f"{len(tasks)} images remaining to process ({len(image_paths) - len(tasks)} already done).")
    
    if tasks:
        if num_workers > 1:
            print(f"Using {num_workers} parallel workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                list(executor.map(_worker_wrapper, tasks))
        else:
            for task in tasks:
                _worker_wrapper(task)
                
    print(f"Completed processing {len(image_paths)} images.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Graph Dataset for Pipeline Crack Detection")
    parser.add_argument('--raw_dir', type=str, required=True, help="Directory containing raw images")
    parser.add_argument('--mask_dir', type=str, default=None, help="Directory containing mask images")
    parser.add_argument('--superpixel_dir', type=str, required=True, help="Directory to cache superpixel numpy arrays")
    parser.add_argument('--graph_dir', type=str, required=True, help="Directory to output PyG graphs")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    build_dataset(args.raw_dir, args.mask_dir, args.superpixel_dir, args.graph_dir, config, num_workers=args.num_workers)
