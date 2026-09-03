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

def assign_node_labels(segments: np.ndarray, mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Assign binary label to each superpixel based on ground truth mask.
    
    Args:
        segments (np.ndarray): Superpixel labels.
        mask (np.ndarray): Binary ground truth mask (crack pixels).
        threshold (float): Fraction of crack pixels required to label superpixel as crack.
        
    Returns:
        np.ndarray: Binary array of node labels.
    """
    labels = np.unique(segments)
    node_labels = np.zeros(len(labels), dtype=np.int64)
    
    # Ensure mask is boolean
    if mask.max() > 1:
        mask = mask > 127
        
    for i, label in enumerate(labels):
        segment_mask = (segments == label)
        crack_pixels = np.logical_and(segment_mask, mask).sum()
        total_pixels = segment_mask.sum()
        
        if total_pixels > 0 and (crack_pixels / total_pixels) > threshold:
            node_labels[i] = 1
            
    return node_labels

def process_single_image(image_path: str, mask_path: Optional[str], superpixel_dir: str, graph_dir: str, config: dict) -> torch.Any:
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
    node_labels = None
    if mask_path is not None and os.path.exists(mask_path):
        mask = skimage.io.imread(mask_path)
        threshold = config.get('labels', {}).get('threshold', 0.5)
        node_labels = assign_node_labels(segments, mask, threshold=threshold)
        
    # 5. Build PyG Data
    data = build_pyg_data(features, edge_index, edge_weight, centroids, node_labels)
    
    # 6. Save Graph
    os.makedirs(graph_dir, exist_ok=True)
    out_path = os.path.join(graph_dir, f"{image_id}.pt")
    torch.save(data, out_path)
    
    return data

def build_dataset(raw_dir: str, mask_dir: Optional[str], superpixel_dir: str, graph_dir: str, config: dict):
    """
    Processes all images in raw_dir.
    
    Args:
        raw_dir (str): Directory with raw images.
        mask_dir (Optional[str]): Directory with ground truth masks.
        superpixel_dir (str): Directory to cache superpixels.
        graph_dir (str): Directory to output PyG graphs.
        config (dict): Configuration dictionary.
    """
    image_paths = glob.glob(os.path.join(raw_dir, "*.*"))
    # Filter common image extensions
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_paths)} images in {raw_dir}")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        mask_path = None
        if mask_dir is not None:
            # Assuming mask has same filename
            mask_path = os.path.join(mask_dir, filename)
            
        print(f"Processing {filename}...")
        process_single_image(img_path, mask_path, superpixel_dir, graph_dir, config)
        
    print(f"Completed processing {len(image_paths)} images.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Graph Dataset for Pipeline Crack Detection")
    parser.add_argument('--raw_dir', type=str, required=True, help="Directory containing raw images")
    parser.add_argument('--mask_dir', type=str, default=None, help="Directory containing mask images")
    parser.add_argument('--superpixel_dir', type=str, required=True, help="Directory to cache superpixel numpy arrays")
    parser.add_argument('--graph_dir', type=str, required=True, help="Directory to output PyG graphs")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    build_dataset(args.raw_dir, args.mask_dir, args.superpixel_dir, args.graph_dir, config)
