"""
Standalone evaluation script for pipeline crack detection models.

Usage:
    python -m src.evaluate --model_path results/M4_hybrid_full/best_model.pt \
                           --config_path configs/hybrid_full.yaml \
                           --test_graph_dir data/graphs
"""

import argparse
import yaml
import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report

from src.models.hybrid_model import build_model
from src.utils.dataset import CrackGraphDataset
from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc


def load_config(path: str) -> dict:
    """Reads YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def reconstruct_mask(segments: np.ndarray, node_preds: np.ndarray, image_shape: tuple) -> np.ndarray:
    """
    Creates full-resolution binary mask from per-node predictions.

    Args:
        segments: Superpixel label array matching the image spatial dimensions.
        node_preds: Binary prediction per superpixel node.
        image_shape: (H, W) of the original image.

    Returns:
        Binary mask of shape (H, W) with crack pixels = 1.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    labels = np.unique(segments)
    for i, label in enumerate(labels):
        if i < len(node_preds) and node_preds[i] == 1:
            mask[segments == label] = 1
    return mask


def _model_forward(model, data, model_type: str) -> torch.Tensor:
    """Dispatch forward call based on model type."""
    if model_type == 'hybrid':
        return model(data.x, data.edge_index, data.pos)
    else:
        return model(data.x, data.edge_index)


def evaluate_model(
    model_path: str,
    config_path: str,
    test_graph_dir: str,
    raw_image_dir: str = None,
    superpixel_dir: str = None,
    output_dir: str = None,
):
    """
    Load a trained model and evaluate on test graphs.

    Optionally reconstructs and saves full-resolution crack masks if
    raw_image_dir and superpixel_dir are provided.
    """
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = CrackGraphDataset(test_graph_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_type = config['model']['type']
    all_preds, all_labels, all_probs = [], [], []

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            out = _model_forward(model, data, model_type)

            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            labels = data.y.cpu().numpy()
            crack_probs = probs[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(crack_probs)

            # Optionally reconstruct masks
            if output_dir and superpixel_dir and raw_image_dir:
                graph_files = sorted(
                    [f for f in os.listdir(test_graph_dir) if f.endswith('.pt')]
                )
                if i < len(graph_files):
                    image_id = os.path.splitext(graph_files[i])[0]
                    sp_path = os.path.join(superpixel_dir, f"{image_id}.npy")
                    if os.path.exists(sp_path):
                        segments = np.load(sp_path)
                        mask = reconstruct_mask(segments, preds, segments.shape)
                        np.save(os.path.join(output_dir, f"{image_id}_pred_mask.npy"), mask)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    f1 = compute_f1(all_preds, all_labels)
    iou = compute_iou(all_preds, all_labels)
    auc = compute_roc_auc(all_probs, all_labels)

    print("\n" + "=" * 50)
    print("TEST EVALUATION REPORT")
    print("=" * 50)
    print(classification_report(all_labels, all_preds, zero_division=0))
    print(f"F1: {f1:.4f} | IoU: {iou:.4f} | ROC-AUC: {auc:.4f}")

    return {'f1': f1, 'iou': iou, 'roc_auc': auc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pipeline crack detection model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model weights')
    parser.add_argument('--config_path', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--test_graph_dir', type=str, required=True, help='Directory of test .pt graphs')
    parser.add_argument('--raw_image_dir', type=str, default=None, help='Directory of raw images (for mask reconstruction)')
    parser.add_argument('--superpixel_dir', type=str, default=None, help='Directory of cached superpixels (for mask reconstruction)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save prediction masks')
    args = parser.parse_args()

    evaluate_model(
        args.model_path,
        args.config_path,
        args.test_graph_dir,
        raw_image_dir=args.raw_image_dir,
        superpixel_dir=args.superpixel_dir,
        output_dir=args.output_dir,
    )
