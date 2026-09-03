"""
Smoke test: verifies that all modules import correctly and a synthetic
forward pass through each model variant completes without errors.
"""

import sys
import os

# Ensure the project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np

def test_imports():
    """Test that all modules import successfully."""
    print("Testing imports...")

    from src.preprocessing.superpixel import compute_superpixels
    from src.preprocessing.features import extract_node_features, sobel_magnitude, compute_lbp_histogram
    from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data
    from src.models.gnn_encoder import DeepGNN, ShallowGNN
    from src.models.positional_encoding import sinusoidal_pe, PositionalEncodingFusion
    from src.models.transformer_block import TransformerBlock, TransformerStack
    from src.models.hybrid_model import HybridGNNTransformer, build_model
    from src.utils.metrics import compute_f1, compute_iou, compute_roc_auc, dirichlet_energy, avg_pairwise_cosine_sim

    print("  All imports successful!")


def test_preprocessing_pipeline():
    """Test superpixel → features → PIRM graph on a synthetic image."""
    print("\nTesting preprocessing pipeline...")

    from src.preprocessing.superpixel import compute_superpixels
    from src.preprocessing.features import extract_node_features
    from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data

    # Create a synthetic 64x64 RGB image
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Superpixel segmentation
    segments = compute_superpixels(image, n_segments=20, compactness=10.0, sigma=1.0)
    n_superpixels = len(np.unique(segments))
    print(f"  Superpixels: {n_superpixels} segments")

    # Feature extraction
    features, centroids = extract_node_features(image, segments)
    print(f"  Features shape: {features.shape}")
    print(f"  Centroids shape: {centroids.shape}")
    assert features.shape[0] == n_superpixels
    assert features.shape[1] == 15
    assert centroids.shape == (n_superpixels, 2)

    # PIRM graph
    k = min(8, n_superpixels - 1)
    edge_index, edge_weight = build_pirm_graph(features, k=k, sigma=0.1)
    edge_index, edge_weight = symmetrize_graph(edge_index, edge_weight)
    print(f"  Graph: {edge_index.shape[1]} edges")

    # Create PyG Data
    node_labels = np.random.randint(0, 2, n_superpixels)
    data = build_pyg_data(features, edge_index, edge_weight, centroids, node_labels)
    print(f"  PyG Data: x={data.x.shape}, edge_index={data.edge_index.shape}, pos={data.pos.shape}, y={data.y.shape}")

    print("  Preprocessing pipeline OK!")
    return data


def test_model_forward(data):
    """Test forward pass through all model variants."""
    import torch

    print("\nTesting model forward passes...")
    in_dim = data.x.shape[1]

    # M1: Deep GNN
    from src.models.gnn_encoder import DeepGNN
    model = DeepGNN(in_dim=in_dim, hidden_dim=64, num_layers=8)
    out = model(data.x, data.edge_index)
    assert out.shape == (data.x.shape[0], 2)
    print(f"  DeepGNN (8 layers): output {out.shape} OK")
    assert len(model.layer_outputs) == 8

    # M2: Shallow GNN
    from src.models.gnn_encoder import ShallowGNN
    model = ShallowGNN(in_dim=in_dim, hidden_dim=64, num_layers=2, use_residual=True)
    out = model(data.x, data.edge_index)
    assert out.shape == (data.x.shape[0], 2)
    print(f"  ShallowGNN (2 layers): output {out.shape} OK")
    assert len(model.layer_outputs) == 2

    # M3: Hybrid without PE
    from src.models.hybrid_model import HybridGNNTransformer
    model = HybridGNNTransformer(in_dim=in_dim, hidden_dim=64, gnn_layers=2,
                                  transformer_layers=2, use_pe=False)
    out = model(data.x, data.edge_index, data.pos)
    assert out.shape == (data.x.shape[0], 2)
    print(f"  Hybrid (no PE): output {out.shape} OK")

    # M4: Hybrid with PE (full model)
    model = HybridGNNTransformer(in_dim=in_dim, hidden_dim=64, gnn_layers=2,
                                  transformer_layers=2, use_pe=True)
    out = model(data.x, data.edge_index, data.pos)
    assert out.shape == (data.x.shape[0], 2)
    print(f"  Hybrid (full, with PE): output {out.shape} OK")
    assert len(model.gnn_layer_outputs) == 2

    # Test build_model factory
    from src.models.hybrid_model import build_model
    config = {
        'model': {
            'type': 'hybrid',
            'in_dim': in_dim,
            'hidden_dim': 64,
            'gnn_layers': 2,
            'transformer_layers': 2,
            'heads': 4,
            'pe_dim': 32,
            'num_freqs': 8,
            'dropout': 0.3,
            'transformer_dropout': 0.1,
            'use_pe': True,
        }
    }
    model = build_model(config)
    out = model(data.x, data.edge_index, data.pos)
    assert out.shape == (data.x.shape[0], 2)
    print(f"  build_model factory: output {out.shape} OK")

    print("  All model forward passes OK!")


def test_metrics(data):
    """Test metric computation."""
    import torch
    from src.utils.metrics import compute_f1, compute_iou, dirichlet_energy, avg_pairwise_cosine_sim

    print("\nTesting metrics...")

    preds = np.random.randint(0, 2, data.y.shape[0])
    labels = data.y.numpy()

    f1 = compute_f1(preds, labels)
    iou = compute_iou(preds, labels)
    print(f"  F1: {f1:.4f}, IoU: {iou:.4f}")

    energy = dirichlet_energy(data.x, data.edge_index)
    cos_sim = avg_pairwise_cosine_sim(data.x)
    print(f"  Dirichlet energy: {energy:.4f}, Avg cosine sim: {cos_sim:.4f}")

    print("  Metrics OK!")


if __name__ == '__main__':
    test_imports()
    data = test_preprocessing_pipeline()
    test_model_forward(data)
    test_metrics(data)
    print("\n" + "=" * 50)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 50)
