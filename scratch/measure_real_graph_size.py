import os
import sys
import glob
import cv2
import torch
import numpy as np

dcs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gnn_dir = os.path.join(dcs_root, 'pipeline-crack-gnn')
if gnn_dir not in sys.path:
    sys.path.insert(0, gnn_dir)

from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data

def main():
    img_dir = os.path.join(dcs_root, 'DeepCrack', 'test_img')
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    print(f'Found {len(img_files)} holdout images in {img_dir}')

    # Check precomputed graphs
    precomp_dir = os.path.join(gnn_dir, 'data', 'graphs', 'test')
    precomp_files = sorted(glob.glob(os.path.join(precomp_dir, '*.pt')))
    print(f'Found {len(precomp_files)} precomputed graphs in {precomp_dir}')

    if precomp_files:
        pre_nodes, pre_edges = [], []
        for pf in precomp_files:
            g = torch.load(pf, weights_only=False)
            pre_nodes.append(g.num_nodes)
            pre_edges.append(g.num_edges)
        print('--- Precomputed Graph Stats (pipeline-crack-gnn/data/graphs/test) ---')
        print(f'Total nodes across 237 graphs: {sum(pre_nodes)}')
        print(f'Nodes: Mean={np.mean(pre_nodes):.2f}, Median={np.median(pre_nodes):.2f}, Min={np.min(pre_nodes)}, Max={np.max(pre_nodes)}, Std={np.std(pre_nodes):.2f}')
        print(f'Edges: Mean={np.mean(pre_edges):.2f}, Median={np.median(pre_edges):.2f}, Min={np.min(pre_edges)}, Max={np.max(pre_edges)}, Std={np.std(pre_edges):.2f}')

    # On-the-fly under enforce_connectivity=False
    otf_nodes, otf_edges = [], []
    conn_nodes, conn_edges = [], []

    for idx, fpath in enumerate(img_files):
        img_bgr = cv2.imread(fpath)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

        # enforce_connectivity=False
        seg_false = compute_superpixels(img_256, n_segments=300, compactness=10.0, sigma=1.0, enforce_connectivity=False)
        feats_f, c_f = extract_node_features(img_256, seg_false)
        e_f, w_f = build_pirm_graph(feats_f, k=8, sigma=0.1)
        e_f, w_f = symmetrize_graph(e_f, w_f)
        d_f = build_pyg_data(feats_f, e_f, w_f, c_f)
        otf_nodes.append(d_f.num_nodes)
        otf_edges.append(d_f.num_edges)

        # enforce_connectivity=True
        seg_true = compute_superpixels(img_256, n_segments=300, compactness=10.0, sigma=1.0, enforce_connectivity=True)
        feats_t, c_t = extract_node_features(img_256, seg_true)
        e_t, w_t = build_pirm_graph(feats_t, k=8, sigma=0.1)
        e_t, w_t = symmetrize_graph(e_t, w_t)
        d_t = build_pyg_data(feats_t, e_t, w_t, c_t)
        conn_nodes.append(d_t.num_nodes)
        conn_edges.append(d_t.num_edges)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(img_files):
            print(f'Processed {idx + 1}/{len(img_files)} images...')

    print('\n' + '=' * 70)
    print('--- REAL GRAPH STATS: enforce_connectivity=False (Current Config) ---')
    print('=' * 70)
    print(f'Total nodes across 237 images: {sum(otf_nodes)}')
    print(f'Nodes: Mean={np.mean(otf_nodes):.2f}, Median={np.median(otf_nodes):.2f}, Min={np.min(otf_nodes)}, Max={np.max(otf_nodes)}, Std={np.std(otf_nodes):.2f}')
    print(f'Edges: Mean={np.mean(otf_edges):.2f}, Median={np.median(otf_edges):.2f}, Min={np.min(otf_edges)}, Max={np.max(otf_edges)}, Std={np.std(otf_edges):.2f}')
    print(f'IQR Nodes: {np.percentile(otf_nodes, 75) - np.percentile(otf_nodes, 25):.2f} (25th: {np.percentile(otf_nodes, 25):.2f}, 75th: {np.percentile(otf_nodes, 75):.2f})')
    print(f'IQR Edges: {np.percentile(otf_edges, 75) - np.percentile(otf_edges, 25):.2f} (25th: {np.percentile(otf_edges, 25):.2f}, 75th: {np.percentile(otf_edges, 75):.2f})')

    print('\n' + '=' * 70)
    print('--- REAL GRAPH STATS: enforce_connectivity=True (Legacy Config) ---')
    print('=' * 70)
    print(f'Total nodes across 237 images: {sum(conn_nodes)}')
    print(f'Nodes: Mean={np.mean(conn_nodes):.2f}, Median={np.median(conn_nodes):.2f}, Min={np.min(conn_nodes)}, Max={np.max(conn_nodes)}, Std={np.std(conn_nodes):.2f}')
    print(f'Edges: Mean={np.mean(conn_edges):.2f}, Median={np.median(conn_edges):.2f}, Min={np.min(conn_edges)}, Max={np.max(conn_edges)}, Std={np.std(conn_edges):.2f}')

if __name__ == '__main__':
    main()
