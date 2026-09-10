import os
import sys
import time
import json
import psutil
import cv2
import numpy as np
import torch
from skimage.segmentation import slic

dcs_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gnn_dir = os.path.join(dcs_root, "pipeline-crack-gnn")
sys.path.insert(0, dcs_root)
sys.path.insert(0, gnn_dir)

from crack_metrics import CrackHoldoutTestDataset, benchmark_end_to_end
from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data
from src.models.hybrid_model import build_model
from src.train import load_config
from src.evaluate import reconstruct_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Initial CPU Frequency: {psutil.cpu_freq().current:.1f} MHz")

# 1. Load M4 model
cfg = load_config(os.path.join(gnn_dir, "configs", "step2_loss", "hybrid_full_focal.yaml"))
m4 = build_model(cfg).to(device)
m4.load_state_dict(torch.load(os.path.join(gnn_dir, "results", "step2_loss", "M4_hybrid_full", "best_model.pt"), map_location=device))
m4.eval()

# 2. Prepare Inputs
dataset = CrackHoldoutTestDataset(
    img_dir=os.path.join(dcs_root, "DeepCrack", "test_img"),
    lab_dir=os.path.join(dcs_root, "DeepCrack", "test_lab")
)
real_img = dataset[0][0]
if real_img.shape[:2] != (256, 256):
    real_img = cv2.resize(real_img, (256, 256))

# Fixed seed for synthetic dummy tensor
np.random.seed(42)
torch.manual_seed(42)
synthetic_img_uint8 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
synthetic_randn_uint8 = ((torch.randn(256, 256, 3).numpy() * 50 + 128).clip(0, 255)).astype(np.uint8)

# Benchmark settings
N_RUNS = 500
N_WARMUP = 15

def run_slic_benchmark(img, label, enforce_conn=False):
    print(f"\n--- Running SLIC Benchmark: {label} (N={N_RUNS}, enforce_conn={enforce_conn}) ---")
    # Warmup
    for _ in range(N_WARMUP):
        _ = slic(img, n_segments=300, compactness=10.0, sigma=1.0, enforce_connectivity=enforce_conn, channel_axis=-1)

    times = []
    freqs = []
    for i in range(N_RUNS):
        if i % 50 == 0:
            freq = psutil.cpu_freq().current
            freqs.append(freq)
        t0 = time.perf_counter()
        _ = slic(img, n_segments=300, compactness=10.0, sigma=1.0, enforce_connectivity=enforce_conn, channel_axis=-1)
        times.append((time.perf_counter() - t0) * 1000.0)

    times_arr = np.array(times)
    blocks = []
    for b in range(10):
        blk_times = times_arr[b*50 : (b+1)*50]
        blocks.append({
            "block": f"{b*50+1}-{(b+1)*50}",
            "median_ms": round(float(np.median(blk_times)), 2),
            "mean_ms": round(float(np.mean(blk_times)), 2),
            "std_ms": round(float(np.std(blk_times, ddof=1)), 2),
            "cpu_freq_mhz": round(freqs[b], 1) if b < len(freqs) else None
        })

    res = {
        "label": label,
        "enforce_connectivity": enforce_conn,
        "n_runs": N_RUNS,
        "median_ms": round(float(np.median(times_arr)), 2),
        "iqr_ms": round(float(np.percentile(times_arr, 75) - np.percentile(times_arr, 25)), 2),
        "mean_ms": round(float(np.mean(times_arr)), 2),
        "std_ms": round(float(np.std(times_arr, ddof=1)), 2),
        "min_ms": round(float(np.min(times_arr)), 2),
        "max_ms": round(float(np.max(times_arr)), 2),
        "drift_block_medians": blocks,
        "start_vs_end_delta_ms": round(float(blocks[-1]["median_ms"] - blocks[0]["median_ms"]), 2)
    }
    print(f"[{label}] Median: {res['median_ms']} ms | IQR: {res['iqr_ms']} ms | Block 1: {blocks[0]['median_ms']} ms -> Block 10: {blocks[-1]['median_ms']} ms (Delta: {res['start_vs_end_delta_ms']} ms)")
    return res

def run_m4_e2e_benchmark(img, label, enforce_conn=False):
    print(f"\n--- Running M4 End-to-End Benchmark: {label} (N={N_RUNS}, enforce_conn={enforce_conn}) ---")
    takes_pos = True
    is_cuda = (device.type == "cuda")

    # Warmup
    for _ in range(N_WARMUP):
        seg = slic(img, n_segments=300, compactness=10.0, sigma=1.0, enforce_connectivity=enforce_conn, channel_axis=-1)
        f, c = extract_node_features(img, seg)
        e, w = build_pirm_graph(f, k=8, sigma=0.1)
        e, w = symmetrize_graph(e, w)
        d = build_pyg_data(f, e, w, c).to(device)
        with torch.no_grad():
            out = m4(d.x, d.edge_index, d.pos)
            probs = torch.softmax(out, dim=1)[:, 1]
            n_preds = (probs >= 0.50).int().cpu().numpy()
        _ = reconstruct_mask(seg, n_preds, seg.shape)
        if is_cuda:
            torch.cuda.synchronize()

    times_slic, times_feat, times_graph, times_fwd, times_recon, times_e2e = [], [], [], [], [], []
    freqs = []

    for i in range(N_RUNS):
        if i % 50 == 0:
            freqs.append(psutil.cpu_freq().current)
        t0 = time.perf_counter()
        seg = slic(img, n_segments=300, compactness=10.0, sigma=1.0, enforce_connectivity=enforce_conn, channel_axis=-1)
        t1 = time.perf_counter()
        f, c = extract_node_features(img, seg)
        t2 = time.perf_counter()
        e, w = build_pirm_graph(f, k=8, sigma=0.1)
        e, w = symmetrize_graph(e, w)
        d = build_pyg_data(f, e, w, c).to(device)
        t3 = time.perf_counter()

        if is_cuda:
            torch.cuda.synchronize()
        t_fwd_0 = time.perf_counter()
        with torch.no_grad():
            out = m4(d.x, d.edge_index, d.pos)
            probs = torch.softmax(out, dim=1)[:, 1]
            n_preds = (probs >= 0.50).int().cpu().numpy()
        if is_cuda:
            torch.cuda.synchronize()
        t_fwd_1 = time.perf_counter()

        _ = reconstruct_mask(seg, n_preds, seg.shape)
        t_end = time.perf_counter()

        times_slic.append((t1 - t0) * 1000.0)
        times_feat.append((t2 - t1) * 1000.0)
        times_graph.append((t3 - t2) * 1000.0)
        times_fwd.append((t_fwd_1 - t_fwd_0) * 1000.0)
        times_recon.append((t_end - t_fwd_1) * 1000.0)
        times_e2e.append((t_end - t0) * 1000.0)

    e2e_arr = np.array(times_e2e)
    blocks = []
    for b in range(10):
        blk_times = e2e_arr[b*50 : (b+1)*50]
        blocks.append({
            "block": f"{b*50+1}-{(b+1)*50}",
            "median_ms": round(float(np.median(blk_times)), 2),
            "mean_ms": round(float(np.mean(blk_times)), 2),
            "cpu_freq_mhz": round(freqs[b], 1) if b < len(freqs) else None
        })

    res = {
        "label": label,
        "enforce_connectivity": enforce_conn,
        "n_runs": N_RUNS,
        "e2e_median_ms": round(float(np.median(e2e_arr)), 2),
        "e2e_iqr_ms": round(float(np.percentile(e2e_arr, 75) - np.percentile(e2e_arr, 25)), 2),
        "e2e_mean_ms": round(float(np.mean(e2e_arr)), 2),
        "e2e_std_ms": round(float(np.std(e2e_arr, ddof=1)), 2),
        "e2e_fps": round(float(1000.0 / np.median(e2e_arr)), 2),
        "slic_median_ms": round(float(np.median(times_slic)), 2),
        "features_median_ms": round(float(np.median(times_feat)), 2),
        "graph_median_ms": round(float(np.median(times_graph)), 2),
        "forward_median_ms": round(float(np.median(times_fwd)), 2),
        "recon_median_ms": round(float(np.median(times_recon)), 2),
        "drift_block_medians": blocks,
        "start_vs_end_delta_ms": round(float(blocks[-1]["median_ms"] - blocks[0]["median_ms"]), 2)
    }
    print(f"[{label}] E2E Median: {res['e2e_median_ms']} ms ({res['e2e_fps']} FPS) | SLIC: {res['slic_median_ms']} ms | Feat: {res['features_median_ms']} ms | Graph: {res['graph_median_ms']} ms | Fwd: {res['forward_median_ms']} ms | Recon: {res['recon_median_ms']} ms")
    print(f"[{label}] Drift: Block 1 = {blocks[0]['median_ms']} ms -> Block 10 = {blocks[-1]['median_ms']} ms (Delta = {res['start_vs_end_delta_ms']} ms)")
    return res

results = {}

# 1. SLIC alone comparison (Synthetic vs Real) under enforce_connectivity=False
results["slic_real_conn_false"] = run_slic_benchmark(real_img, "Real Image (111212-1.jpg)", enforce_conn=False)
results["slic_synthetic_randn_conn_false"] = run_slic_benchmark(synthetic_randn_uint8, "Synthetic Dummy (torch.randn uint8)", enforce_conn=False)
results["slic_synthetic_uniform_conn_false"] = run_slic_benchmark(synthetic_img_uint8, "Synthetic Dummy (uniform uint8)", enforce_conn=False)

# 2. SLIC alone comparison under enforce_connectivity=True (for historical audit)
results["slic_real_conn_true"] = run_slic_benchmark(real_img, "Real Image (111212-1.jpg, conn=True)", enforce_conn=True)
results["slic_synthetic_randn_conn_true"] = run_slic_benchmark(synthetic_randn_uint8, "Synthetic Dummy (torch.randn uint8, conn=True)", enforce_conn=True)

# 3. M4 End-to-End comparison (Synthetic vs Real) under enforce_connectivity=False
results["m4_e2e_real_conn_false"] = run_m4_e2e_benchmark(real_img, "M4 E2E Real Image (111212-1.jpg)", enforce_conn=False)
results["m4_e2e_synthetic_randn_conn_false"] = run_m4_e2e_benchmark(synthetic_randn_uint8, "M4 E2E Synthetic Dummy (torch.randn uint8)", enforce_conn=False)

output_json = os.path.join(dcs_root, "scratch", "controlled_comparison_results.json")
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nAll results saved to: {output_json}")
