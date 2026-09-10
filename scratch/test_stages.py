import sys, os, time, cv2, torch
import numpy as np

dcs_root = os.path.dirname(os.path.abspath('crack_metrics.py'))
gnn_dir = os.path.join(dcs_root, 'pipeline-crack-gnn')
sys.path.insert(0, dcs_root)
sys.path.insert(0, gnn_dir)

from crack_metrics import CrackHoldoutTestDataset
from src.preprocessing.superpixel import compute_superpixels
from src.preprocessing.features import extract_node_features
from src.preprocessing.pirm_graph import build_pirm_graph, symmetrize_graph, build_pyg_data

print("torch.get_num_threads():", torch.get_num_threads())
print("cv2.getNumThreads():", cv2.getNumThreads())
print("os.cpu_count():", os.cpu_count())

dataset = CrackHoldoutTestDataset(
    img_dir=os.path.join(dcs_root, 'DeepCrack', 'test_img'),
    lab_dir=os.path.join(dcs_root, 'DeepCrack', 'test_lab')
)
sample_img = dataset[0][0]
if sample_img.shape[:2] != (256, 256):
    sample_img = cv2.resize(sample_img, (256, 256))

# Let's time 50 runs of SLIC alone, features alone, graph alone
times_slic = []
times_feat = []
times_graph = []

for _ in range(10):
    s = compute_superpixels(sample_img, n_segments=300, compactness=10.0, sigma=1.0)
    f, c = extract_node_features(sample_img, s)
    e, w = build_pirm_graph(f, k=8, sigma=0.1)

for _ in range(50):
    t0 = time.perf_counter()
    s = compute_superpixels(sample_img, n_segments=300, compactness=10.0, sigma=1.0)
    t1 = time.perf_counter()
    f, c = extract_node_features(sample_img, s)
    t2 = time.perf_counter()
    e, w = build_pirm_graph(f, k=8, sigma=0.1)
    t3 = time.perf_counter()
    times_slic.append((t1 - t0)*1000)
    times_feat.append((t2 - t1)*1000)
    times_graph.append((t3 - t2)*1000)

print(f"SLIC: Median = {np.median(times_slic):.2f} ms")
print(f"Feat: Median = {np.median(times_feat):.2f} ms")
print(f"Graph: Median = {np.median(times_graph):.2f} ms")
