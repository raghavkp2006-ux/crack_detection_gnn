"""
test_crack_metrics.py
=====================
Comprehensive Unit Tests and Sanity Checks for crack_metrics.py:
  1. Parameter counting (count_params)
  2. GFLOPs calculation (count_gflops)
  3. Latency & FPS measurement (measure_fps)
  4. clDice calculation (cl_dice) on synthetic cases
  5. Connectivity Ratio calculation (connectivity_ratio) on synthetic cases
  6. Connectivity Ratio breakdown (connectivity_ratio_breakdown)
  7. Threshold sensitivity sweep (sweep_cr_thresholds)
  8. Stratified CR by segment size (cr_by_segment_size)
  9. End-to-end benchmark wrapper (benchmark_end_to_end)
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from crack_metrics import (
    count_params,
    count_gflops,
    measure_fps,
    cl_dice,
    connectivity_ratio,
    connectivity_ratio_breakdown,
    sweep_cr_thresholds,
    cr_by_segment_size,
    benchmark_model,
    benchmark_end_to_end
)


class DummyCNN(nn.Module):
    """Simple conv model for testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        return self.out(torch.relu(self.conv(x)))


class TestCrackMetrics(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_count_params(self):
        model = DummyCNN()
        params_m = count_params(model)
        self.assertAlmostEqual(params_m, 465 / 1e6, places=6)

    def test_count_gflops(self):
        model = DummyCNN()
        gflops = count_gflops(model, input_size=(1, 3, 64, 64), device=self.device)
        self.assertGreater(gflops, 0.0)
        self.assertIsInstance(gflops, float)

    def test_measure_fps(self):
        model = DummyCNN()
        res = measure_fps(model, input_size=(1, 3, 64, 64), device=self.device, n_runs=20)
        self.assertIn("fps", res)
        self.assertIn("latency_ms", res)
        self.assertGreater(res["fps"], 0.0)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_cldice_perfect_prediction(self):
        gt = np.zeros((50, 50), dtype=np.uint8)
        gt[10:40, 25] = 1
        pred = gt.copy()
        score = cl_dice(pred, gt)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_cldice_all_background(self):
        gt = np.zeros((50, 50), dtype=np.uint8)
        pred = np.zeros((50, 50), dtype=np.uint8)
        score = cl_dice(pred, gt)
        self.assertEqual(score, 1.0)

    def test_cldice_empty_prediction_with_nonempty_gt(self):
        gt = np.zeros((50, 50), dtype=np.uint8)
        gt[10:40, 25] = 1
        pred = np.zeros((50, 50), dtype=np.uint8)
        score = cl_dice(pred, gt)
        self.assertEqual(score, 0.0)

    def test_cldice_gap_in_line(self):
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[20:80, 50] = 1
        pred = gt.copy()
        pred[48:52, 50] = 0
        score = cl_dice(pred, gt)
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.85)

    def test_cldice_shifted_by_one_pixel(self):
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[20:80, 49:52] = 1
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[20:80, 50:53] = 1
        score = cl_dice(pred, gt)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_connectivity_ratio_perfect(self):
        gt = np.zeros((50, 50), dtype=np.uint8)
        gt[10:40, 25] = 1
        pred = gt.copy()
        cr = connectivity_ratio(pred, gt, overlap_thresh=0.5)
        self.assertAlmostEqual(cr, 1.0, places=4)

    def test_connectivity_ratio_gap(self):
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[20:80, 50] = 1
        pred = gt.copy()
        pred[48:52, 50] = 0
        cr = connectivity_ratio(pred, gt, overlap_thresh=0.5)
        self.assertEqual(cr, 0.0)

    def test_connectivity_ratio_breakdown(self):
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[10:30, 20] = 1   # Segment 1: will be preserved
        gt[40:70, 50] = 1   # Segment 2: will be fragmented
        gt[80:90, 80] = 1   # Segment 3: will be missed

        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[10:30, 20] = 1  # exact match -> preserved
        pred[40:50, 50] = 1  # part 1 of seg 2
        pred[55:70, 50] = 1  # part 2 of seg 2 -> coverage >= 50% but split into 2 components
        # Segment 3 has no prediction -> missed

        bd = connectivity_ratio_breakdown(pred, gt, overlap_thresh=0.5)
        self.assertEqual(bd["n_total"], 3)
        self.assertEqual(bd["n_preserved"], 1)
        self.assertEqual(bd["n_fragmented"], 1)
        self.assertEqual(bd["n_missed"], 1)

    def test_sweep_cr_thresholds(self):
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[10:30, 20] = 1  # length 20
        pred = np.zeros((100, 100), dtype=np.uint8)
        pred[10:20, 20] = 1  # covers 10/20 = 50%
        
        # at thresh=0.4: covered (50% >= 40%) -> CR=1.0
        # at thresh=0.5: covered (50% >= 50%) -> CR=1.0
        # at thresh=0.6: not covered (50% < 60%) -> CR=0.0
        sweep = sweep_cr_thresholds([pred], [gt], thresholds=[0.4, 0.5, 0.6])
        self.assertEqual(sweep["micro"][0], 1.0)
        self.assertEqual(sweep["micro"][1], 1.0)
        self.assertEqual(sweep["micro"][2], 0.0)

    def test_cr_by_segment_size(self):
        gt = np.zeros((100, 100), dtype=np.uint8)
        gt[10:25, 20] = 1  # 15 px (Small)
        gt[40:100, 50] = 1  # 60 px (Medium)
        
        pred = gt.copy()
        pred[10:25, 20] = 0  # miss small segment, keep medium
        
        df = cr_by_segment_size([pred], [gt], bins=[(0, 50), (50, 200), (200, float("inf"))], overlap_thresh=0.5)
        small_row = df[df["Size_Bucket"] == "Small (<50px)"].iloc[0]
        med_row = df[df["Size_Bucket"] == "Medium (50-200px)"].iloc[0]
        
        self.assertEqual(small_row["Total_Segments"], 1)
        self.assertEqual(small_row["CR"], 0.0)
        self.assertEqual(med_row["Total_Segments"], 1)
        self.assertEqual(med_row["CR"], 1.0)

    def test_benchmark_end_to_end_cnn(self):
        model = DummyCNN()
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        res = benchmark_end_to_end(model, img, model_name="DeepCrack", device=self.device, n_runs=10)
        self.assertIn("end_to_end_latency_ms", res)
        self.assertIn("end_to_end_fps", res)
        self.assertGreater(res["end_to_end_fps"], 0.0)


if __name__ == "__main__":
    unittest.main()
