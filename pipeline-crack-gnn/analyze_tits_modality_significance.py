"""
Significance Test on Multi-Draw TITS Modality Divergence (M4 vs M1)
======================================================================

REPORT.md Section 8 states the modality divergence finding (M4 leads
AIGLE_RN ~3.0x, M1 leads LCMS ~3.5x) using only mean +/- std across the
10 calibration draws in results/tits_multidraw_calibration.json. Unlike
the morphology-bucket comparisons in Section 6 (which report Cohen's d
and explicit CI-overlap / disjoint checks), this is the one architecture
comparison in the report with no formal significance test attached.

This script closes that gap using the per-draw values already stored in
results/tits_multidraw_calibration.json -- no new experiments or model
runs are required, since the raw 10-draw F1 values per sensor are already
persisted under each model's 'per_sensor_f1' -> '<SENSOR>' -> 'values' key.
"""
import json
import os
import numpy as np
from scipy.stats import ttest_rel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    return float(diff.mean() / diff.std(ddof=1))


def run():
    path = os.path.join(RESULTS_DIR, 'tits_multidraw_calibration.json')
    with open(path) as f:
        data = json.load(f)

    m1 = data['M1_deep_gnn']
    m4 = data['M4_hybrid_full']

    report = {}
    for sensor in ['AIGLE_RN', 'ESAR', 'LCMS']:
        m1_vals = m1['per_sensor_f1'][sensor]['values']
        m4_vals = m4['per_sensor_f1'][sensor]['values']

        t_stat, p_val = ttest_rel(m4_vals, m1_vals)
        d = cohens_d_paired(m4_vals, m1_vals)

        report[sensor] = {
            'm4_mean': round(float(np.mean(m4_vals)), 4),
            'm1_mean': round(float(np.mean(m1_vals)), 4),
            'paired_t': round(float(t_stat), 4),
            'p_value': float(p_val),
            'cohens_d': round(d, 4),
            'significant_at_0.05': bool(p_val < 0.05),
        }

    out_path = os.path.join(RESULTS_DIR, 'tits_modality_significance.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print('=' * 80)
    print('TITS MODALITY DIVERGENCE -- PAIRED SIGNIFICANCE TEST (10 CALIBRATION DRAWS)')
    print('=' * 80)
    for sensor, r in report.items():
        sig = 'SIGNIFICANT' if r['significant_at_0.05'] else 'NOT significant'
        print(f"{sensor:10s}  M4={r['m4_mean']:.4f}  M1={r['m1_mean']:.4f}  "
              f"t={r['paired_t']:.3f}  p={r['p_value']:.4f}  d={r['cohens_d']:.3f}  ({sig})")
    print(f'\nSaved to {out_path}')
    return report


if __name__ == '__main__':
    run()
