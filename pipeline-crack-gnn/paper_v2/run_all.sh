#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python paper_v2/dataset_stats.py
python paper_v2/predict_all.py            # Verify the 20 original graph runs; fail on mismatch.
python paper_v2/run_factorial.py          # Train the 20 missing graph runs.
python paper_v2/predict_all.py            # Verify all 40 graph runs.
python paper_v2/train_deepcrack.py > results/paper_v2/deepcrack_train.log 2>&1
python paper_v2/analyze_main.py
python paper_v2/topology.py
python paper_v2/benchmark_efficiency.py   # Run with GPU otherwise idle.
python paper_v2/make_tables.py
