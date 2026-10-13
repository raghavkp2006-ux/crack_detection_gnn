# Paper v2 reproduction code

Run commands from `pipeline-crack-gnn/`. Prompt 1 only adds code and performs the
cheap checks; the commands below that train/analyze/benchmark are for Prompt 2
**after author review**. Do not tag before experiments are complete.

## Environment and data

On this Windows host use Python 3.11 (`py -3.11`); default `python` points at an
incomplete Python 3.12 environment. The existing Python 3.11 environment has
CUDA-capable PyTorch, torchvision, PyG and fvcore. Elsewhere activate a compatible
PyTorch/torchvision environment and install:

```bash
python -m pip install -r paper_v2/requirements-paper.txt
```

Place the raw data outside the pipeline at:

```text
../DeepCrack/train_img/
../DeepCrack/train_lab/
../DeepCrack/test_img/
../DeepCrack/test_lab/
```

Source: [Liu et al., DeepCrack dataset](https://github.com/yhlleo/DeepCrack),
Neurocomputing 338 (2019), 139–153, DOI 10.1016/j.neucom.2019.01.036.
The authors restrict dataset images/annotations to non-commercial research and
education; the dataset is not covered by this project's software license.
Retain attribution and follow the [upstream usage terms](https://github.com/yhlleo/DeepCrack#deepcrack-a-deep-hierarchical-feature-learning-architecture-for-crack-segmentation).
The local dataset README contains the citation.

Existing graphs must be in `data/graphs/{train,test}` (300/237 files) and existing
superpixel maps in `data/superpixels/{train,test}`. Maps are looked up by split
and stem. Missing maps are recomputed with unchanged SLIC settings; inspect any
node-count mismatch before continuing. Node order is `np.unique(segments)`.

## Run order

`run_all.sh` executes this sequence, stops on any error, and keeps training
sequential on one GPU. On Windows run the individual commands with `py -3.11`
in place of `python`; Git Bash can run the shell script after activating a
suitable environment. Ensure the GPU is otherwise idle before benchmarking.

```bash
python paper_v2/dataset_stats.py
python paper_v2/predict_all.py
python paper_v2/run_factorial.py
python paper_v2/predict_all.py
python paper_v2/train_deepcrack.py --bs 1 > results/paper_v2/deepcrack_train.log 2>&1
python paper_v2/analyze_main.py
python paper_v2/topology.py
python paper_v2/benchmark_efficiency.py
python paper_v2/make_tables.py
```

The cheap statistics check found both 384×544 and 544×384 test mask sizes.
The CNN trainer checks raw image dimensions in both splits before launching,
automatically falls back to batch 1 for mixed sizes and prints first-epoch
wall time including validation. `--bs 1` makes the intended setting explicit.
All other supplied settings, fixed seeds, splits, and validation threshold grid
are unchanged. ImageNet weights may download on the first CNN initialization.

`predict_all.py` verifies recomputed F1 against stored holdout F1 (tolerance
0.002) for every available checkpoint, including resumed caches. Original
missing checkpoints or mismatches halt the pipeline; new cells are reported
pending until trained. The primary set is all four focal graph models plus
**DeepCrack-style (re-implemented)**. WCE is a secondary loss comparison.
The CNN is VGG-16 + five HED side outputs and fusion, not the authors' released
DeepCrack model. Scoring uses the same node labels; CNN pixel probabilities
are averaged within identical superpixels before validation calibration.

## Outputs and interpretation

All artifacts are written to `results/paper_v2/`: dataset_stats.json,
test_labels.npz, predict_check.json, runs/<cell>/seed_<seed>/result.json,
probs/<cell>/seed_<seed>.npz, analysis_main.json, test_buckets.json,
topology.json, efficiency.json and PAPER_NUMBERS.md. Probability NPZ files
contain a flat float32 `probs` array and scalar `tau`; test_labels.npz holds
aligned `y`, `offsets`, and `stems`. Commit generated JSON/NPZ and the final
numbers only after the relevant stages have actually run.

Topology reconstructs masks from nodes for all methods; clDice skips empty GT
skeletons. Connectivity uses 8-connected GT components ≥20 pixels and considers
a component intact when one predicted component covers ≥50% of its pixels.
Micro pools component counts; macro averages per-image ratios. Earlier metrics
are not directly comparable (details in NOTES.md).

Efficiency uses 237 RGB arrays in RAM, FP32, no_grad, 30 warm-up and 500 timed
cyclic inputs, CUDA synchronization, and clock/temperature/power telemetry.
Model-only timing excludes preparation/transfer; end-to-end includes RGB→mask.
Both complexity counters use **GMACs**, not GFLOPs. Inspect recorded fvcore
unsupported operations before interpreting gaps. Analytic/fvcore graph counts
use the same representative input; mean-graph analytic counts are separate.
Per-iteration stages add to E2E; medians need not add. Stage means and total
mean are also saved for an additive summary.

`make_tables.py` only formats saved summaries. It refuses missing final inputs
or incomplete graph verification. No placeholder results are generated. The
manuscript is absent from this workspace; later use
`python paper_v2/make_tables.py --paper /path/to/manuscript.tex` to inventory
quantitative abstract/conclusion sentences, then review their meaning against
the tables before editing the paper.

Do not commit `results/paper_v2/runs/DeepCrack/**/best_model.pt`; after training,
attach those checkpoints to a GitHub Release. Release creation/tagging is
reserved for a later prompt. See NOTES.md for code deviations and check results.
