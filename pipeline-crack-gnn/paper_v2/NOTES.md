# Paper v2 implementation notes

Scope: Prompt 1 only. No training, efficiency benchmark, topology analysis, main analysis,
or paper edits are run in this change. Output artifacts from dataset statistics and
checkpoint inference are the explicitly allowed exceptions to the results write restriction.

## Repository and environment

- Git root is one directory above the pipeline; use pipeline-crack-gnn as cwd.
- Python 3.11 has PyTorch 2.11.0+cu128, torchvision, PyG, scikit-image, YAML,
  and fvcore. Default Python 3.12 is missing the required packages. Use `py -3.11`
  on this Windows host (or activate an equivalent environment and use `python`).
- No AGENTS.md was found in the workspace. Existing unrelated untracked files and
  the DeepSegmentor submodule worktree are excluded from the commit.

## Changes from the supplied code

- common.py: key cached segments by (split, stem), scanning only train/test folders.
  A recursive stem-only index could silently use test_single or TITS superpixels.
- dataset_stats.py: absolute graph paths, stable ordering of image sizes, main guard.
- run_factorial.py: set cwd for the existing trainer's relative config data paths;
  resume only when both result and checkpoint exist. Training API/settings unchanged.
- predict_all.py: compare cached labels, offsets and stem order against actual graphs;
  validate probability shape/range; re-infer when checkpoint/result changes; verify
  cached predictions on every invocation. Missing original runs are fatal; missing
  new cells are reported as pending. Persist verification before immediately stopping
  on a mismatch and append its details here. Only passing probabilities are saved.
- deepcrack_model.py: supplied architecture and loss copied unchanged. It is
  **DeepCrack-style (re-implemented)**, not the authors' released model.
- train_deepcrack.py: defer data reads to main, check both train/test image sizes and
  automatically use batch 1 for mixed sizes, validate validation node counts, print
  first-epoch duration including validation, reject non-positive CLI settings,
  and resume only when checkpoint/result/probabilities all exist.
  Seeds, splits, losses, optimizers and threshold grid are unchanged.
- stats_core.py: close NPZ handles, validate probability alignment/range/threshold,
  reject invalid Holm p-values, and recognize complete predictions by the five exact seed filenames.
- analyze_main.py: exact seed-file completeness, empty-bucket handling, explicit
  zero-variance paired effect size. Identical paired vectors yield p=1 instead of
  NaN; cap bootstrap p at 1. Preserve four/five-comparison Holm family sizes in
  partial analyses by padding absent comparisons with p=1. Resampling unchanged.
- topology.py: require complete five-seed models and assert mask/segment dimensions.
- benchmark_efficiency.py: main guard; real holdout ordering for all 237 preloaded
  images; FP32 with TF32 disabled; explicit CUDA synchronization; cycle inputs for
  warm-up and timed passes including model-only calls; DeepCrack mid telemetry;
  pretrained VGG fallback; compare analytic and fvcore counters on the same graph;
  retain mean-graph analytic count separately; add DeepCrack convolution MAC count
  and unsupported-op details. Stage durations sum exactly to each iteration's total.
  Stage medians generally do NOT sum to the median total: report stage means and
  total mean as an additive summary alongside the requested medians/IQR.
- make_tables.py: implemented the nine requested sections from stored JSON only.
  Fail on missing/incomplete final inputs instead of inventing values. The exact
  manuscript is absent (only unrelated dataset BibTeX .tex files exist); an optional
  --paper argument can inventory quantitative abstract/conclusion sentences later.
  Known claim checks are explicitly labeled comparisons, not manuscript quotations.
- run_all.sh: fail-fast sequential pipeline; removed masked analysis errors and
  background training/wait so a failed DeepCrack job cannot silently continue.
- requirements-paper.txt: includes repo requirements and additional torchvision,
  scipy, fvcore, psutil, Pillow dependencies; no environment/package installs done.
- README hygiene: update both root and pipeline README architecture/config sections;
  banner root REPORT.md and both TEST_RESULTS.md copies. Existing results docs are
  untouched. Ignore only DeepCrack paper-v2 checkpoints, retaining JSON/NPZ outputs.

## Comparison with earlier topology definitions (source inspection only)

`../crack_metrics.py:535` counts every 8-connected GT component, with no 20-pixel
minimum. It requires total overlap >=50%, exactly one connected overlap region,
and exactly one predicted component. The new definition drops GT components under
20 pixels and checks whether any single predicted component covers >=50% of a GT
component. Other predicted components may overlap it; overlap can be disconnected
inside the GT when connected outside it. These definitions are not interchangeable.
Earlier clDice returns 1 for two empty masks and 0 for empty GT with a prediction;
new clDice excludes all empty GT skeletons. New connectivity macro omits images
without retained GT components. Old topology tables must be replaced after Prompt 2.
No topology script or synthetic topology experiment was executed here.

## Later-run limitations

Training convergence, learned checkpoint compatibility, validation-selected thresholds
for the missing cells, statistical tables, topology outputs, sustained efficiency,
actual fvcore tracing coverage and GPU idle state during timing require Prompt 2.
ImageNet VGG weights may download on the first training/fallback benchmark invocation.
Attach DeepCrack checkpoints to a GitHub Release after experiments; do not tag yet.
Dataset source and non-commercial research/education terms are linked in README.md.

## Cheap-check results

All checks used `py -3.11` from pipeline-crack-gnn. No packages were installed.

| Section 5 check | Result | Evidence |
| --- | --- | --- |
| 1. py_compile all Python files | PASS | All 11 files compile. PowerShell passes the expanded path list to py_compile. |
| 2. common/stats_core/deepcrack_model imports | PASS | All imports resolve in Python 3.11. |
| 3. CPU CNN forward | PASS | Fused (1,1,384,544); five sides each (1,1,384,544); 14,716,171 parameters; pretrained=False. |
| 4. dataset_stats.py | PASS | 300 train-directory graphs; every seed 210/45/45 train/val/unused. Test: 237 images, 67,998 nodes, crack-node fraction 0.0425159564, mean 286.9114 nodes and 2941.9494 directed edges, mean crack-pixel fraction 0.0432707396. Mask sizes [384,544] and [544,384]. |
| 5. predict_all.py | PASS | All 20 original checkpoints verified on CUDA. Maximum absolute stored/recomputed F1 difference 0.0; 20 new graph runs pending. |
| 6. Consistency review | PASS | Static discovery resolves every import; all eight cell config paths exist. All 20 NPZs have aligned float32 probs and scalar tau. 300/300 train and 237/237 test stems have cached superpixels. Writers/readers share paths and node ordering. |

`fvcore` is installed; actual tracing is reserved for Prompt 2. Raw image header
sizes are checked by the training script before launch, but this turn's statistics
size inventory is from test masks. Cache presence was checked by filenames;
cache pixel geometry/node alignment with CNN inference remains a later-run check.
`PAPER_NUMBERS.md`, analysis_main.json, topology.json and efficiency.json are not
created here: producing them would require running the deferred analyses/benchmark.
Only dataset_stats.json, predict_check.json, test_labels.npz and the 20 verified
probability NPZ files are committed as generated artifacts.

## Prompt 2 graph stage — 2026-10-13

Executed on the existing Python 3.11 CUDA environment / RTX 5050 Laptop GPU.
Skipped dataset_stats.py and the initial predict_all.py as requested.
Ran run_factorial.py sequentially for all 20 missing graph cells, then predict_all.py.
All 40 graph runs passed the F1 verification gate; maximum absolute error = 0.00000000.
All seeds, graph settings, split logic and validation threshold grid stayed unchanged.

| Model | Seed | τ* | Validation F1 | Holdout node F1 | Train seconds |
| --- | --- | --- | --- | --- | --- |
| M1_focal | 42 | 0.50 | 0.6851 | 0.4925 | 89.5 |
| M1_focal | 123 | 0.62 | 0.6864 | 0.7603 | 139.6 |
| M1_focal | 456 | 0.46 | 0.7011 | 0.7525 | 101.5 |
| M1_focal | 789 | 0.58 | 0.7067 | 0.7300 | 131.9 |
| M1_focal | 2026 | 0.58 | 0.7147 | 0.7345 | 82.8 |
| M2_wce | 42 | 0.90 | 0.7050 | 0.6894 | 43.3 |
| M2_wce | 123 | 0.80 | 0.7440 | 0.7728 | 46.0 |
| M2_wce | 456 | 0.82 | 0.7916 | 0.7242 | 46.1 |
| M2_wce | 789 | 0.90 | 0.6398 | 0.7221 | 36.9 |
| M2_wce | 2026 | 0.86 | 0.7452 | 0.7237 | 34.8 |
| M3_wce | 42 | 0.90 | 0.5450 | 0.5546 | 57.6 |
| M3_wce | 123 | 0.50 | 0.5181 | 0.5528 | 56.1 |
| M3_wce | 456 | 0.52 | 0.7172 | 0.6387 | 113.8 |
| M3_wce | 789 | 0.68 | 0.5229 | 0.6422 | 63.9 |
| M3_wce | 2026 | 0.66 | 0.5911 | 0.5569 | 96.9 |
| M4_wce | 42 | 0.52 | 0.7459 | 0.6260 | 69.8 |
| M4_wce | 123 | 0.76 | 0.6780 | 0.7539 | 62.9 |
| M4_wce | 456 | 0.38 | 0.7393 | 0.6781 | 92.4 |
| M4_wce | 789 | 0.88 | 0.6637 | 0.7260 | 77.9 |
| M4_wce | 2026 | 0.84 | 0.7418 | 0.7769 | 64.4 |

Review flags: M1 focal seed 42 has holdout F1 0.4925 versus validation F1 0.6851;
its inference exactly reproduces the training result. M2 WCE seeds 42 and 789,
and M3 WCE seed 42 select τ*=0.90, the top of the unchanged grid.
No test-driven retuning or seed dropping was performed.

Paused at the user-requested review checkpoint before starting DeepCrack.
train_deepcrack.py, analyze_main.py, topology.py, benchmark_efficiency.py and
make_tables.py have not been executed in this stage. On continuation use --bs 1,
check the first epoch duration, and if over about 40 s reduce max_epochs and
patience while retaining all five seeds, as authorized by the user.
factorial_train.log and factorial_verify.log preserve the actual command output.

## Shape guards and ordered continuation — 2026-10-13

Step 1 adds image-stem-specific probability/segment and test-GT/segment shape
assertions to CNN prediction, plus image/segment, node-count, and reconstructed
mask assertions in graph benchmarking. Dense CNN benchmarking does not map
through segments. Added an environment-enabled epoch-boundary timing gate
(`PAPER_V2_FIRST_EPOCH_LIMIT_SECONDS=40`) to enforce the user's step 3 stop rule
without changing max_epochs, patience, seeds, splits, or primary threshold grid.
The gate records unrounded first-epoch timing and simple maximum-epoch runtime
extrapolations in a new JSON file. No experiments run during this code step.
PowerShell expands the .py paths explicitly because Python does not expand globs.

Step 2 adds sensitivity_tau.py: all eight graph cells plus DeepCrack, all five
seeds, the original 45 validation graphs per seed, validation-only maximization
on 0.10–0.98 in 0.02 steps, with first-maximum tie handling. Test scores use the
existing immutable test probabilities at the newly selected threshold. It saves
old/new thresholds and test F1, validation curves/stems, the old-0.90 boundary
count, and validation/test crack-node fractions per seed. Missing runs fail
before inference; output uses exclusive creation. make_tables.py optionally
includes the sensitivity JSON verbatim. The sensitivity script has not run in
this code step.

## Ordered execution report — stopped at step 6 (2026-10-13)

Steps 1–5 completed. Step 6 wrote the complete UTF-8 PAPER_NUMBERS.md, then
failed while printing it to the Windows cp1252 console:

```text
UnicodeEncodeError: 'charmap' codec can't encode character '\u03b1' in position 486: character maps to <undefined>
```

Stopped immediately under the user's fail-stop instruction. No retry, overwrite,
step 7 result commit, or push was performed. Current local code commit:
9a3120e17d598a2af30dad24343125f100c4a5cc.
Remote main remains at 2ea15653846116bc843b8f7f8488f73b54fc353a.
Generated experiment artifacts remain local and uncommitted.

PAPER_NUMBERS.md path:
C:\antigravity_projects\dcsnew\pipeline-crack-gnn\results\paper_v2\PAPER_NUMBERS.md

Recorded wall seconds (not rounded):
- Step 1 editing, compilation, commit: 1.887
- Step 2 coding, compilation, commit: 103.561
- Step 3 training command including weight download: 2247.6033194
- Step 4 analyze_main.py: 7.4256805
- Step 4 topology.py: 48.1041447
- Step 4 sensitivity_tau.py: 62.7566007
- Step 5 preparation and benchmark: 915.406
- Step 5 benchmark command only: 713.0136982
- Step 6 make_tables.py (failed): 1.4431867
- Step 7: not run because step 6 failed.

DeepCrack saved result values (copied without rounding):
```json
[
  {
    "seed": 42,
    "best_tau": 0.5000000000000001,
    "best_val_f1": 0.8504672897196262,
    "holdout_f1": 0.8457004160887656,
    "epochs": 21,
    "params": 14716171
  },
  {
    "seed": 123,
    "best_tau": 0.42000000000000004,
    "best_val_f1": 0.8993020937188435,
    "holdout_f1": 0.8521798365122616,
    "epochs": 41,
    "params": 14716171
  },
  {
    "seed": 456,
    "best_tau": 0.5000000000000001,
    "best_val_f1": 0.8498498498498499,
    "holdout_f1": 0.8508145707486728,
    "epochs": 44,
    "params": 14716171
  },
  {
    "seed": 789,
    "best_tau": 0.5200000000000001,
    "best_val_f1": 0.861878453038674,
    "holdout_f1": 0.8242880171950564,
    "epochs": 25,
    "params": 14716171
  },
  {
    "seed": 2026,
    "best_tau": 0.54,
    "best_val_f1": 0.850632911392405,
    "holdout_f1": 0.8333939614405238,
    "epochs": 16,
    "params": 14716171
  }
]
```

Warnings and deviations:
- First seed's first epoch, including validation, was 28.500596699999733 seconds,
  below 40.0; max_epochs=60 and patience=10 were retained for every seed.
- ImageNet VGG weights downloaded once; network stalled temporarily and recovered.
  The download is included in step 3 wall time and excluded from epoch time.
- PowerShell explicitly expanded .py paths for py_compile. The environment-enabled
  epoch gate enforces the user stop rule without changing training parameters.
- Closed Antigravity IDE, Chrome, Brave, NVIDIA Overlay, RadeonSoftware,
  SnippingTool, and SystemSettings before benchmarking; Windows/Codex remained.
  Recorded idle GPU utilization samples were [1, 1, 1]. An optional Get-Process
  python absence query returned exit 1 because no Python process remained;
  the application-closing and GPU-sampling commands succeeded.
- fvcore unsupported operations are copied below. Native attention in M3/M4 is
  unsupported, so their fvcore totals omit that work; analytic GMACs are retained
  alongside them. Do not interpret fvcore hybrid counts as complete attention costs.
- The original broad DeepCrack/ ignore pattern also hides newly generated CNN
  result JSON and probability NPZ files. A future authorized result commit must
  explicitly include those paths while keeping best_model.pt excluded. No release
  was created and CNN checkpoints remain local/ignored.
- The manuscript is absent. No LaTeX was touched.
- SHA-256 comparison confirmed all 85 pre-existing paper_v2 result files
  are unchanged. No seeds, splits, training hyperparameters or primary grid changed.

```json
{
  "M1_focal": {
    "aten::index_select": 8,
    "aten::new_ones": 8,
    "aten::scatter_add_": 16,
    "aten::expand_as": 16,
    "aten::div": 8,
    "aten::add": 8
  },
  "M2_focal": {
    "aten::index_select": 2,
    "aten::new_ones": 2,
    "aten::scatter_add_": 4,
    "aten::expand_as": 4,
    "aten::div": 2,
    "aten::add": 3
  },
  "M3_focal": {
    "aten::index_select": 2,
    "aten::new_ones": 2,
    "aten::scatter_add_": 4,
    "aten::expand_as": 4,
    "aten::div": 2,
    "aten::add": 7,
    "aten::_native_multi_head_attention": 2,
    "aten::gelu": 2
  },
  "M4_focal": {
    "aten::index_select": 2,
    "aten::new_ones": 2,
    "aten::scatter_add_": 4,
    "aten::expand_as": 4,
    "aten::div": 2,
    "aten::add": 7,
    "aten::pow": 1,
    "aten::mul": 2,
    "aten::sin": 1,
    "aten::cos": 1,
    "aten::_native_multi_head_attention": 2,
    "aten::gelu": 2
  },
  "DeepCrack": {
    "aten::rsub": 2,
    "aten::pad": 1,
    "aten::add": 10,
    "aten::max_pool2d": 4
  }
}
```

Claims to re-check (verbatim from PAPER_NUMBERS.md):

These checks compare stored summaries; they do not replace manuscript review.

- DeepCrack ahead of M4: supported numerically; 0.8413 vs 0.7378. See pairwise tests for significance.
- M4 ahead of M2: supported numerically; 0.7378 vs 0.7199. See pairwise tests for significance.
- Focal improves M1: contradicted numerically; ΔF1=-0.0220; Holm p=1.0000.
- Focal improves M2: contradicted numerically; ΔF1=-0.0066; Holm p=1.0000.
- Focal improves M3: supported numerically; ΔF1=0.1373; Holm p=0.0338.
- Focal improves M4: supported numerically; ΔF1=0.0256; Holm p=1.0000.
- Historical 126× parameter claim: actual DeepCrack/M4 parameter ratio = 125.9925×.
- Historical ≈24× slower claim: actual M4/DeepCrack median end-to-end ratio = 8.2644×.
- Topology superiority claims must be checked against section 6 using the revised component/empty-GT definitions.
- Exact abstract/conclusion text unavailable. Rerun with --paper /path/to/manuscript.tex to inventory every quantitative sentence; author review remains required.


## Prompt 4 — 2026-10-14: Task A

Task A source overlap and provenance are in source_overlap.json and data_checks.json.
Sandbox CIM/Python launcher access required elevation for the requested read-only checks. Initial rg literal-glob searches were corrected to -g filters. No experiment or existing result was changed.
Dirichlet seed 42 is config-derived provenance; historical summary JSON does not embed the seed. Existing attention decay panel is simulated and will be omitted. Oracle plot uses 35 resized images, occupancy >0.30 labels, and pixel F1; it is not rerun.

```json
{
  "cpu_name": "AMD Ryzen 5 220 w/ Radeon 740M Graphics",
  "package_versions": {
    "torch": "2.11.0+cu128",
    "torch_geometric": "2.8.0.post1",
    "torchvision": "0.26.0+cu128",
    "scikit-image": "0.26.0",
    "numpy": "2.4.4",
    "scipy": "1.17.1",
    "fvcore": "0.1.5.post20221221"
  },
  "dirichlet_source": "results/full_evaluation_summary.json",
  "dirichlet_script": "run_full_evaluation_and_analysis.py",
  "dirichlet": {
    "M1 (8-Layer Deep GNN)": {
      "raw_energy": [
        5.9190298427006365,
        7.6016827965084515,
        10.871292145191868,
        10.57247290353169,
        8.625360453729263,
        5.955756836928396,
        3.769928246815119,
        2.483561240718209
      ],
      "normalized_energy": [
        0.052066130122024455,
        0.05318410154594155,
        0.06713157076521714,
        0.07974246048626642,
        0.07900087692953385,
        0.0740455933308972,
        0.055710283205897264,
        0.045633196904898275
      ],
      "checkpoint": "results/M1_deep_gnn/best_model.pt",
      "config": "configs/baseline_deep_gnn.yaml",
      "configured_seed": 42,
      "loss": "weighted_ce",
      "provenance_note": "Checkpoint/config paths from run_full_evaluation_and_analysis.py; seed from YAML, not embedded in the summary JSON."
    },
    "M2 (2-Layer Shallow GNN)": {
      "raw_energy": [
        0.203382689266108,
        0.2959360863290381
      ],
      "normalized_energy": [
        0.029014457686203,
        0.022715015317222993
      ],
      "checkpoint": "results/step2_loss/M2_shallow_gnn/best_model.pt",
      "config": "configs/step2_loss/shallow_gnn_focal.yaml",
      "configured_seed": 42,
      "loss": "focal",
      "provenance_note": "Checkpoint/config paths from run_full_evaluation_and_analysis.py; seed from YAML, not embedded in the summary JSON."
    },
    "M4 (Hybrid GNN+Trans)": {
      "raw_energy": [
        0.11441161515258001,
        0.17066476893127908
      ],
      "normalized_energy": [
        0.07396169422972429,
        0.049041076402183824
      ],
      "checkpoint": "results/step2_loss/M4_hybrid_full/best_model.pt",
      "config": "configs/step2_loss/hybrid_full_focal.yaml",
      "configured_seed": 42,
      "loss": "focal",
      "provenance_note": "Checkpoint/config paths from run_full_evaluation_and_analysis.py; seed from YAML, not embedded in the summary JSON."
    }
  },
  "dirichlet_definition": "Mean over test graphs of the mean directed-edge squared distance between row-wise L2-normalized hidden vectors.",
  "oracle_ceiling": {
    "script": "../create_all_figures.py",
    "function": "make_figS8_superpixel_ceiling",
    "image_selection": "First 35 sorted test JPG images",
    "image_processing": "OpenCV BGR images and grayscale masks resized to 256x256; mask binarized >128.",
    "segments": [
      100,
      200,
      300,
      500
    ],
    "slic": {
      "compactness": 10.0,
      "sigma": 1.0,
      "enforce_connectivity": false
    },
    "label_rule": "Superpixel positive iff mean of resized binary GT pixels strictly exceeds 0.30.",
    "scoring": "Pixel-level binary F1 on flattened GT and reconstructed mask, then mean and population SD across images.",
    "caveat": "Fixed 0.30 occupancy labeling is not an optimized oracle for pixel F1; old reference model scores are hardcoded.",
    "rerun": false
  },
  "attention_decay_profile": {
    "json_found": false,
    "source": "../create_all_figures.py:make_figS4_attention_temperature_ablation",
    "finding": "Existing panel (b) simulates exp(-alpha*distance); no measured spatial attention decay JSON found. Omit panel (b)."
  },
  "wall_seconds": 1.7677103000023635
}
```

### Prompt 4 stopped at Task B data preflight — 2026-10-14

Task A completed and exclusively created source_overlap.json and data_checks.json.
Task B's required fig14 DeepCrack end-to-end MEAN cannot be sourced from the
existing efficiency.json: DeepCrack has e2e_ms median/q1/q3/iqr/p95 and FPS,
but no e2e_mean_ms or raw timing samples. Graph models do have e2e_mean_ms and
stage_mean_ms. The median is not substituted for the requested mean. Under the
user's stop-on-failure rule, no figures or qualitative selections were generated,
no Task C inference was run, and no new commit/push was performed.

Additional source observation for a future continuation: the historical TITS
multidraw JSON records draw seeds and aggregate scores, but its inspected draw
records do not contain calibration/evaluation image IDs or indices. Exact stored
membership verification will need an authoritative membership source; it must not
be invented from metric values. No draw assertion was executed in this stopped run.

Recorded timings and status:
```json
{
  "failure_task": "B",
  "failure_figure": "fig14_efficiency_stages.png",
  "reason": "DeepCrack end-to-end mean is absent from efficiency.json; only median/quantiles and FPS are stored, with no raw samples.",
  "task_A_internal_check_seconds": 1.7677103000023635,
  "task_B_source_preflight_command_seconds": 0.4011072,
  "task_C": "not run",
  "figure_files_generated": [],
  "qualitative_stems_selected": [],
  "task_A_plus_B_preflight_workflow_wall_seconds": 204.946,
  "unchanged_existing_results": true
}
```
The Task A internal timer excludes Python imports and manual source inspection;
the workflow wall timer includes coding/inspection and Task B preflight. Separate
full interactive task durations were not instrumented. All available numbers are
copied unchanged into the report.

## Prompt 5 — 2026-10-14

Task 0: create_all_figures.py is excluded: it contains the personal absolute
TITS path C:\Users\ragha\Downloads\CrackDataset\TITS and a fixed local project
path C:\antigravity_projects\dcsnew. No credential-shaped literal was detected.
DeepSegmentor and images remain excluded. Existing CNN JSON/NPZ artifacts are
explicitly force-added; DeepCrack best_model.pt files remain ignored.

Correction to Task 0 note: create_all_figures.py derives PROJECT_ROOT dynamically;
the personal absolute path found is the TITS path, not a fixed project-root literal.
Task 1 uses 2000 shared cluster resamples with numpy.default_rng(0), fixed model
seeds, and row-normalized clDice definitions copied from topology.py. Holm applies
to the five scene-level Wilcoxon p-values. Bootstrap differences pool node counts
within each seed; scene-test means give equal weight to each scene. Leakage tests
repeat the same scene-level family after excluding the recorded 19 images.

### Prompt 5 completion

Task 0 local and remote main matched b6b414ae2a1715a6887a83fc1052e8e43885de5b.
Task 1 exclusively created analysis_cluster.json and CLUSTER_NUMBERS.md. Task 2
exclusively created the eight requested PNG/PDF pairs, captions_notes.md and
qualitative_selection.json. No figure was skipped. No model was retrained, and
existing results, model seeds/splits, thresholds, primary grids and LaTeX were
unchanged. Scene bootstrap RNG seed 0 is separate from the fixed model seeds.
No additional experiment, test suite, or figure was added.

The shared scene resampling preserves all images within sampled scenes; bucket
resampling includes only each bucket's images. clDice was reconstructed because
per-image values were not saved, with the original topology.py definitions.
Leakage significance comparisons use the same five-test Holm-Wilcoxon family
before/after removing the 19 overlapping test images.

Figure conventions/implementation choices: all labels use fonts >=8 pt; all
figures use the same model palette. Connectivity bars are unchanged seed means
without uncertainty bars; only clDice receives scene-bootstrap CIs in fig09.
For fig07, circles have exactly proportional analytic-GMAC area; crosshairs
locate the tiny graph circles without an artificial minimum area. fig14 uses the
stored DeepCrack median 35.97904999696766 and explicitly distinguishes it from
graph stage means. Transfer/forward includes H2D and D2H as in the source JSON.
Qualitative ties use first stored sorted order, with no replacement or hand
selection. Native aspect ratios are retained. Layout review found no overlapping
labels. Caption details and all sources are in figures/captions_notes.md.

Workflow wall seconds (including coding/inspection for each task; copied without rounding):
```json
{
  "task_0_wall_seconds": 148.23,
  "task_1_wall_seconds": 205.199,
  "task_2_wall_seconds": 137.701
}
```
Internal computation/render timings are separately saved in analysis_cluster.json
and figures/captions_notes.md. Elevated access was required for the Python
launcher and Git metadata/network access; no automatic-approval rejection occurred.

## Project pitch rerun 2026-10-14

The user requested all tests mentioned in Crack_Segmentation_Project_Pitch.docx
again (referred to as a PDF). Added rerun_pitch.py and exclusively created
results/paper_v2/pitch_rerun_20261014_01. All 14 existing metric unit tests passed.
Reran the five historical checkpoints on all 237 images with their original
fixed thresholds and 256x256 inference; each model received 500 timed model-only
and historical end-to-end iterations. Recomputed clDice, strict micro/macro
connectivity, fragmentation, size bins, and overlap thresholds 0.3 through 0.7.
Reran M4 SLIC connectivity True/False on the full holdout, graph-size statistics,
500-iteration SLIC timings for each setting, and direct GPU PE-fusion timing.
Recomputed strict historical size/topology metrics for all five current seeds
using saved predictions, with no retraining or recalibration. Existing results
file sizes and modification times matched the pre-run manifest on completion.

All historical clDice/connectivity results reproduced the pitch. M4's small and
medium component wins reproduced (21 versus 12, and 89 versus 69 components).
Historical M2 also exceeds original DeepCrack overall strict micro connectivity
(0.4223790322580645 versus 0.33669354838709675). Current M4 does not reproduce
the small/medium wins against the newer CNN. Current M1 weighted CE exceeds
that CNN in mean strict micro connectivity (0.32983870967741935 versus
0.3080645161290322); this rerun does not establish significance for that gap.

Protocol limitations are recorded in the report and JSON: different historical
and current CNN architectures; historical synthetic graph FLOP/timing inputs;
unsupported attention operations in fvcore; historical CNN end-to-end timing
excludes input preparation/transfer and binary reconstruction; historical size
bins actually use [0,50), [50,200), [200,infinity); SLIC control is inference-only.
Timing values were newly measured and differ from the pitch. Cross-sensor and
morphology experiments are labeled future work in the pitch and were not run.
The sandbox Python launcher failed to locate the installed interpreter before
any tests executed; approved elevated py -3.11 execution succeeded. No actual
test or experiment failed. No primary seeds, splits, thresholds, hyperparameters,
predictions, or LaTeX were changed. Computation wall time: 458.6033415999991 s;
per-section times are in pitch_rerun.json and PITCH_RERUN.md.
