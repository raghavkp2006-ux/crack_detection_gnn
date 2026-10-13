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
