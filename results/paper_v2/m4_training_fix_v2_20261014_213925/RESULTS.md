# M4 stabilization v2 results

**Budget-stopped partial delivery:** 11 completed runs plus D seed 1 at epoch 56 (1,156 logged epochs). No TITS inference, no winner, and no valid held-out lock. Both preregistered claims remain inconclusive for the full study. Finishing D, its calibration, selection, and the once-only held-out comparison requires an explicitly extended budget.

**Status: partial / failed**. Preregistered claims are in `claim.md`. Every selection uses the frozen 210/45/45 DeepCrack seed-42 split; training seeds change randomness only. The consumed 237-image DeepCrack test was never evaluated or used as new evidence.

Reason: Nine-hour budget cannot fit next mandatory operation (0.0s)

## Dataset provenance

TITS: 269 source images, 66 uniquely paired annotated native masks across five sensor groups. Supplied layout: `IMAGES/<sensor>, GROUND_TRUTH/<sensor>, RESULTS/<sensor>; no train/test directories in supplied data`. All 66 annotated pairs form the locked cohort; unannotated images are excluded. No TITS-finetuned checkpoints are used. Grayscale images are promoted to three identical channels. Eight masks have larger native dimensions than their source images; inference remains 256×256 and probabilities are projected to native GT size. Black pixels are cracks. No byte duplicate was found against the 300 DeepCrack training-source images. The external folder is read-only; hashes and metadata are recorded.

DeepCrack source-ID overlap exists between train and validation/test in the historical source audit, despite fixed membership. Validation is therefore not claimed to be source-independent; this study does not silently repair or reshuffle the split.

## Fixed-epoch diagnosis

The protocol requires 100 epochs per run with no early stopping. Eleven runs completed 100 epochs; D seed 1 stopped at 56 because the nine-hour budget expired. D seed 1 curves and peak epochs below describe only its observed partial trajectory. Native pixel F1, clDice and strict CR are logged each epoch at raw threshold 0.5, plus node F1 on historical native training graphs. Epoch selection uses the trailing five complete epochs of (clDice + strict CR)/2, retaining the endpoint weights, with the earliest epoch winning ties. Checkpoints include every tenth and final epoch and each improving smoothed selection.

A retains the historical five-epoch warmup and node-F1 ReduceLROnPlateau schedule, but disables early stopping. B combines effective-batch-8 gradient accumulation with cosine LR 1e-3 to 1e-5. C adds an eight-view deterministic augmented graph bank shared across seeds; a random cached view per training source is chosen each epoch. D adds EMA parameters at decay 0.99 after each optimizer step, copying current BatchNorm buffers. Accumulation retains one graph per forward (including per-graph BatchNorm); the final two-graph group is normalized by two.

![A epoch curves](plots/A_epochs.png)

![B epoch curves](plots/B_epochs.png)

![C epoch curves](plots/C_epochs.png)

![D epoch curves](plots/D_epochs.png)

| Candidate | Seed | Node F1 peak | Pixel F1 peak | clDice peak | Strict CR peak | Connectivity peaks after node peak |
| --- | --- | --- | --- | --- | --- | --- |
| A | 42 | 84 | 84 | 84 | 7 | {'cldice': False, 'strict_cr': False} |
| A | 1 | 61 | 61 | 99 | 10 | {'cldice': True, 'strict_cr': False} |
| A | 2 | 89 | 89 | 89 | 10 | {'cldice': False, 'strict_cr': False} |
| B | 42 | 85 | 85 | 84 | 14 | {'cldice': False, 'strict_cr': False} |
| B | 1 | 99 | 99 | 99 | 71 | {'cldice': False, 'strict_cr': False} |
| B | 2 | 89 | 89 | 89 | 19 | {'cldice': False, 'strict_cr': False} |
| C | 42 | 85 | 85 | 85 | 52 | {'cldice': False, 'strict_cr': False} |
| C | 1 | 87 | 87 | 87 | 26 | {'cldice': False, 'strict_cr': False} |
| C | 2 | 90 | 90 | 90 | 35 | {'cldice': False, 'strict_cr': False} |
| D | 42 | 20 | 19 | 47 | 80 | {'cldice': True, 'strict_cr': True} |
| D | 1 | 24 | 9 | 24 | 5 | {'cldice': False, 'strict_cr': False} |
| D | 2 | 18 | 18 | 18 | 58 | {'cldice': False, 'strict_cr': True} |

## Validation selection

No final candidate selection or held-out lock is available. A/B/C have complete three-seed validation, shown below; D has only two completed seeds and is excluded from aggregate selection. The incumbent remains unreplaced. Scores are fractions.

| Recipe | Processed strict CR mean ± sample SD | Processed clDice mean ± sample SD | Replacement guard |
| --- | --- | --- | --- |
| old_M4_processed | 0.394904 ± 0.022064 | 0.660229 ± 0.005005 | incumbent |
| A | 0.530786 ± 0.030101 | 0.578887 ± 0.025655 | fails clDice guard |
| B | 0.507431 ± 0.040451 | 0.599574 ± 0.053453 | fails clDice guard |
| C | 0.496815 ± 0.016852 | 0.606049 ± 0.045936 | fails clDice guard |

## Pre-registered verdicts

**Claim 1: inconclusive.** The inference concerns validation clDice and strict CR only, under the frozen split and three-seed rule. A lower training loss alone does not establish the claim. If no candidate replaces the incumbent this is a negative result; no additional recipe is introduced after results.

**Claim 2: inconclusive.** This verdict compares RAW new M4 against RAW DeepCrack on mean strict CR and small-crack [0,50) connectivity only, at their locked fixed thresholds. Processed results are separate and do not substitute for raw evidence. Zero-shot TITS transfer may hurt every model, and any other metric/model contrast is exploratory.

## Negative results, failures and skipped work

- Optional M2 retraining was skipped; its three historical seed checkpoints and locked recipes are retained.
- No final selection was made: D is incomplete and `selection.json` does not exist. Completed A/B/C comparisons are recorded in `partial_validation_summary.json`; none passes the replacement guard because mean clDice falls below the incumbent.
- Runtime failures (if any) are recorded in `failures.jsonl`; incomplete runs retain all epoch logs and immutable checkpoints.
- No consumed DeepCrack test run, TITS tuning, source split reshuffling or post-confirmation config changes were performed.

## Metric definitions

Strict CR counts all 8-connected GT components, requires at least 50% native GT-area coverage and exactly one connected region in GT∩prediction, and pools preserved/total components across images. Earlier paper_v2 topology excludes components below 20 pixels and accepts coverage by one predicted component without requiring connected GT intersection. Bridging outside GT therefore need not fix strict CR. clDice averages historical per-image skeleton scores with historical empty-mask handling; pixel F1 pools TP/FP/FN. Native size bins are [0,50), [50,200), [200,inf) GT-component area pixels.

## Reproduction and timing

The original planned command is preserved in `REPRODUCE.initial_plan.ps1`; it includes held-out confirmation and must not be run as an automatic verification step. `REPRODUCE.ps1` now records validation-only instructions. A new experiment requires a fresh preregistered output folder and a separately authorized time budget. Per-run exact commands are in `runs/<candidate>/seed_<seed>/command.json`. `protocol.json`, per-run configs, `checkpoint_manifest.json`, source hashes and input hashes record the full settings. Runs cannot overwrite completed or interrupted runs. Held-out confirmation has an exclusive consumed marker and cannot retry. `events.jsonl` and `trial_records.json` include wall times and statuses.

## Final partial verification and remaining work

`verification_partial.json` verifies 284 checkpoint hashes, 29 source/snapshot records, 3959 historical file metadata records, 255 DeepCrack training/validation input pairs and 66 TITS pairs. No mismatches were found. These checks read inputs without evaluating models. The existing 13 synthetic checks passed (`synthetic_verification.json`).

Unfinished mandatory work: D seed 1 epochs 57–100 and calibration; complete three-seed selection and immutable lock; new fvcore/attention compute measurements and held-out tradeoff plots; once-only TITS inference and raw/processed tables, size-bin comparisons and claim-2 verdict. No fabricated lock JSON or held-out scores were written. `lock_status.json` explicitly records NOT_LOCKED.

The nine-hour wall-clock deadline was 2026-10-15 06:39:25 Asia/Kolkata. Training stopped at the budget gate; report finalization occurred after the cutoff. Elapsed wall time through verification, including implementation, pauses and report finalization: 9.247 hours. Parallel job durations overlap and must not be summed.

The shape-only CNN calculation is available in `cnn_analytic.json` (14,720,389 parameters; 40.105804 convolution GFLOPs at 256×256, excluding other operations). New full comparison compute measurements were not reached. Historical compute references remain historical evidence, not a new measurement.
