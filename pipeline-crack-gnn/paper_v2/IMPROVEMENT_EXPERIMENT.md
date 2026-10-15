# Validation-first M2/M4 experiment

Use Python 3.11 and run from `pipeline-crack-gnn/`. The isolated driver never
calls the legacy trainers' automatic test evaluation. It creates immutable run
configs, probability files, checkpoint files, and a final configuration lock.

```powershell
py -3.11 -m unittest discover -s paper_v2 -p test_improvement.py -v
py -3.11 -m unittest discover -s paper_v2 -p test_blend_selection.py -v
py -3.11 paper_v2/improve_m2_m4.py run --output results/paper_v2/improve_m2_m4_NEW --cache-policy preserve
```

Run the existing metric tests from the repository root:

```powershell
py -3.11 -m unittest test_crack_metrics -v
```

The five budgets are 1/2/4/2/1 hours. `--started-at <UTC-ISO-timestamp>` charges
earlier implementation work against the protocol budget. The epoch timing gate
uses three measured epochs, excludes the first epoch from steady-state timing,
and extrapolates to the configured maximum. Above 30 minutes, only skeleton
weight 0.3 is swept. Budget estimates also include validation and checkpoint I/O.

Seed 42 fixes train/validation membership for model seeds 42/1/2. Processing
replaces an incumbent only with at least +0.010 absolute validation strict CR
and a satisfied validation clDice guard. M4 and blends recalibrate per seed.
M2 and DeepCrack transfer seed-42 operating points without another search.
DeepCrack raw and processed use one original DeepSegmentor implementation.

The original native training caches reproduce with SLIC connectivity=True.
They are preserved for loss comparisons. All inference and newly built
600-segment training graphs use False; the finer-SLIC contrast therefore also
changes native training connectivity. Dataset identities include the directory
split: ten train/test filenames collide but are byte-distinct images. Existing
source-ID overlap is disclosed rather than silently changing the required split.

Explicit stage commands are `protocol`, `search`, `train`, `final-test`, and
`report`, each with `--output <directory>`. An existing successful stage must
not be executed again. `train-one --config <run/config.json>` reproduces a run
inside its original training budget; copy the configuration into a fresh
experiment with the same frozen protocol for a separate reproduction.
Every exact per-run invocation and environment is recorded in the output.

Only `final-test` decodes test images or labels. Protocol hashes same-name
image bytes only to audit identity. `final-test` requires the configuration and
source hashes to match, and exclusively creates `test_consumed.json` before
loading test labels. Failure consumes the test attempt; it cannot be retried.
The final job reproduces the old fixed-threshold controls before accepting any
new comparisons. Summary tables and overlap sweeps all use the same locked masks.

Main artifacts: `RESULTS.md`, `protocol.json`, `original_processing_locked.json`,
`locked_configs.json`, `checkpoint_manifest.json`, `final_results.json`,
`validation_trials/`, `runs/`, and `events.jsonl`. Failures produce a partial
report and preserve all completed artifacts. Old checkpoints/NPZ files are
never overwritten; epoch checkpoint filenames are immutable.

The completed 20261014_1112_02 experiment includes a pre-test independent
replay of every final blend grid; all selected recipes were unchanged. After
that experiment, the working runner's blend selector was corrected to retain
a superior zero-weight M4 control and prefer fewer processing operations on
metric ties. Synthetic regression tests cover both cases. The exact code used
for the historical run remains in its source_snapshot directory; the later
correction and hashes are documented in postrun_selector_fix.json. Replaying
the historical code requires restoring the archived files at their recorded
repository-relative paths into a separate checkout, then using a fresh output
directory. Do not rerun a completed stage against the original experiment.
