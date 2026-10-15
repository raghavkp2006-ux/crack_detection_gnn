"""Reproduce a metadata-only partial report. No training or image decoding.

The absence of a second repository-local dataset triggers the user's stop rule.
Only historical validation summaries, epoch logs, and checkpoint bytes are read.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    output = args.output.resolve()
    # A reproduction must target a fresh folder and retains the preregistration.
    if output != here:
        output.mkdir(parents=True, exist_ok=False)
        (output / "claim.md").write_bytes((here / "claim.md").read_bytes())
    if (output / "RESULTS.md").exists():
        raise RuntimeError("Refusing to overwrite a completed report")
    assert (output / "claim.md").exists(), "Preregistration must precede audit"
    old = root / "pipeline-crack-gnn/results/paper_v2/improve_m2_m4_20261014_1112_02"
    source_paths = [old / name for name in ["protocol.json", "locked_configs.json", "selected_model.json"]]
    source_paths += [root / "pipeline-crack-gnn" / name for name in [
        "paper_v2/improve_m2_m4.py", "paper_v2/improvement_core.py",
        "src/preprocessing/build_dataset.py", "run_tits_pipeline.py"]]
    source_paths += [root / "DeepSegmentor/datasets/README.md", old / "RESULTS.md"]
    initial_hashes = {str(p): digest(p) for p in source_paths}
    protocol = read(old / "protocol.json")
    lock = read(old / "locked_configs.json")
    split = {key: protocol["split"][key] for key in ["train", "val", "unused"]}
    assert [len(split[k]) for k in ["train", "val", "unused"]] == [210, 45, 45]
    assert not (set(split["train"]) & set(split["val"]))
    save(output / "frozen_split.json", {"split_seed": 42, "split": split,
         "unused_role": "45 unused images from the 300 training-source images; not the consumed 237-image test"})

    images = Counter()
    archives = []
    # Enumerate names, sizes and directories; never decode dataset images/labels.
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.is_relative_to(output):
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            images[path.parent.relative_to(root).as_posix()] += 1
        if path.suffix.lower() in {".zip", ".tar", ".gz", ".7z", ".rar", ".h5", ".hdf5", ".mat"}:
            record = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
            if path.name == "images.zip":
                with zipfile.ZipFile(path) as archive:
                    record["members"] = archive.namelist()
            archives.append(record)
    caches = {}
    for suffix in ["graphs/tits", "graphs/tits_test", "superpixels/tits", "superpixels/tits_test"]:
        directory = root / "pipeline-crack-gnn/data" / suffix
        caches[suffix] = dict(pt=len(list(directory.rglob("*.pt"))), npy=len(list(directory.rglob("*.npy"))))
    inventory = dict(image_directories=dict(sorted(images.items())), archives=archives,
        cached_tits=caches, eligible_second_dataset=None,
        excluded=[
            "DeepCrack and DeepSegmentor/datasets/DeepCrack are the same named training dataset, not independent datasets.",
            "TITS source RGB images and native masks are absent from the repo; run_tits_pipeline.py points to C:/Users/ragha/Downloads/CrackDataset/TITS. That external path was not followed.",
            "TITS graph caches contain node features/labels; superpixel caches contain segment IDs. Neither substitutes for source RGB images for DeepCrack or native GT masks for scoring.",
            "DeepSegmentor/figures/roadnet contains published output figures; datasets/demo contains examples, not a complete held-out crack dataset.",
            "images.zip contains visualization assets, not another dataset."])
    save(output / "dataset_inventory.json", inventory)
    (output / "STOPPED_NO_SECOND_DATASET.json").write_text(json.dumps(dict(
        status="stopped", time=datetime.now(timezone.utc).isoformat(),
        rule="Hard rule 2: If no second dataset is available in the repo, stop and report instead of choosing one.",
        held_out_pass_started=False, training_started=False), indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    diagnosis, checkpoints, historical_validation = [], [], []
    figures = output / "diagnosis"
    figures.mkdir(exist_ok=False)
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    for ax, seed in zip(axes, [42, 1, 2]):
        case = next(c for c in lock["cases"] if c["model"] == "M4" and c["seed"] == seed)
        descriptor = case["descriptor"]
        training_path = Path(descriptor["training"])
        epochs_path = training_path.with_name("epochs.jsonl")
        training, rows = read(training_path), [json.loads(line) for line in epochs_path.read_text().splitlines()]
        source_paths.extend([training_path, epochs_path])
        for p in [training_path, epochs_path]:
            initial_hashes[str(p)] = digest(p)
        max_f1 = max(row["val_f1"] for row in rows)
        peaks = [row["epoch"] for row in rows if row["val_f1"] == max_f1]
        record = dict(seed=seed, run=training_path.parent.parent.name, epochs=len(rows),
            logged_f1_kind="node-label positive-class F1 (argmax logits); not native pixel F1",
            node_f1_peak_epochs=peaks, node_f1_peak=max_f1,
            selected_checkpoint=descriptor["checkpoint"],
            pixel_f1_peak_epoch=None, cldice_peak_epoch=None, strict_cr_peak_epoch=None,
            connectivity_after_pixel_f1_peak="unknown: native per-epoch metrics were not logged",
            historical_wall_seconds=training["wall_seconds_including_setup"],
            logged_fields=sorted(rows[0]), source=str(epochs_path))
        diagnosis.append(record)
        save(figures / f"seed_{seed}_epochs.json", rows)
        epoch = [row["epoch"] for row in rows]
        loss_ax = ax.twinx()
        line1, = ax.plot(epoch, [r["val_f1"] for r in rows], color="#1764ab", label="Logged validation node F1")
        line2, = loss_ax.plot(epoch, [r["train_loss"] for r in rows], color="#d16b16", label="Train loss")
        ax.axvline(peaks[-1], color="#1764ab", linestyle=":", alpha=.6)
        ax.set(xlabel="Epoch", ylabel="Node F1", ylim=(0, 1), title=f"M4 seed {seed} ({record['run']}); node F1 peak epoch {peaks[-1]}")
        loss_ax.set_ylabel("Training focal loss")
        ax.grid(alpha=.2)
        ax.legend(handles=[line1, line2], loc="upper right")
        ax.text(.02, .05, "Native pixel F1 / clDice / strict CR: not recorded per epoch", transform=ax.transAxes,
                bbox=dict(facecolor="white", alpha=.9, edgecolor=".8"), fontsize=9)
        for checkpoint in sorted(training_path.parent.glob("checkpoint_epoch_*.pt")):
            checkpoints.append(dict(seed=seed, path=str(checkpoint), sha256=digest(checkpoint),
                bytes=checkpoint.stat().st_size, selected=str(checkpoint)==descriptor["checkpoint"],
                role="historical only; no new checkpoints"))
        historical_validation.append(dict(seed=seed, recipe=case["recipe"],
            metrics=case["validation"], checkpoint_sha256=descriptor["checkpoint_sha256"], role="historical validation only"))
    fig.suptitle("Archived M4 training logs — native connectivity history unavailable", fontsize=14)
    fig.savefig(figures / "m4_archived_training.png")
    fig.savefig(figures / "m4_archived_training.pdf")
    plt.close(fig)
    for case in historical_validation:
        selected = next(c for c in checkpoints if c["seed"] == case["seed"] and c["selected"])
        assert selected["sha256"] == case["checkpoint_sha256"], "Historical checkpoint hash mismatch"
    save(output / "diagnosis.json", diagnosis)
    save(output / "checkpoint_manifest.json", dict(new_checkpoints=[], historical=checkpoints))
    save(output / "historical_validation.json", historical_validation)
    save(output / "historical_config_reference.json", dict(
        status="reference only; not a new selected recipe", source=str(old / "locked_configs.json"),
        source_sha256=initial_hashes[str(old / "locked_configs.json")],
        cases=[c for c in lock["cases"] if c["model"] in ["M4", "M2", "DeepCrack raw", "DeepCrack processed"]],
        frozen_grids=lock["grids"]))
    save(output / "locked_configs.json", dict(status="NOT_LOCKED_STOPPED_NO_SECOND_DATASET",
        winning_recipe=None, inference_thresholds=None, held_out_authorized=False,
        reason="No new training, validation selection, or held-out confirmation performed",
        intended_selection=dict(checkpoint_metric="mean(validation cldice, validation strict_cr)",
            replacement_min_strict_cr_gain=.01, cldice_must_not_fall=True),
        split_sha256=digest(output / "frozen_split.json")))
    trials = [dict(trial=trial, seed=seed, status="skipped", reason="Mandatory stop: no eligible second repository-local dataset")
        for trial, seed in [("selection_fix",42),("selection_fix_augmentation",42),
            ("best_previous_lower_lr_warmup",42),("locked_winner",1),("locked_winner",2),
            ("optional_M2_selection_fix",42),("held_out_confirmation",None),
            ("validation_hysteresis_tradeoff",None)]]
    save(output / "trial_records.json", dict(training_runs_attempted=0, trials=trials,
        operational_failures=[
            "Initial root-relative previous-results lookup failed; resolved to pipeline-crack-gnn/results/paper_v2/improve_m2_m4_20261014_1112_02.",
            "PowerShell wildcard arguments to rg and nonexistent optional source paths returned lookup errors; resolved by direct source paths and metadata inventory.",
            "Initial M4_selected/seed_42 log lookup failed; the historical lock correctly maps seed 42 to M4_finer600.",
            "py -3.11 launcher returned No installed Python found; direct Python311/python.exe worked."] ))

    # Copy only complexity/parameter information from the archived report. No
    # consumed-test scores or generalization gaps are used in this partial report.
    report_lines = (old / "RESULTS.md").read_text(encoding="utf-8").splitlines()
    complexity = []
    for i, line in enumerate(report_lines):
        if line.startswith("| Model | Actual graph nodes"):
            complexity = report_lines[i:i+8]
    parameters = {}
    for i, line in enumerate(report_lines):
        if line.startswith("| Model | Seeds |") and "Parameters" in line:
            for row in report_lines[i+2:]:
                if not row.startswith("|"):
                    break
                columns = [c.strip() for c in row.strip("|").split("|")]
                parameters[columns[0]] = columns[5]
    save(output / "historical_compute_reference.json", dict(parameters=parameters,
        complexity_table=complexity, source=str(old / "RESULTS.md"),
        provenance="Historical model-cost measurements, not new held-out evidence; graph counts depend on input"))
    text = ["# M4 training fix — partial report", "",
        "**Stopped before retraining or held-out evaluation: no eligible second public crack dataset is available inside the repository.** This follows hard rule 2. No new empirical support for either pre-registered claim was obtained.", "",
        "Claims were written to `claim.md` before this audit. Existing checkpoints/results were only read. No dataset image or mask was decoded, no model inference was run, and the consumed 237-image DeepCrack test pass was not rerun or used for selection or reported as new evidence.", "",
        "## Dataset gate", "",
        "`dataset_inventory.json` records a recursive filesystem inventory, including ignored files and archive member names. Available paired data are DeepCrack (300 training-source pairs and 237 consumed test pairs) and another DeepCrack directory with those same counts. Another directory of the same named dataset does not satisfy the different-dataset requirement.", "",
        "TITS has 94 cached graphs and 100 superpixel maps, plus one test graph/map. The preprocessing source saves node features/labels and segment maps; it does not preserve the original RGB images and native masks needed for the requested M4/M2/DeepCrack comparison. Source paths point outside the repository to `C:/Users/ragha/Downloads/CrackDataset/TITS`; no external data was selected or accessed. RoadNet assets are figures and demo examples; its dataset directory is absent. `images.zip` contains figures only.", "",
        "## Diagnosis from archived logs", "",
        "The historical trainer's `val_f1` concatenates graph-node labels and argmax predictions. It is **node F1**, not native-resolution pixel F1. The three selected focal/600-segment M4 checkpoints peak at epochs **22, 1, 8** for seeds **42, 1, 2**. The stated 1/3/8 history does not match the actual final selected M4 lock; epoch 3 belongs to several rejected seed-42 WCE trials.", "",
        "Per-epoch native pixel F1, clDice and strict CR were never logged. Checkpoints were saved only on node-F1 improvements, so a complete connectivity history cannot be reconstructed for unsaved epochs. Whether connectivity continues improving after the pixel-F1 peak is **unknown**. Missing metrics are explicitly labeled in the plot rather than fabricated.", "",
        "![Archived M4 loss and validation node F1](diagnosis/m4_archived_training.png)", "",
        "Training loss and node F1 share the epoch axis, with separate scales. `diagnosis.json` and per-seed epoch JSON files preserve the exact logged values.", "",
        "| Seed | Run | Logged epochs | Node F1 peak epoch | Peak node F1 | Native pixel F1 / clDice / strict CR peak |", "| --- | --- | --- | --- | --- | --- |"]
    for d in diagnosis:
        text.append(f"| {d['seed']} | {d['run']} | {d['epochs']} | {d['node_f1_peak_epochs']} | {d['node_f1_peak']:.6f} | Unavailable |")
    text += ["", "## Archived validation reference", "",
        "These are the previously locked processed M4 validation results, provided solely as baseline provenance. They are neither before/after results from this task nor held-out evidence. There are no new validation trials or winners.", "",
        "| Seed | Validation strict CR | Validation clDice | Validation pixel F1 |", "| --- | --- | --- | --- |"]
    for c in historical_validation:
        m = c["metrics"]
        text.append(f"| {c['seed']} | {m['strict_cr']:.6f} | {m['cldice']:.6f} | {m['pixel_f1']:.6f} |")
    text += ["", "The seed-42 split is frozen in `frozen_split.json`: 210 train / 45 validation / 45 unused. Training randomness seeds would be 42, 1, 2. Historical configs, thresholds and frozen grids are preserved as references, but `locked_configs.json` explicitly records **no new lock**.", "",
        "## Metrics and compute", "",
        "The required inference protocol remains 256×256 model input with native-resolution scoring. Strict CR counts every 8-connected GT component, requires at least 50% GT-pixel coverage, and requires exactly one connected component of the GT/prediction intersection. It aggregates preserved/total components across images. The earlier paper_v2 topology metric excludes components below 20 pixels and accepts coverage by one predicted component without requiring a connected intersection. These definitions differ; a bridge outside GT may improve prediction connectivity without improving strict CR.", "",
        "clDice is the mean per-image skeleton overlap score with the historical empty-mask convention. Pixel F1 pools TP/FP/FN across images. Size bins remain [0,50), [50,200), [200,inf), in native GT-component area pixels.", "",
        "Historical compute references follow; these are unchanged-architecture provenance, not measurements on a fresh dataset. Both GFLOPs columns use 2×MACs; the analytic graph estimate includes attention matrix products omitted by some fvcore operators. Costs exclude CPU morphology and input-dependent graph preprocessing.", ""]
    text += complexity
    text += ["", "| Historical model | Parameters |", "| --- | --- |"]
    text += [f"| {k} | {v} |" for k,v in parameters.items()]
    text += ["", "## Negative findings and skipped work", "",
        "- The repository does not contain a usable independent dataset for the mandatory confirmation pass.",
        "- Historical logs do not test the delayed-connectivity hypothesis: all three requested native per-epoch metrics are absent.",
        "- The prompt's reported epoch peaks and pixel-F1 checkpoint rule conflict with the final selected run artifacts and trainer source.",
        "- Selection-fix, augmentation, lower-LR/warmup trials, three-epoch timing gates, winner selection and seed retraining were skipped under the dataset stop rule.",
        "- Optional M2 retraining, validation hysteresis trade-off sweeps, held-out raw/processed tables, small-crack confirmation, sample SD and validation-minus-held-out gaps were skipped. No held-out values exist to summarize; old consumed-test numbers are not substituted.",
        "- No new training run failed because none was started. Operational lookup/launcher failures and resolutions are recorded in `trial_records.json`.", "",
        "## Pre-registered verdicts", "",
        "**Claim 1 — inconclusive (not supported by this experiment).** No connectivity-selected M4 was trained and no independent confirmation was permitted. Historical node-F1 logs cannot establish how native clDice or strict CR evolved after a pixel-F1 peak. The hypothesis remains untested; no positive claim is made.", "",
        "**Claim 2 — inconclusive (not supported by this experiment).** No fresh raw-M4/raw-DeepCrack or processed-M4/processed-DeepCrack comparison was run, and [0,50) connectivity has no new held-out result. Historical model costs do not establish fresh-data accuracy. This task makes no claim that M4 matches or beats raw or processed DeepCrack.", "",
        "## Artifacts and reproduction", "",
        "`claim.md`, `dataset_inventory.json`, `diagnosis.json`, `diagnosis/`, `frozen_split.json`, `historical_validation.json`, `historical_config_reference.json`, `locked_configs.json`, `trial_records.json`, `checkpoint_manifest.json`, `historical_compute_reference.json`, `source_manifest.json`, and `execution.json` record this partial audit. Checkpoint hashes are verified against the old lock. `REPRODUCE.ps1` reproduces only this metadata audit into a fresh timestamped output; it cannot train or evaluate.", "",
        "The audit script intentionally has no model-loading, training, threshold-search or held-out path. To undertake the original full experiment, the missing eligible dataset prerequisite must first be resolved by the user; this report does not silently relax that rule.", ""]
    (output / "RESULTS.md").write_text("\n".join(text), encoding="utf-8")
    reproduction = f'''$auditStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$auditOutput = Join-Path '{root}' ('results\\paper_v2\\m4_training_fix_' + $auditStamp + '_audit_repro')
& '{sys.executable}' '{here / 'audit_partial.py'}' --output $auditOutput
if ($LASTEXITCODE -ne 0) {{ throw 'Partial audit reproduction failed' }}
'''
    (output / "REPRODUCE.ps1").write_text(reproduction, encoding="utf-8")
    for source, before in initial_hashes.items():
        assert digest(Path(source)) == before, f"Historical source changed: {source}"
    initial_hashes[str(here / "audit_partial.py")] = digest(here / "audit_partial.py")
    initial_hashes[str(output / "claim.md")] = digest(output / "claim.md")
    save(output / "source_manifest.json", dict(sha256=initial_hashes, historical_sources_unchanged=True))
    save(output / "execution.json", dict(status="partial_stopped", command=[sys.executable, str(here / "audit_partial.py"), "--output", str(output)],
        cwd=str(root), python=sys.version, platform=platform.platform(),
        started_after_preregistration=True, audit_wall_seconds=time.perf_counter()-start,
        training_runs=0, new_inference_runs=0, held_out_passes=0,
        historical_test_reused=False, finished_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(output=str(output), status="partial_stopped", diagnosis=diagnosis,
                          checkpoint_hashes_verified=len(checkpoints)), indent=2))


if __name__ == "__main__":
    main()
