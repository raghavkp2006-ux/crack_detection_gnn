# Pre-registered M4 training-fix v2 claims

1. A stabilised M4 training recipe (no node-F1 early stopping, LR decay, native-metric checkpoint selection) improves M4 validation clDice and strict CR over the current M4 on DeepCrack validation.
2. On held-out TITS (zero-shot, locked thresholds), M4 matches or beats RAW DeepCrack on strict CR and small-crack [0,50) connectivity.

Anything else is exploratory. No claims will be added after seeing results.

All recipe, epoch and threshold selection uses the frozen DeepCrack seed-42 split (210 train / 45 validation / 45 unused). The consumed 237-image DeepCrack test set is excluded. Training randomness seeds 42, 1, 2 do not change split membership. Every candidate uses all three seeds, and replacement requires at least +0.010 mean validation strict CR with no mean clDice fall.

Baseline A: historical focal/600-segment architecture and schedule, fixed 100 epochs, no early stopping; select epoch with trailing 5-epoch moving average of (native clDice + strict CR)/2 at raw threshold 0.5. B adds effective batch 8 gradient accumulation and cosine LR 1e-3 to 1e-5. C adds cached image augmentation before graph construction. D adds EMA weights with decay 0.99. Per-epoch native metrics are evaluated at 0.5 without processing. Each run times its first three epochs and stops if projected duration exceeds 40 minutes.

TITS at C:/Users/ragha/Downloads/CrackDataset/TITS is authorized read-only. Verify image/mask availability and dataset layout before training; if missing, stop and report. Held-out scoring runs once after an immutable lock, with no subsequent configuration changes. Inference is 256x256, scoring at native resolution, retaining historical strict CR, clDice, pixel F1 and native size bins. All artifacts are new; failures and skipped work are reported. Nine-hour total budget includes implementation; if mandatory work cannot fit, stop and report partial results before held-out evaluation.
