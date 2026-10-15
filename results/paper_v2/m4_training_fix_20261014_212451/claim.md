# Pre-registered M4 training-fix claims

1. Selecting the M4 checkpoint by validation clDice / strict CR instead of pixel F1 improves M4's connectivity on a held-out dataset.
2. M4 matches or beats RAW DeepCrack on strict CR and small-crack ([0,50)) connectivity on a held-out dataset, at far lower compute.

Any result that does not support these is reported as a negative result. No claims will be added after seeing results.

Selection of every config, epoch, and threshold uses validation only. The previous experiment's consumed 237-image test pass will not be rerun or used for selection or new evidence. The training split remains the seed-42 210 / 45 / 45 split; seeds 42, 1, 2 change training randomness only.

Held-out confirmation is permitted once, only on a different public crack dataset already present in this repository, using locked configurations and thresholds. If none is available, stop and report. If a mandatory step cannot fit within the nine-hour total budget, stop and report partial results without running held-out confirmation.

Runs are sequential, change one thing at a time, never overwrite previous artifacts, and include all failures. Each trial is stopped after timing its first three epochs if its projected duration exceeds 40 minutes. Historical 256x256 inference, native-resolution scoring, strict CR, clDice, pixel F1, and size connectivity are preserved.

Validation replacement rule: strict CR improves by at least 1.0 percentage point and clDice does not fall. Checkpoint selection uses the mean of validation clDice and strict CR with patience measured on that mean.
