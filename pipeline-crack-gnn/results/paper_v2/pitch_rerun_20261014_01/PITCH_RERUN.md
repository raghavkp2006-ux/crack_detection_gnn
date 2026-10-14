# Project pitch rerun

Historical pitch checkpoints and 256x256 inference are evaluated separately from current five-seed node predictions.
No retraining, primary threshold changes, or test-set threshold fitting.

## Historical checkpoint benchmark

| Model | clDice | Strict micro CR | Pixel F1 | Parameters | GFLOPs fvcore | Model median ms | Historical E2E median ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | 0.6875265677870454 | 0.28125 | 0.6638805130071302 | 60930 | 0.034920448 | 5.715899998904206 | 102.36745000293013 |
| M2 | 0.7452905251641695 | 0.4223790322580645 | 0.6478326492683272 | 10626 | 0.006066688 | 1.2673000019276515 | 94.04289999656612 |
| M3 | 0.544586404783426 | 0.17036290322580644 | 0.5680341662903293 | 110594 | 0.106388992 | 2.902599997469224 | 78.52015000389656 |
| M4 | 0.740701217040343 | 0.29939516129032256 | 0.7278926739266397 | 116802 | 0.109940224 | 2.6116500011994503 | 66.37134999618866 |
| DeepCrack | 0.7832438708370932 | 0.33669354838709675 | 0.7826919834657892 | 14720389 | 40.178679808 | 5.243499996140599 | 5.227500005275942 |

## Historical size connectivity

| Model | Small | Medium | Large |
| --- | --- | --- | --- |
| M1 | 0.11377245508982035 | 0.2956204379562044 | 0.3248638838475499 |
| M2 | 0.24550898203592814 | 0.45255474452554745 | 0.46098003629764067 |
| M3 | 0.07784431137724551 | 0.16058394160583941 | 0.20326678765880218 |
| M4 | 0.12574850299401197 | 0.3248175182481752 | 0.3393829401088929 |
| DeepCrack | 0.0718562874251497 | 0.2518248175182482 | 0.4591651542649728 |

## Current five-seed mean size connectivity

| Model | Small | Medium | Large |
| --- | --- | --- | --- |
| M1_focal | 0.1065868263473054 | 0.1686131386861314 | 0.38656987295825773 |
| M2_focal | 0.05868263473053893 | 0.09999999999999999 | 0.3016333938294011 |
| M3_focal | 0.061077844311377236 | 0.09124087591240874 | 0.29328493647912884 |
| M4_focal | 0.05868263473053893 | 0.1 | 0.31542649727767696 |
| DeepCrack | 0.14850299401197603 | 0.2116788321167883 | 0.40435571687840294 |
| M1_wce | 0.13532934131736526 | 0.2167883211678832 | 0.4450090744101633 |

## Historical overlap sweep

| Overlap | M4 micro | DeepCrack micro | Winner |
| --- | --- | --- | --- |
| 0.3 | 0.3235887096774194 | 0.3629032258064516 | DeepCrack |
| 0.4 | 0.3074596774193548 | 0.35080645161290325 | DeepCrack |
| 0.5 | 0.29939516129032256 | 0.33669354838709675 | DeepCrack |
| 0.6 | 0.28024193548387094 | 0.3175403225806452 | DeepCrack |
| 0.7 | 0.24899193548387097 | 0.2913306451612903 | DeepCrack |

## SLIC connectivity control

Fixed historical M4 checkpoint and tau=0.66; inference-only preprocessing comparison.

| enforce_connectivity | clDice | Strict micro CR | SLIC median ms | Mean nodes | Mean edges |
| --- | --- | --- | --- | --- | --- |
| False | 0.740701217040343 | 0.29939516129032256 | 40.01375000370899 | 289.0 | 2931.3417721518986 |
| True | 0.618249401704697 | 0.2661290322580645 | 41.112700004305225 | 271.4978902953587 | 2784.6329113924053 |

## Positional encoding cost

```json
{
  "parameters": 6208,
  "real_graph_nodes": 289,
  "pe_fusion_gpu_ms": {
    "median": 0.2466000005370006,
    "q1": 0.1912500010803342,
    "q3": 0.2843250003934372,
    "iqr": 0.09307499931310304,
    "p95": 0.3982699912739917,
    "mean": 0.25698000015108846,
    "sd": 0.08816636505971115,
    "timed_iterations": 500,
    "warmup": 15
  },
  "percent_of_historical_m4_e2e": 0.37154585608272467,
  "note": "Direct PE module timing includes sinusoidal construction, concatenation, and projection; M3/M4 whole-model timing difference is not a pure PE measurement."
}
```

## Protocol notes

- Historical baseline: DeepSegmentor DeepCrackNet with batch normalization, seed-42 checkpoint. Current baseline: separate VGG16/HED DeepCrack-style reimplementation.
- Historical fixed thresholds M1=.60, M2=.70, M3=.74, M4=.66, DeepCrack=.36 are unchanged; current thresholds loaded unchanged from saved NPZ files.
- Strict historical CR requires sufficient GT overlap AND exactly one connected intersection; includes every GT component without a minimum-area cutoff. This differs from paper_v2 topology definition.
- Historical size-bin implementation is [0,50), [50,200), [200,infinity), despite its printed medium/large labels; preserved exactly for reproduction.
- Historical E2E function resizes outside timing; CNN input normalization/transfer and binary-mask reconstruction are excluded. These times are not full native-resolution RGB-to-mask latency.
- Historical model-only timing/FLOPs uses synthetic graph tensors, despite the pitch describing real-graph profiling. Real first-image graph FLOPs are additionally measured; unsupported fvcore ops are recorded.
- Connectivity switch is an inference-only sensitivity check using one fixed trained checkpoint; it does not establish a retraining-based causal improvement.
- Cross-sensor and morphology experiments appear as proposed future work in the pitch; no new sensor experiment or retraining is included.
- Python launcher could not locate Python inside sandbox; requested py -3.11 works with approved elevated access.
- Current five-seed supplementary analysis uses existing probabilities, recomputes all pixel masks and strict historical topology metrics, and leaves primary predictions intact.

## Wall times in seconds

```json
{
  "metric_tests": 0.7374773999908939,
  "historical_M1": 86.8812154000043,
  "historical_M2": 77.76275320000423,
  "historical_M3": 65.72214410000015,
  "historical_M4": 54.72585170000093,
  "historical_DeepCrack": 9.977960599993821,
  "slic_control": 108.06135519999953,
  "pe_cost": 0.19637230000807904,
  "current_five_seed": 51.92821249998815,
  "total": 458.6033415999991
}
```
