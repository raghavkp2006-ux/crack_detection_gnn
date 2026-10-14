# Figure captions and sources

All graph-model primary plots use focal loss. Model colors are fixed across figures. Fonts are at least 8 pt; widths are 3.5 in or 7.16 in. PNGs are 300 dpi; PDFs retain vector plotting elements.

## fig04_threshold_calibration
Mean validation F1 across the five fixed seeds; shading is the seed minimum–maximum, not a confidence interval. Dotted vertical line marks the primary grid edge 0.90. Source: sensitivity_tau.json.

## fig05_multiseed_benchmark
Bars are seed mean; error bars are ±1 sample SD (ddof=1) over five seeds. White points are all five seed values at deterministic horizontal offsets. Sources: analysis_main.json, overall.*.{f1,iou}.

## fig06_loss_comparison
Bars are mean F1 ±1 sample SD across the same five seeds for each loss/model cell. White points show every seed; hatched bars are weighted CE. Source: analysis_main.json, overall.*.f1.

## fig07_accuracy_cost_tradeoff
No error bars. F1 is the five-seed mean. Circle area is exactly 18 times analytic GMACs in points squared; thin crosshairs locate the very small graph-model circles, without adding an area offset. Graph sizes use gmacs_analytic_mean_graph; DeepCrack uses gmacs_analytic. Sources: analysis_main.json and efficiency.json.

## fig09_topology_metrics
clDice intervals are 95% percentile scene-cluster-bootstrap CIs with 2000 resamples. Connectivity bars show seed means without uncertainty bars. Sources: analysis_cluster.json, topology.models; topology.json, models.*.conn_micro_seed_mean and conn_macro_seed_mean. M1 WCE is the secondary comparison.

## fig10_morphology_stratified_f1
Bars show pooled node F1 averaged over seeds. Error bars are 95% percentile cluster-bootstrap CIs, resampling only source scenes containing images of each bucket and retaining that bucket’s images from sampled scenes. Source: analysis_cluster.json, buckets.

## fig14_efficiency_stages
DeepCrack bar is a median; graph-model bars are means. No error bars. Stage hatching encodes SLIC, features, graph construction, H2D/forward/D2H, and reconstruction; the transfer/forward stage is not model-only GPU latency. Model color remains fixed. Source: efficiency.json, models.*.stage_mean_ms and models.DeepCrack.e2e_ms.median.

## fig11_qualitative_results
No uncertainty bars or overlays. Columns show native RGB, white-crack GT, and node predictions reconstructed from seed 42 probabilities. Every image retains its native aspect ratio; blank space is intentional. Bucket rows use the closest-to-median seed-averaged M4 image, and failure uses the global minimum. Ties use the first stored sorted stem; duplicates are not replaced. Sources: test_labels.npz, test_buckets.json, probs/*/seed_42.npz, cached test superpixels, native DeepCrack test images/masks; qualitative_selection.json records stems and all primary-model F1 values.

Figure-generation wall seconds: 2.6944779999903403
