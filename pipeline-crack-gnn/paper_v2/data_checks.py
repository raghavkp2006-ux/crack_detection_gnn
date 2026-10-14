"""Read-only provenance/source checks; exclusively create new audit JSON outputs."""
import hashlib
import importlib.metadata
import json
import re
import subprocess
import time
from pathlib import Path

from common import BASE, OUT, DC, SEEDS, stem
from src.utils.dataset import get_splits
from src.train import load_config


def main():
    start = time.perf_counter()
    out = Path(OUT)
    for name in ('source_overlap.json', 'data_checks.json'):
        if (out / name).exists(): raise FileExistsError(out / name)
    train = sorted(Path(BASE, 'data/graphs/train').glob('*.pt'))
    test = sorted(Path(BASE, 'data/graphs/test').glob('*.pt'))
    assert len(train) == 300 and len(test) == 237
    source = lambda p: re.sub(r'-\d+$', '', p.stem)
    tr_ids, te_ids = {source(p) for p in train}, {source(p) for p in test}
    for part, graphs in [('train', train), ('test', test)]:
        raw = {p.stem for p in Path(DC, f'{part}_img').iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}}
        assert raw == {p.stem for p in graphs}, f'Raw/graph stems differ for {part}'
    overlap = [p.stem for p in test if source(p) in tr_ids]
    result = dict(source_id_rule='remove trailing -<digits> only', n_train_folder_images=len(train),
                  n_test_folder_images=len(test), n_train_source_ids=len(tr_ids), n_test_source_ids=len(te_ids),
                  test_source_overlap_count=len(overlap), test_source_overlap_fraction=len(overlap)/len(test),
                  overlapping_test_stems=overlap, seeds={})
    for seed in SEEDS:
        tr, va, _ = get_splits(train, 0.7, 0.15, seed)
        ids = {source(train[i]) for i in tr}
        matched = [train[i].stem for i in va if source(train[i]) in ids]
        result['seeds'][str(seed)] = dict(n_train=len(tr), n_val=len(va), val_overlap_count=len(matched),
                                          val_overlap_fraction=len(matched)/len(va), overlapping_val_stems=matched)
    with (out/'source_overlap.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    with Path(BASE,'results/full_evaluation_summary.json').open() as f: full=json.load(f)
    provenance = {}
    for name,cfg,checkpoint,loss in [
        ('M1 (8-Layer Deep GNN)','configs/baseline_deep_gnn.yaml','results/M1_deep_gnn/best_model.pt','weighted_ce'),
        ('M2 (2-Layer Shallow GNN)','configs/step2_loss/shallow_gnn_focal.yaml','results/step2_loss/M2_shallow_gnn/best_model.pt','focal'),
        ('M4 (Hybrid GNN+Trans)','configs/step2_loss/hybrid_full_focal.yaml','results/step2_loss/M4_hybrid_full/best_model.pt','focal')]:
        provenance[name] = dict(**full['dirichlet_summary'][name],checkpoint=checkpoint,config=cfg,
                                configured_seed=load_config(str(Path(BASE,cfg)))['training']['seed'],loss=loss,
                                provenance_note='Checkpoint/config paths from run_full_evaluation_and_analysis.py; seed from YAML, not embedded in the summary JSON.')
    cpu = subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name)'],text=True).strip()
    versions={p:importlib.metadata.version(p) for p in ['torch','torch_geometric','torchvision','scikit-image','numpy','scipy','fvcore']}
    checks=dict(cpu_name=cpu,package_versions=versions,dirichlet_source='results/full_evaluation_summary.json',
                dirichlet_script='run_full_evaluation_and_analysis.py',dirichlet=provenance,
                dirichlet_definition='Mean over test graphs of the mean directed-edge squared distance between row-wise L2-normalized hidden vectors.',
                oracle_ceiling=dict(script='../create_all_figures.py',function='make_figS8_superpixel_ceiling',
                    image_selection='First 35 sorted test JPG images',image_processing='OpenCV BGR images and grayscale masks resized to 256x256; mask binarized >128.',
                    segments=[100,200,300,500],slic=dict(compactness=10.0,sigma=1.0,enforce_connectivity=False),
                    label_rule='Superpixel positive iff mean of resized binary GT pixels strictly exceeds 0.30.',
                    scoring='Pixel-level binary F1 on flattened GT and reconstructed mask, then mean and population SD across images.',
                    caveat='Fixed 0.30 occupancy labeling is not an optimized oracle for pixel F1; old reference model scores are hardcoded.',rerun=False),
                attention_decay_profile=dict(json_found=False,source='../create_all_figures.py:make_figS4_attention_temperature_ablation',
                    finding='Existing panel (b) simulates exp(-alpha*distance); no measured spatial attention decay JSON found. Omit panel (b).'),
                wall_seconds=time.perf_counter()-start)
    with (out/'data_checks.json').open('x',encoding='utf-8') as f:json.dump(checks,f,indent=2)
    brief={k:v for k,v in result.items() if k not in ('overlapping_test_stems','seeds')}
    brief['seeds']={k:{a:b for a,b in v.items() if a!='overlapping_val_stems'} for k,v in result['seeds'].items()}
    print(json.dumps(brief,indent=2));print(json.dumps(checks,indent=2))
    with Path(BASE,'paper_v2/NOTES.md').open('a',encoding='utf-8') as f:
        f.write('\n## Prompt 4 — 2026-10-14: Task A\n\nTask A source overlap and provenance are in source_overlap.json and data_checks.json.\n')
        f.write('Sandbox CIM/Python launcher access required elevation for the requested read-only checks. Initial rg literal-glob searches were corrected to -g filters. No experiment or existing result was changed.\n')
        f.write('Dirichlet seed 42 is config-derived provenance; historical summary JSON does not embed the seed. Existing attention decay panel is simulated and will be omitted. Oracle plot uses 35 resized images, occupancy >0.30 labels, and pixel F1; it is not rerun.\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n')


if __name__ == '__main__':
    main()
