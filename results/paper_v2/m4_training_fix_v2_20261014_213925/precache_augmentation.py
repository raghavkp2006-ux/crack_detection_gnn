"""Prebuild the registered graph bank while B trains; publish complete files only."""
from pathlib import Path
import sys
import json

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'pipeline-crack-gnn/paper_v2'))
from m4_training_fix_v2 import Study, read, save, sha, utc


class Prebuild(Study):
    @property
    def protocol(self):
        return read(OUT/'protocol.json')


if __name__=='__main__':
    final=OUT/'augmented_graphs'
    if (final/'manifest.json').exists():
        print('Registered augmentation cache is already complete',flush=True)
    else:
        study=Prebuild(OUT)
        study.verify_sources()
        study.out=OUT/'augmentation_prebuild_workspace'
        study.out.mkdir(exist_ok=False)
        study.augment()
        temporary=study.out/'augmented_graphs'
        if final.exists():
            print('Main runner has begun its own cache; preserving unused prebuild without overwriting',flush=True)
        else:
            manifest=read(temporary/'manifest.json')
            for record in manifest['records']:
                assert sha(record['path'])==record['sha256']
                record['path']=str(final/Path(record['path']).name)
            (temporary/'manifest.json').rename(temporary/'manifest_prepublication.json')
            manifest['publication']=dict(time=utc(),strategy='complete directory renamed atomically; identical frozen augmentation function/settings')
            save(temporary/'manifest.json',manifest)
            temporary.rename(final)
            print(f'PUBLISHED {len(manifest["records"])} registered augmented graphs',flush=True)
