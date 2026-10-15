"""Execute registered D/seed-2 once while earlier trials validate; no TITS access."""
from pathlib import Path
import sys
import traceback

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'pipeline-crack-gnn/paper_v2'))
from m4_training_fix_v2 import Study, append, utc

if __name__=='__main__':
    study=Study(OUT)
    append(OUT/'commands.jsonl',dict(time=utc(),argv=sys.argv,executable=sys.executable,
        operation='registered D seed2 in isolated process; parent runner reuses completed artifacts'))
    source=OUT/'training_inputs.json'
    archived=OUT/'training_inputs_before_parallel_D2.json'
    assert source.resolve().is_relative_to(ROOT) and archived.resolve().is_relative_to(ROOT)
    if source.exists():
        if archived.exists():raise FileExistsError('Parallel setup already dispatched')
        source.rename(archived)
    try:
        study.preflight();study.prepare()
        study.train('D',2);study.calibrate('D',2)
        print('D SEED2 COMPLETE; no TITS scoring or configuration selection performed',flush=True)
    except Exception as exc:
        append(OUT/'failures.jsonl',dict(time=utc(),stage='parallel_D_seed2',error=str(exc),traceback=traceback.format_exc()))
        raise
