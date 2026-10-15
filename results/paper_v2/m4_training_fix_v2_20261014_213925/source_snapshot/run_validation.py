"""Run every mandatory validation trial and lock; deliberately stop before TITS.

The first seed-42 A run was dispatched separately to inspect the timing gate.
This continuation preserves that completed run and its first setup audit.
"""
import json
from pathlib import Path
import sys
import traceback

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]
sys.path.insert(0, str(ROOT / 'pipeline-crack-gnn/paper_v2'))
from m4_training_fix_v2 import Study, append, utc

if __name__ == '__main__':
    study = Study(OUTPUT)
    append(OUTPUT/'commands.jsonl',dict(time=utc(),argv=sys.argv,executable=sys.executable,cwd=str(Path.cwd())))
    # The first setup was read-only. Archive its audit rather than overwriting it
    # when the continuing process independently reloads the frozen graph inputs.
    source = OUTPUT/'training_inputs.json'
    if source.exists():
        archive = OUTPUT/'first_setup_training_inputs.json'
        if archive.exists():
            raise FileExistsError('Setup audit already archived; inspect before resuming')
        source.rename(archive)
    try:
        study.train_all()
        print('VALIDATION COMPLETE AND LOCKED; TITS not yet scored',flush=True)
    except Exception as exc:
        append(OUTPUT/'failures.jsonl',dict(time=utc(),stage='validation_continuation',error=str(exc),traceback=traceback.format_exc()))
        study.report(str(exc))
        raise
