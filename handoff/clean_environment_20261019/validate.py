"""Validate this handoff from the interpreter executing this script.

Run only after installing requirements.research.lock.txt into a new venv.
Outputs are confined to this dated validation directory; saved studies are read.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys


def main():
    output = Path(__file__).resolve().parent
    root = output.parent.parent
    environment = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    checks = []
    commands = [
        ("isolation", ["-c", "import json,sys; from pathlib import Path; print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,path=sys.path),indent=2)); assert sys.prefix != sys.base_prefix; assert not any(Path(p).name == 'site-packages' and '.venv' in Path(p).parts for p in sys.path)"]),
        ("packages", ["-m", "pip", "check"]),
        ("snapshot_and_versions", ["scripts/verify_research_handoff.py", "--environment"]),
        ("regression", ["-m", "pytest", "-q", f"--junitxml={output / 'tests.xml'}", "-o", "cache_dir=" + str(output / "pytest_cache")]),
        ("saved_v7_predictions", ["-c", "from pathlib import Path; from scripts.report_cost_confirmation import verify; verify(Path('results/cost_confirmation_v7')); print('All 60 saved v7 heads reproduce their predictions')"]),
        ("raw_encoder", ["-c", "import json; from dataclasses import asdict; import numpy as np; import pandas as pd; from aura.config import AuraConfig; from aura.encoders.wav2vec2 import Wav2Vec2Encoder; from experiments.raw_pipeline import encode_recordings; cfg=AuraConfig.from_yaml('configs/experiment.v2.yaml'); encoder=Wav2Vec2Encoder(batch_size=16); assert encoder.info.revision == '0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8'; assert encoder.info.device == 'cuda'; rows=json.load(open('results/cost_confirmation_v7/inventory.json'))[:2]; x=encode_recordings(pd.DataFrame(rows),encoder,cfg.audio); assert x.shape == (2,768) and np.isfinite(x).all() and np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4); print(json.dumps(dict(encoder=asdict(encoder.info),shape=list(x.shape),norms=np.linalg.norm(x,axis=1).tolist()),indent=2))"]),
    ]
    for name, arguments in commands:
        print(f"Checking {name}", flush=True)
        completed = subprocess.run([sys.executable, *arguments], cwd=root, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output / f"{name}.log").write_text(completed.stdout, encoding="utf-8")
        checks.append(dict(name=name,arguments=arguments,exit_code=completed.returncode))
        result = dict(date="2026-10-19",interpreter=sys.executable,
                      installed_lock_hash=hashlib.sha256((root / "requirements.research.lock.txt").read_bytes()).hexdigest(),
                      all_checks_passed=all(c["exit_code"] == 0 for c in checks),
                      complete=len(checks) == len(commands),checks=checks,
                      scope="New isolated virtual environment on the same Windows machine; shared package-download cache, source, raw data and model cache. No retraining or new hardware replication.")
        (output / "validation.json").write_text(json.dumps(result,indent=2), encoding="utf-8")
        if completed.returncode:
            print(completed.stdout, flush=True)
            raise SystemExit(completed.returncode)
    print("All clean-environment validation checks passed", flush=True)


if __name__ == "__main__":
    main()
