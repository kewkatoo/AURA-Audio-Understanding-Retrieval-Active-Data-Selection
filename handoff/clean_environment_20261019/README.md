# Isolated-environment validation procedure

This dated supplement preserves the original handoff snapshot. It records a new virtual-environment installation on the same Windows machine, with the same Python base interpreter, source tree, pip download cache, raw/prepared data and Hugging Face model cache. It is not a new-machine, new-dataset or full training replication.

From the project root, the installation sequence used is:

```powershell
.venv/Scripts/python.exe -m venv venv/reproduction-20261019
venv/reproduction-20261019/Scripts/python.exe -m pip install --index-url https://pypi.org/simple pip==26.2.1 --disable-pip-version-check --progress-bar off
venv/reproduction-20261019/Scripts/python.exe -m pip install --index-url https://pypi.org/simple -r requirements.research.lock.txt --disable-pip-version-check --progress-bar off --report handoff/clean_environment_20261019/install_report.json
venv/reproduction-20261019/Scripts/python.exe handoff/clean_environment_20261019/validate.py
```

The initial attempt used the venv's bundled pip 22.3 and stalled while retrieving the 2.5 GB CUDA PyTorch wheel. It was stopped; the retry first installs the captured pip 26.2.1. Project dependency versions remain unchanged. Cached distribution archives are allowed; packages are freshly installed into the new venv, not copied from the original environment's site-packages.

`validate.py` records checks sequentially in `validation.json` and individual logs. It checks interpreter isolation, installed-package consistency, the original 2,060-file snapshot and all captured package versions, regression tests, all 60 saved FSDD classifiers' predictions, and raw-audio GPU inference for two recordings using the exact recorded checkpoint revision. Regression details are written to `tests.xml`.

The raw encoder check is a shape/finite/unit-norm smoke test. It does not assert numerical equality to stored arrays with a different batching layout, retrain classifiers, measure training time or establish cross-device reproducibility. Run completion and actual pass/fail outcomes must be read from `validation.json`; this procedure document alone is not evidence that checks passed.
