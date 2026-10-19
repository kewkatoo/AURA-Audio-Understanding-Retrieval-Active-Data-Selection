# Relocated verification of v5 and v6 evidence

This supplement extends the v7 path-mapping approach to the independent AudioMNIST study, the Speech Commands duplicate control, and the AudioMNIST acquisition-cost development study. It leaves frozen source, manifests and saved results unchanged.

From the project root, using the validated clean environment:

```powershell
venv/reproduction-20261019/Scripts/python.exe handoff/portable_earlier_studies/verify.py --root venv/relocated-evidence-20261019 --stage audiomnist
venv/reproduction-20261019/Scripts/python.exe handoff/portable_earlier_studies/verify.py --root venv/relocated-evidence-20261019 --stage duplicate-control
venv/reproduction-20261019/Scripts/python.exe handoff/portable_earlier_studies/verify.py --root venv/relocated-evidence-20261019 --stage cost
```

Replace `--root` with the copied project's absolute location when needed. The copy must include all original 2,060 snapshotted files and the original snapshot/checksum. It additionally needs the prepared `data/audiomnist_v5/` and `data/scaling_v4/` artifacts, plus the AudioMNIST and Speech Commands source recordings referenced by their saved partition declarations. Preserve Git-ignored arrays and raw data; a Git checkout is insufficient.

Keep `handoff/portable_verification/` alongside this utility: its mapper is reused and its source checked against that earlier supplement's checksum inventory. Package/Python versions must match the captured environment. Each process imports the frozen semantic verifier and scaling helper from the copied source tree and maps only their filesystem lookups. The original snapshot is checked before and after analysis.

For v5, the original verifier attempts to refresh `verification.json` and `experiment.json`. The wrapper translates generated report path keys back to the original root solely for comparison, requires the generated content to match the frozen evidence, and suppresses both writes. Unexpected writes or changed report content fail. V6 calls its existing read-only `verify` function directly and does not invoke report generation.

The v5 checks cover raw/prepared artifact hashes, source provenance, split/content separation, locked job sets, subset membership/class budgets, saved prediction cohorts and recomputed metrics. The v6 checks add canonical/fresh embedding drift, pre-encoding selections, replay hashes, numerical-audit consistency and recorded cost-component sums. These earlier verifiers do not execute every saved classifier to reproduce predictions; that stronger saved-head check is available for v7. No models are fitted and no timing is repeated here.

Combined regression and supplementary tests:

```powershell
venv/reproduction-20261019/Scripts/python.exe -m pytest -q tests handoff/portable_verification/test_paths.py handoff/portable_earlier_studies/test_reports.py -o cache_dir=venv/portable-earlier-pytest-cache
```

Observed results: all three stages passed, covering 182 locked model jobs and 83,269 raw-file hash checks across studies (60,769 distinct copied source recordings). The combined suite passed 112 tests, with one optional network test skipped. A missing prepared-array test also failed at the copied path while the original array remained accessible; the copied array was restored and its hash checked afterwards.

This is directory relocation on the same Windows machine, not a new-machine, different-OS, full training or portable-resume result. Actual outcomes are recorded in `validation.json` and per-stage logs.
