# Clean-environment validation addendum

Completed 2026-10-19. The previously unvalidated clean virtual-environment installation now passes on the existing Windows machine. This addendum updates the clean-install limitation in the original handoff documents; their original bytes and 2,060-file checksum snapshot are preserved.

| Check | Result |
|---|---|
| Fresh isolated Python environment | Pass; installed into `venv/reproduction-20261019`, without copying the original environment's site-packages |
| Captured Python and 64 distribution versions | Exact match |
| Dependency consistency | `pip check` passes |
| Original source/result integrity | All 2,060 snapshotted files pass |
| Regression suite | 102 passed, one optional network test skipped; one expected short-input FFT warning |
| Saved FSDD model verification | All 60 classifiers reproduce their saved predictions; 3,000 source-audio hashes and experiment provenance verified |
| Raw-audio GPU smoke test | Two recordings produce finite, unit-normalized 768-dimensional vectors using the recorded wav2vec2 revision on CUDA |

## Installation finding

The virtual environment initially bundled pip 22.3. The first dependency installation stalled while retrieving the 2.5 GB CUDA PyTorch wheel and was stopped. Installing the captured pip 26.2.1 first allowed the retry to reuse a cached distribution archive and complete. No research package versions were changed.

Use the explicit pip bootstrap in `handoff/clean_environment_20261019/README.md` before installing the captured requirements. The new `install_report.json` records source URLs and archive SHA-256 hashes for 63 installed distributions; pip itself was installed separately in the bootstrap. This report records the completed installation, not an offline wheel bundle or a complete installer enforcing wheel hashes.

## Scope of this validation

The environment was newly created and its packages freshly installed, with existing pip download caches permitted. Source, operating system, Python base interpreter, GPU, raw/prepared data and the Hugging Face model cache were shared with the original run. No experiment was retrained or retimed, no held-out score was used for tuning, and no original research artifact was rewritten by the validation checks.

This closes the same-machine clean-venv installation and compatibility check. It does not establish a clean-machine or second-device replication, availability without caches/network, cross-layout numerical identity, or recovery of legacy absolute-path manifests after relocation. The original FSDD result is unchanged: a 61.78% median paired warm training-time saving with a 2.327-point accuracy loss, missing the one-point target.

The next research extension, if accuracy-preserving compute savings remain required, is a budget/stopping rule developed without observed holdout scores and then confirmed on new speakers. Annotation acquisition, quality calibration and retrieval/API delivery remain separately scoped goals.

## Evidence and repeat command

Detailed outcomes are in `handoff/clean_environment_20261019/validation.json`, with individual logs and `tests.xml`. The reusable validation procedure is:

```powershell
venv/reproduction-20261019/Scripts/python.exe handoff/clean_environment_20261019/validate.py
```

That command reruns compatibility checks and updates this supplement's logs; it does not fit models. The original archive check remains:

```powershell
.venv/Scripts/python.exe scripts/verify_research_handoff.py --environment
```

The supplement's own file hashes are recorded separately in `handoff/clean_environment_20261019/artifacts.json`, with its checksum alongside it. The original snapshot does not cover files added after handoff. Preserve both inventories when archiving these results.
