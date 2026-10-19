# Read-only verification after relocation

Completed 2026-10-19. `verify.py` verifies the frozen v7 FSDD study from a copied project tree while preserving every original JSON manifest and hash. It uses the original verifier, with only its filesystem path construction and file-hash lookups translated in memory.

```powershell
venv/reproduction-20261019/Scripts/python.exe handoff/portable_verification/verify.py --root venv/relocated-evidence-20261019
```

For another copied tree, replace `--root` with its absolute path. Use an interpreter matching the captured Python/package versions. The copied root must contain all original snapshotted source/results/configuration files, `handoff/research_snapshot.json`, its checksum and FSDD's original README and recordings under `data/raw/FSDD`. Copy the complete preserved files, including Git-ignored arrays; a Git checkout alone is insufficient. The external wav2vec2 checkpoint cache is not needed for this saved-classifier verification.

The verifier identifies the original Windows project root from the frozen v7 protocol. It maps paths under that root into the copy, permits native paths already inside the copy, and rejects other absolute paths, drive-relative paths and traversal outside the copy. Missing copied files raise errors; it never falls back to the original directory. It requires the reporter to be imported from the copied tree, verifies the original snapshot and package versions before analysis, and rechecks the snapshot afterwards. It prints results without writing reports or fitting models.

Validated on a physical same-machine copy containing 2,062 baseline/snapshot files plus FSDD audio. Verification checked all 2,060 baseline hashes, all 3,000 raw hashes, six folds and all 60 saved models' predictions, including 36 primary configurations. A deliberate missing-copy test failed at the copied audio path while the corresponding original file remained accessible; the copied file was restored afterwards.

Run the combined regression/path tests explicitly:

```powershell
venv/reproduction-20261019/Scripts/python.exe -m pytest -q tests handoff/portable_verification/test_paths.py -o cache_dir=venv/portable-pytest-cache
```

Observed result: 110 passed, one optional network test skipped, one expected short-input FFT warning. The default project test discovery continues to cover the original baseline suite; these supplementary tests live outside the frozen source inventory so that its checksums remain valid.

This is a Windows directory-relocation check for v7 saved evidence. It does not port v5/v6 semantic verifiers, make training/checkpoint resume relocatable, repeat training/timing, establish Linux compatibility, or validate another machine. Original manifests remain immutable; `PORTABILITY_UPDATE.md` records this additional evidence separately.
