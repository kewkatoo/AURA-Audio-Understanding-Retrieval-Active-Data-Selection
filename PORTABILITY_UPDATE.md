# Saved-evidence portability addendum

Completed 2026-10-19. V7 FSDD saved-model verification now works from a physically copied project directory without rewriting legacy absolute paths in frozen manifests.

The new read-only wrapper translates filesystem lookups into the copied tree and rejects access outside it. It preserves the original manifest bytes, hash identities, splits and saved models. The original 2,060-file snapshot remains unchanged; additional utility code, tests and evidence are recorded in this separate supplement.

Validation passed for 2,060 baseline file hashes, 3,000 copied audio hashes, six folds and all 60 saved classifiers' predictions. A missing copied audio file caused verification to fail even while its original counterpart remained accessible. The copied file was restored after the negative test. The combined suite passes 110 tests, with one optional network test skipped.

Run:

```powershell
venv/reproduction-20261019/Scripts/python.exe handoff/portable_verification/verify.py --root venv/relocated-evidence-20261019
```

See `handoff/portable_verification/README.md` for required files, test commands and scope, and `handoff/portable_verification/validation.json` for recorded outcomes. The supplement has its own file hashes in `handoff/portable_verification/artifacts.json` and checksum alongside it.

This closes a specific saved-evidence relocation check on Windows. It does not establish portable training/resume, port the older v5/v6 scientific verifiers, reproduce training time, or test a different operating system or machine. The earlier clean-environment result remains recorded in `VALIDATION_UPDATE.md`; the scientific accuracy/cost conclusions are unchanged.
