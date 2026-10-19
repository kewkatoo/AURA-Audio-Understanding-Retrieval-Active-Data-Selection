# Earlier-study portability addendum

Completed 2026-10-19. Relocation verification now passes for the independent AudioMNIST study, Speech Commands duplicate control and acquisition-cost development study. This extends the earlier v7-only portability result without rewriting frozen manifests or research source.

| Relocated study | Locked models checked | Raw-file hash checks |
|---|---:|---:|
| AudioMNIST independent study, v5 | 101 | 30,000 |
| Speech Commands duplicate control, v5 | 25 | 30,769 |
| AudioMNIST cost development study, v6 | 56 | 22,500 |
| Total checks across these studies | 182 | 83,269 |

The raw-check total includes repeated checks of AudioMNIST recordings; the copied input tree contains 60,769 distinct source recordings for these studies. Prepared data and raw recordings were physically copied to a separate directory on the same Windows machine. All 2,060 original snapshotted files were checked before and after each verification.

## What was verified

The wrapper runs each frozen semantic verifier using copied source modules. It translates absolute-path lookups into the copied tree and uses the same strict root-boundary mapping already tested for v7. The v5 checks validate source/raw/prepared/model hashes, split and content separation, locked job identities, selected budgets, prediction cohorts and recomputed metrics. The v6 checks additionally validate replay/selection declarations, canonical-versus-fresh embedding audits, numerical-sensitivity consistency and recorded cost-component sums.

V5's attempted report writes are suppressed. Generated report content must match the frozen evidence after translating generated path keys back to the original root. Changed report content or an unexpected write fails. V6 calls only its read-only verifier function.

A missing prepared-array test failed at the copied path even while the original array remained accessible. The copied array was restored and its hash checked afterwards, confirming that this deeper prepared-artifact path does not fall back to the original data tree.

These checks do not execute all earlier saved classifiers to reproduce predictions; the separate v7 verifier supplies that stronger check for its 60 classifiers. No training, retiming or new test-set tuning occurred. The combined regression and supplement suite passes **112 tests**, with one optional network test skipped and one expected synthetic short-input FFT warning.

## Use and limits

Commands, required files and comparison details are in `handoff/portable_earlier_studies/README.md`. Outcomes and per-stage logs are preserved beside `validation.json`. This supplement has its own `artifacts.json` inventory and adjacent checksum; the original handoff and previous addendum inventories remain unchanged.

The original v5/v6 semantic verification paths are now tested after Windows directory relocation. Training/checkpoint-resume portability, another operating system or machine, full model-training replication and a transferable accuracy-preserving cost rule remain unestablished. Scientific conclusions are unchanged: small-budget representative selection helps in the tested settings, while the independent fixed random retention fraction misses its one-point accuracy target.
