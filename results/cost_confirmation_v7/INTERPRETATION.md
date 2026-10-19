# Independent cost confirmation: interpretation

Completed 2026-10-19. The experiment is complete, but the fixed retention rule fails its one-percentage-point accuracy target on this corpus. This is a useful negative transfer result, not a reason to adjust the rule using the same test scores.

## Frozen question and result

Can the AudioMNIST development choice—class-balanced random retention of 64/175 of the training pool—reduce actual raw-to-classifier training time while preserving accuracy within one point on previously unused FSDD speakers?

Each of six leave-one-speaker-out folds has 2,500 training recordings and 500 evaluation recordings. The transferred budget rounds to 920 training recordings, 92 per class. C=1, the frozen wav2vec2 checkpoint, source preprocessing and retention fraction are unchanged. Five subset seeds quantify selection variability. All 60 fitted heads were locked before evaluation predictions. Folds share training speakers; this is standard leave-one-speaker-out evaluation, not six independent training populations.

| Quantity | Observed result |
|---|---:|
| Full-pool accuracy | 69.03% |
| Random subset accuracy, five-seed mean | 66.71% |
| Full-minus-subset loss | 2.327 percentage points |
| Exploratory speaker-bootstrap 95% interval | [1.020, 3.687] points |
| Median full warm training time | 19.87 seconds |
| Median subset warm training time | 7.83 seconds |
| Median paired time saving | 61.78% |
| Pairs with subset faster | 18/18 |

The timing saving is the median of paired savings, not a ratio calculated from the two separate time medians. Three repeats per speaker counterbalance full/subset execution order. Timers include selection, raw decoding and preprocessing, uncached neural encoding, matrix assembly, scaling and head fitting. Resident-model startup and synthetic warm-up (5.47 seconds), inventory checks, serialization and evaluation are excluded. Results describe this warm hardware session, not energy, monetary cost or cold-machine latency.

## What this changes

The development result suggested that a larger random subset could save compute with a small accuracy loss. Independent confirmation supports the compute saving here but rejects the observed one-point accuracy goal for the transferred fraction. Neither study establishes a universally safe retention percentage or matched-accuracy cost superiority.

Speaker losses range from 0.16 to 5.40 points; four of six exceed one point. Even the full-pool baseline has substantial speaker-transfer error, with speaker accuracies from 56.2% to 75.6%. These observations motivate a development-only investigation of representation and speaker robustness. They do not identify the cause: the small speaker population, recording conditions, 8 kHz source bandwidth and the frozen head could contribute. No causal attribution has been tested.

The interval conditions on fitted models, resamples only six speaker blocks and omits training-population uncertainty. Overlapping fold training sets limit independence. It is exploratory, not a powered noninferiority test. The observed mean itself exceeds the target, so stronger statistical machinery is unnecessary to describe this run as missing its goal.

All 24 same-layout repeat comparisons had byte-identical training embeddings and identical evaluation predictions. This supports repeatability under the recorded session and batching layout. It does not resolve v6's sensitivity to changing batch composition or guarantee identical exemplar selections across devices.

## Supported project position

The research prototype now has evidence for low-budget representative-selection accuracy benefits on Speech Commands and AudioMNIST, plus measured limits: curation overhead can erase compute benefits, quality gating is inconsistent, and an attractive fixed random retention fraction does not preserve the chosen accuracy margin on FSDD. These are baseline and evaluation contributions; algorithmic novelty and annotation savings remain unestablished.

The next useful deliverable is a reproducible research handoff with these claims and failures. If further optimization is required, choose the downstream learner and acceptable accuracy loss first, then develop an adaptive stopping or budget rule using training/development data only. Compare it against random and full-pool baselines with actual total training time, and reserve genuinely new speakers for final confirmation. Selecting a larger fraction from the now-observed FSDD scores would be development work and would need a separate confirmation cohort.

Quality calibration, unlabeled acquisition and retrieval/API deployment are distinct extensions, not missing implementation necessary to report the completed baseline study.

## Verification and artifacts

The reporter checked 3,000 raw hashes, split/content separation, all selected class budgets, 60 checkpoint manifests, all saved-model predictions and metrics, 36 primary models and 18 timing pairs. The project suite passes 95 tests with one optional network test skipped. The report figure was visually inspected. See `REPORT.md`, `verification.json`, `accuracy_summary.json`, `timing_summary.json`, `speaker_accuracy.csv` and `repeat_stability.csv`.

Dataset attribution: Free Spoken Digit Dataset, Jakobovski and contributors, revision `26eb9aaf76e81b692f806f9140c2d2777410d7a1`; its README declares CC BY-SA 4.0. See `DATASET_NOTICE.md`. This custom speaker-disjoint evaluation differs from FSDD's official index split.
