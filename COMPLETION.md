# AURA completion criteria

The current deliverable is a reproducible speech-data subset-selection research prototype. A deployable retrieval product or an annotation-acquisition system would require additional scope and evaluation. An overall percentage would mix those different endpoints; use the milestone evidence below instead.

As of 2026-10-19, the baseline experiment program and local handoff are complete. See `RESEARCH_FINDINGS.md` and `REPRODUCIBILITY.md`. The handoff includes exact installed package versions and source/result checksums; clean installation and independent hardware replication have not been validated.

The final local regression suite has 102 passing tests and one optional network test skipped. Stage-specific test counts below describe checks at those earlier milestones.

| Milestone | Status | Evidence / completion condition |
|---|---|---|
| Audit and repair data preparation | Complete | Source-audio diagnostics, padding-independent embeddings, cache provenance, train-only normalization, split and selector guards; regression tests |
| Implement and compare subset selectors | Complete | Random, quality, diversity, combined, representatives, quality-filtered representatives; saved selected IDs and locked settings |
| Small-pool and fresh-speaker confirmation | Complete | `results/benchmark_v2/`, `results/representative_v3/` |
| Full available ten-class training-corpus scaling | Complete | 28,755 candidates; 2,014 reserved-speaker holdout recordings; `results/scaling_v4/` |
| Distinguish representation benefits from exact duplicate avoidance | Complete | `results/duplicate_control_v5/REPORT.md`; advantage over deduplicated random supported at budgets 100–400 on the observed v4 cohort |
| Independent dataset transfer | Complete | AudioMNIST: 22,500 candidates, 7,500 recordings from 15 held-out speakers, 101 locked models; +8.69 points over deduplicated random at budget 100; all artifacts verified |
| Demonstrate useful total cost savings | Measured speed saving; accuracy target not confirmed | Independent FSDD: 61.78% median paired warm training-time saving, but 66.71% versus full 69.03% accuracy; 2.327-point loss exceeds the one-point goal |
| Development cost comparison | Complete | 56 same-C models; MFCC-before-neural selection, larger random controls, uncached computation and numerical audit; tested curation recipes lose the cross-budget accuracy/cost comparison; `results/acquisition_cost_v6/INTERPRETATION.md` |
| Independent fixed-rule cost confirmation | Complete; negative accuracy result | FSDD six-speaker LOSO, fixed 920/2,500 random budget, 60 locked models, five subset seeds and 18 timed pairs; all artifacts verified; `results/cost_confirmation_v7/INTERPRETATION.md` |
| Raw-to-subset numerical stability | Further work needed | Fresh CUDA batch shapes change embeddings and selected exemplar IDs; canonical results remain frozen, v6 records precision flags and seed-zero sensitivity; strict reproduction needs an expanded numerical contract |
| Establish a dependable quality signal | Pending / optional for representative baseline | Current proxy fails monotonic corruption checks and fixed gating does not consistently improve results; calibrate only on development data and confirm on new data, or omit the gate from the main method |
| Support annotation savings | Pending if claimed | Present methods already require candidate labels; add unlabeled acquisition, sequential learner feedback, and label-budget evaluation before making this claim |
| Establish algorithmic novelty | Not established | KMeans exemplars, quality filters and farthest-point selection are existing baseline ideas. Current contribution is the measured protocol, evidence, and limitations; a novel method needs a distinct mechanism and ablations |
| Final research handoff | Complete locally; clean replication unvalidated | Exact environment capture, relative source/result checksum snapshot, read-only archive/legacy verification, result narrative, failure cases, attribution and reproduction runbook; `REPRODUCIBILITY.md` |
| Retrieval/API/product | Separate future scope | Define users, retrieval objectives, corpus and relevance judgments before selecting vector-store/API architecture; add operational tests only after that objective is defined |

## Recommended remaining work after independent validation

1. Validate the completed handoff in a separate clean environment and hardware session. Exact dependencies, numerical settings, result narrative, checksum snapshot and reproduction instructions are now recorded and locally checked. The current baseline experiment program is complete; missing a performance target does not make its experiment incomplete.
2. If preserving accuracy while reducing compute remains mandatory, develop a different budget/stopping rule using training/development data, then confirm it on new speakers. The fixed fraction failed the independent one-point goal. Keep v3, v4, AudioMNIST and FSDD evaluation scores out of future tuning claimed as fresh confirmation. Choose the downstream learner and success metric before further selector work.
3. Decide whether quality filtering is essential. The unfiltered representative baseline currently has more consistent evidence. A learned gate requires its own development data, calibration and ablations.
4. If the objective is annotation efficiency, change the acquisition protocol so it never sees candidate labels before selection. Evaluate iterative acquisition against random under the same total label and compute budget.
5. Add retrieval/API or product components only after defining the user-facing objective and its success metric. They are separate scope from this research prototype.

## Claims currently supported

On Speech Commands and an independent AudioMNIST digit task with frozen wav2vec2 embeddings, class-balanced cluster exemplars improve low-budget classifier accuracy over the tested random controls. Exact-byte duplicate avoidance alone does not explain the advantage at budgets 100–400. AudioMNIST's full-pool reference reaches 97.01%; the best tested subset reaches 94.73%, leaving a 2.29-point gap. This does not establish a universally superior selector, a new clustering algorithm, label savings, demographic fairness, end-to-end fine-tuning performance, or total compute savings. See `results/audiomnist_v5/INTERPRETATION.md` and the earlier reports for effect sizes and uncertainty.

The later training-pool development cost study uses a different cohort and C=1, so its absolute scores are not directly comparable with v5. It finds a promising larger-random cost baseline, while the tested MFCC and wav2vec2 curation recipes are dominated when subset size may vary. Strict matched-accuracy cost superiority still needs confirmation. Numerical sensitivity can alter exemplar IDs even when aggregate accuracy changes less; source/seed alone is not a complete raw-to-subset reproducibility guarantee.

The completed independent FSDD cost confirmation transfers the fixed 64/175 retention rule without tuning. It reduces median paired warm training time by 61.78% and is faster in all 18 pairs, but loses 2.327 accuracy points (exploratory interval [1.020, 3.687]) versus the full pool. Thus the one-point accuracy goal is not achieved. Six speakers and overlapping fold training sets limit uncertainty claims. Same-layout repeat embeddings and predictions are identical in all 24 comparisons; cross-layout/device stability remains unproven. All 60 models and 3,000 raw hashes are verified; 95 tests pass and one optional network test is skipped. See `results/cost_confirmation_v7/INTERPRETATION.md`.
