# AURA research handoff

Status: 2026-10-19. The baseline experiment program is complete. The implementation is an offline, label-aware speech subset-selection research prototype with frozen wav2vec2 features and linear probes. Retrieval, RAG and annotation acquisition are separate extensions.

## Problem and approach

Can a smaller training set preserve useful speech-classification accuracy, and can selecting it save enough computation to justify its acquisition cost? These are two different objectives: accuracy at a fixed example budget, and accuracy versus total training computation when the budget may vary.

AURA repairs ingestion, source diagnostics, normalization, padding-sensitive inference, cache provenance and evaluation separation, then compares random selection, quality heuristics, coverage methods and actual examples nearest class-wise KMeans centers. Speaker-disjoint evaluation and exact-byte content exclusions reduce leakage. Validation/development choices are locked before final scores are opened. The later cost studies count raw preprocessing and neural encoding instead of reporting head fitting alone.

## Evidence and limits

| Study | Main finding | Practical implication |
|---|---|---|
| Speech Commands scaling, v4 | At budget 100, representatives reach 59.42% versus random 51.69%; the advantage vanishes at 1,600 | Coverage helps under a strict small training-example budget |
| Duplicate control, v5 | Representative gains over content-uniform random persist at budgets 100–400 on the observed v4 cohort | Exact duplicate avoidance alone does not explain those gains; this control is post-hoc |
| Independent AudioMNIST, v5 | Representatives at budget 100 reach 85.61% versus deduplicated random 76.92%; best tested subset 94.73%, full pool 97.01% | Low-budget benefit transfers, but subsets leave a meaningful full-pool gap |
| AudioMNIST cost development, v6 | Random 6,400/17,500 reaches 96.71% versus full 97.52%, with 37.55 versus 104.99 measured component seconds | Larger random subsets are strong compute baselines; tested curation recipes are dominated on this cross-budget grid |
| Independent FSDD cost confirmation, v7 | Random 920/2,500 reaches 66.71% versus full 69.03%; median paired warm training saving 61.78%, faster in 18/18 pairs | Speed saving transfers; the fixed retention rule fails the one-point accuracy goal |

These are different corpora, cohorts and protocols; absolute scores are not a controlled comparison between stages. V6 sums measured components and uses canonical primary accuracy matrices; v7 times actual raw-to-head trials and trains on the fresh timed vectors. The v7 loss is 2.327 points, exploratory speaker-bootstrap interval [1.020, 3.687]. Six evaluation speakers and overlapping training folds limit statistical independence; this is not a powered noninferiority result.

The quality proxy fails monotonic corruption checks, and fixed gating does not consistently improve representative selection. Its components are diagnostics, not a validated perceptual or downstream-utility metric. Numerical changes from CUDA batch composition alter exemplar IDs in v6. V7's same-layout repeats are identical, but that does not establish cross-layout or cross-device reproducibility.

## Contribution, benefits and alternatives

The contribution established here is a corrected, auditable evaluation pipeline and empirical evidence about when baseline selection helps and when its costs erase the benefit. KMeans exemplars, quality filtering and coverage sampling do not establish algorithmic novelty. Class-balanced methods already use candidate labels, so no annotation savings have been demonstrated.

For a hard example budget, unfiltered representatives have the most consistent tested accuracy evidence. For flexible budgets and this frozen learner, compare against larger random subsets before paying for full-pool curation. If accuracy-preserving compute savings remain the objective, develop a budget or stopping rule on training/development data and confirm it on genuinely new speakers. If labels are the scarce resource, replace this protocol with unlabeled, sequential acquisition and count total labels and computation. For expensive fine-tuning, benchmark the downstream learner directly; linear-probe results do not establish fine-tuning savings.

The immediate benefits are reliable experiment reuse, traceable selections and models, stronger inexpensive baselines, and evidence that prevents unsupported savings or novelty claims. The measured 61.78% saving applies to warm training in the recorded FSDD workload and hardware session; it does not measure energy, monetary cost, cold startup or production service latency.

## Completion boundary

The baseline study and local research handoff are complete. Exact environment capture, a source/result checksum snapshot, a read-only verification command and a reproduction runbook are included. Clean-environment installation, relocated legacy reporter execution, cross-device replication, a new selection mechanism and deployment have not been validated. They are explicit limitations or separately scoped next work, rather than positive results implied by this handoff.

Local regression verification: 102 tests passed, one optional network test skipped. Start with `REPRODUCIBILITY.md` and `COMPLETION.md`. Detailed stage narratives are in `results/scaling_v4/INTERPRETATION.md`, `results/duplicate_control_v5/REPORT.md`, `results/audiomnist_v5/INTERPRETATION.md`, `results/acquisition_cost_v6/INTERPRETATION.md` and `results/cost_confirmation_v7/INTERPRETATION.md`. `AUDIT.md` is historical pre-repair evidence.
