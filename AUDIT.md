# AURA project audit — 18 October 2026

## Overall assessment

AURA has a functioning research prototype for audio preprocessing, acoustic feature extraction, heuristic quality scoring, frozen speech embeddings, and class-balanced subset selection. Real dataset artifacts and reproducible selection outputs exist. It does not yet demonstrate that the proposed combined selector improves downstream accuracy, data efficiency, robustness, or total compute cost.

The direction is reasonable for a data-selection study. The immediate priority is to correct measurement and embedding problems, then evaluate the hypothesis. Building retrieval infrastructure first would leave the central research question unanswered.

## Scope and verification

Reviewed the project-owned Python modules, scripts, configurations, documentation, tests, Git state, generated metadata/embedding artifacts, and all 16 saved selection outputs. Inspected the installed production feature extractor and ran an actual pretrained-model batching diagnostic. Dependency and Git internals were not treated as project source; raw audio was inspected through inventory, metadata references, and selected diagnostic samples, rather than listening to every recording.

Verification performed:

- Existing suite: **47 passed, 1 skipped in 91.51 seconds**. The skipped test downloads a real dataset. Encoder unit/integration tests use a small randomly initialized local model, not the production pretrained checkpoint.
- All 16 saved selection ID lists reproduced exactly with current code and saved embeddings.
- Saved metadata IDs match embedding index order; embedding values are finite and unit-normalized.
- No sample-ID or speaker-ID overlap between saved train, validation, and test splits. This is not a raw-content duplicate audit.
- Referenced audio paths exist according to the inspection script.
- No implementation files or existing experiment outputs were changed for this audit.

## Problem statement and current approach

Given an audio training pool and a fixed budget, choose a smaller subset that preserves useful task coverage while avoiding redundant or damaged samples. The intended outcome is comparable downstream performance using fewer training examples, or better performance at the same budget.

The current implementation focuses on ten isolated spoken-command classes from Speech Commands v0.02. It is not yet a general audio-understanding, ASR, or audio-RAG system.

Pipeline:

1. Load official dataset splits and restrict to the configured ten classes.
2. Convert to mono, resample, peak-normalize, estimate activity with a relative energy threshold, and trim.
3. Extract time-domain, spectral, and MFCC features.
4. Compute a four-component acoustic quality proxy using train-fitted normalization in the standard all-splits run.
5. Extract frozen Wav2Vec2 embeddings and persist metadata, dense matrices, indexes, and caches.
6. Allocate equal per-class budgets and select by random sampling, quality ranking, farthest-point diversity, or a weighted combination of quality and diversity.

The combined selector dynamically updates nearest-selected-point distance. This is a useful implementation, but its score is a heuristic; the code does not establish a global optimization guarantee for the combined objective.

## Verified progress

| Component | Evidence | Status |
|---|---|---|
| Dataset ingestion | Official torchaudio splits; real files present | Implemented |
| Preprocessing and DSP | Modules, tests, populated metadata | Implemented; quality issues below |
| Quality scoring | Scores in saved metadata | Implemented; unvalidated proxy |
| Frozen neural embeddings | Real 768-dimensional matrices | Implemented; batching defect below |
| Persistence and caching | Parquet, NumPy, diskcache | Implemented; provenance gaps |
| Four selection methods | Code plus 16 reproduced outputs | Implemented; no dedicated selection tests |
| Downstream classifier evaluation | No training/evaluation implementation or accuracy artifacts found | Not implemented |
| Iterative active learning | No prediction/uncertainty/acquisition/retraining loop | Not implemented |
| Retrieval/vector database/API/UI | Roadmap only | Not implemented |

| Split | Saved samples | Available samples in configured ten-class dataset | Embedding shape |
|---|---:|---:|---|
| Train | 2,000 | 30,769 | 2,000 × 768 |
| Validation | 500 | 3,703 | 500 × 768 |
| Test | 4,074 | 4,074 | 4,074 × 768 |

The current training pool uses approximately 6.5% of the available ten-class training split. Each training class has 200 samples; validation has 50 per class. Saved selections use budgets 100, 200, 400, and 800, seed 0, and combined alpha 0.5. Every saved run has its requested budget, unique training IDs, and equal class counts.

Completion estimates are judgment calls, not measured task percentages:

- Phase 1: substantially implemented, approximately **80–90% of a dependable prototype**, pending correctness repairs.
- Phase 2: approximately **75–85% of a dependable prototype**, pending tests, stronger contracts, and calibration.
- Phase 3 and Phase 4: no implementation found.
- Whole documented roadmap: approximately **35–45%** complete, allowing for substantial evaluation and retrieval work. Counting checked-off phases alone gives 2/4 but understates the remaining effort.
- The central claim that AURA improves learning efficiency remains unverified.

Git history contains two commits covering Phase 1 and a label-parsing fix. PROJECT.md, aura/selection/, the selection runner, and results/ were untracked at audit start. Local Phase 2 progress is therefore not captured in the committed repository snapshot.

## Priority findings

### High: production embeddings depend materially on batch companions

Location: `aura/encoders/wav2vec2.py:95–128`.

The actual cached `facebook/wav2vec2-base` feature extractor has `return_attention_mask=False`. The encoder therefore takes its unmasked mean-pooling branch for padded batches. The test fixture explicitly enables masks, so current tests do not reproduce production behavior.

Diagnostic using actual processed training clips: encoding the same 3,840-sample clip alone versus alongside a 16,000-sample clip produced cosine similarity **0.449056** and L2 difference **1.049708**. Both outputs were unit vectors. Unit-norm and finite-value checks do not establish representation correctness.

Padding can also affect internal model behavior, so correcting final pooling alone must not be assumed to guarantee invariance. Derive valid output lengths from original inputs independently of whether a model attention mask is appropriate; follow the checkpoint's model-mask conventions; compare against individual inference and test variable-length batching. Grouping similar lengths may reduce effects, but must be measured.

Embedding cache keys omit batch composition and batch size. Currently a vector can thus depend on how a batch was formed while being cached as a per-sample invariant. After repair, version/invalidate caches, regenerate embeddings, and rerun selection.

### High: speech-ratio component is almost predetermined

Location: `aura/ingest/audio.py:76–105,159–169`.

Speech activity is defined as frames above each clip's own 50th energy percentile. For clips without ties, this assigns approximately half the frames to speech by construction, regardless of actual speech occupancy.

Observed training speech ratio: mean **0.499778**, standard deviation **0.001228**, median **0.5**, maximum **0.5**. A quarter of the quality formula therefore contributes almost no discrimination. Median-based activity also makes the SNR proxy an upper/lower-energy contrast rather than independently measured speech-to-noise ratio.

Use a validated speech detector or an absolute/noise-floor-relative detector and evaluate on controlled silence, speech, and noise. Measure quality before trimming when assessing original recording defects.

### High: quality components can reward misleading properties

Locations: `aura/scoring/features.py`, `aura/ingest/audio.py`.

Clipping is measured after peak normalization. This loses original full-scale context: a clean quiet signal can acquire near-unit samples after normalization. Measure recording clipping on the original waveform, and use plateau/run-length evidence if appropriate.

RMS stability rewards constant energy, which can describe noise or silence rather than useful speech. Direct diagnostic: a one-second all-zero waveform yields RMS stability approximately **1.0**, because the RMS epsilon becomes a positive floor. A sub-frame all-zero clip is also marked speech by the short-clip VAD branch. These results contradict the intended acoustic-health interpretation.

Quality needs corruption tests for added noise, hard clipping, silence, gain changes, and truncation, followed by task-based validation. Do not interpret a higher current Q score as demonstrated human-perceived or downstream quality.

### High: evaluation split normalization is unsafe on partial reruns

Location: `scripts/prepare_data.py:249–279`.

The standard train-first run reuses training statistics correctly. A validation/test-only run explicitly fits statistics on the evaluation split instead. Training normalization statistics are not persisted for reliable later reuse. This does not establish that existing saved artifacts leaked, but the advertised unconditional guarantee is false.

Persist train-fitted statistics with their pool/config fingerprint and require them for evaluation-only runs, or fail clearly. Reject non-training rows at the selector API boundary: a diagnostic confirmed `RandomSelector.select()` accepts rows marked `test`. The demo runner filters correctly; the shared API does not enforce the restriction.

### Medium: combined selection currently resembles quality-only selection

Overlap between combined and quality-only selected IDs:

| Budget | Shared IDs | Share of combined subset |
|---|---:|---:|
| 100 | 87 | 87.0% |
| 200 | 187 | 93.5% |
| 400 | 378 | 94.5% |
| 800 | 771 | 96.4% |

At budget 400, mean Q scores are random **0.48188**, diversity **0.48455**, quality **0.53668**, combined **0.53637**. This shows the intended score-ranking behavior, not a learning improvement.

At that budget, quality selects 305 distinct speakers, combined 305, random 341, and diversity 343. This is a useful diagnostic of concentration; it is not proof of fairness or accuracy differences.

Quality is stretched to a full per-class min-max range, while diversity is divided by its theoretical maximum of two. Actual embedding distances may occupy a much narrower range. Equal numeric weights therefore need not create equal practical influence. Inspect score distributions, sweep alpha on validation, and consider quality constraints plus coverage rather than an uncalibrated sum.

### Medium: incomplete reproducibility and cache contract

- `feature_cache.set(feature_key, (processed_audio, features))` stores the waveform inside `ProcessedAudio`. Claims that the feature cache never contains audio arrays are inaccurate; it contains processed audio, although not the original encoded file bytes.
- Cache keys do not capture implementation revisions, DSP constants, dependency versions, or an immutable model revision. Correcting code may still reuse stale values unless explicitly invalidated.
- Output paths are reused across configurations; artifacts lack a comprehensive run manifest tying metadata, embeddings, quality stats, code revision, and selection together.
- Dependency ranges permit substantial environment changes; there is no lockfile or CI configuration in the reviewed project tree.

### Other gaps

- No dedicated selection tests cover capacity redistribution, budgets, determinism, duplicate IDs, invalid/nonfinite vectors, normalization assumptions, alpha endpoints, or split rejection.
- The config permits non-normalized embeddings although cosine-distance code assumes unit vectors.
- Dataset sample-limit allocation can underfill non-multiples of the number of classes: e.g. a requested 25 with ten sufficiently populated classes returns 20. Zero/negative and invalid configuration values need validation.
- Speech trimming calculates the final boundary from the last frame's start plus padding, omitting the frame duration from the stated padding contract.
- PROJECT.md points to nonexistent `phase2/` modules and runner. Correct executable: `python scripts/run_selection_demo.py ...`. Its network-test command also omits the required `AURA_RUN_NETWORK_TESTS=1` environment flag.
- PROJECT.md displays an MIT license badge/link but no LICENSE file exists in the project root.

## Is the approach appropriate?

Yes, as a first controlled experiment in **offline, label-aware speech subset selection**. Modular components, official splits, an explicit random baseline, frozen representations, and reusable artifacts are good foundations.

Two scope distinctions matter:

1. Class-balanced selection uses ground-truth labels throughout the candidate pool. This supports reducing training examples when labels already exist. It does not establish a deployable annotation-saving method for an unlabeled pool. Such a method needs label-free selection or explicitly evaluated pseudo-label balancing.
2. Fixed subset selection is not yet a complete iterative active-learning system. There is no learner-feedback acquisition loop.

Wav2Vec2 is a reasonable speech baseline, but mean-pooled embeddings should not automatically be called semantic representations. They may encode speaker, channel, duration, and other acoustic properties. Retrieval of similar recordings, text-to-audio semantic retrieval, and retrieval-augmented generation need different objectives and evaluation.

## Novelty assessment

The implemented contribution is currently engineering integration and an interpretable combined selection heuristic. Farthest-point/core-set selection and self-supervised speech-based data selection already have published precedent:

- Sener and Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach*, ICLR 2018: https://arxiv.org/abs/1708.00489
- Lu et al., *Unsupervised Data Selection via Discrete Speech Representation for ASR*, 2022: https://arxiv.org/abs/2204.01981

These precedents do not prove that the exact AURA implementation has appeared before, but they rule out claiming the broad ingredients as new. This was a targeted literature check, not an exhaustive novelty search.

A stronger contribution could be a validated task-aware quality/coverage objective, selection under measured annotation-time budgets, explicit speaker/domain coverage, or an adaptive acquisition policy. Any novelty claim needs comparisons and ablations showing why the new component matters.

## Recommended next work

1. Correct encoder batching and quality semantics; persist training stats; enforce train-only selection; add focused regression tests. Version and regenerate affected caches/artifacts.
2. Train a simple linear probe on frozen embeddings. Compare random, quality, diversity, combined, and the complete candidate pool at identical budgets. A simple head makes selection effects easier to interpret.
3. Run at least five seeds, use validation for alpha/head tuning, and reserve test for final comparisons. Report accuracy, macro-F1, per-class recall, mean/std or confidence intervals, and coverage.
4. Measure total cost: preprocessing, full-pool embedding extraction, selection, and training. Savings in classifier training can be outweighed by the up-front encoder pass.
5. Add ablations for each quality component, preprocessing, alpha, and a cheaper MFCC-based representation. Compare quality-filter-then-diversify and representative clustering/facility-location baselines.
6. Expand from the current 2,000 candidates to the complete 30,769-sample ten-class training pool. Add controlled corruption and a second dataset/task before claiming broad audio benefits.
7. Decide the primary objective: labeled training-data reduction, unlabeled annotation acquisition, or retrieval. Add retrieval/API/UI only when that objective and its success metrics justify it.

A defensible success target would be: achieve within a predeclared tolerance of full-pool accuracy using fewer examples, consistently beat class-balanced random at matched budgets, and show where total cost breaks even. These are proposed acceptance criteria, not achieved results.

## Benefits and limitations

Already present: reusable feature/embedding artifacts, repeatable saved selections, transparent scoring components, modular architecture, and a small automated test suite.

Potential after validation: lower repeated training cost, more effective data review, less redundant subsets, improved robustness, and annotation savings if label-free acquisition is implemented.

Not demonstrated: accuracy gains, actual compute savings, perceptual-quality improvement, annotation savings, production scalability, semantic retrieval, or RAG capability.
