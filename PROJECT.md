# AURA: Audio Understanding, Retrieval & Active Data Selection

AURA is a research prototype for **offline, label-aware speech subset selection**. It extracts acoustic diagnostics and frozen speech embeddings, selects class-balanced training subsets, and evaluates them with a linear classifier. Semantic retrieval, RAG, APIs, and an interactive UI remain future work.

The research question is whether quality and representation coverage improve downstream accuracy at a fixed training-example budget. Annotation savings on an unlabeled pool are outside the current experiment: all selectors use known class labels.

The baseline experiment program and local research handoff are complete. Start with `RESEARCH_FINDINGS.md` for the supported conclusions and `REPRODUCIBILITY.md` for archive checks, environment capture and replication commands. `handoff/research_snapshot.json` identifies the exact source and saved results; Git HEAD alone does not capture the uncommitted research implementation. A clean installation and second-device replication remain unvalidated.

## Current implementation

- Speech Commands v0.02, ten configured command classes, official train/validation/test splits.
- Mono conversion and resampling; source diagnostics before gain normalization or trimming.
- Energy activity using a low-percentile noise floor, a 6 dB margin, and an absolute RMS floor. This is a heuristic, not a validated speech detector.
- DSP features including RMS, spectral features, 13 MFCC means/stds, original-source clipping fraction, active/inactive energy contrast, and RMS stability.
- An interpretable quality proxy combining normalized energy contrast, clipping avoidance, activity ratio, and normalized stability. Silence/no detected activity is gated to zero. This is not a perceptual quality metric.
- Frozen `facebook/wav2vec2-base`, mean pooling and L2 normalization. Only equal-length clips share an inference batch, avoiding padding-induced changes in convolutional group normalization.
- Random, quality, diversity (farthest-point), and combined quality/diversity selection. Budgets are distributed evenly per class, capped at capacity, and redistributed where needed. A combined score is a heuristic, not a guarantee of an optimal subset.
- Validation-tuned logistic-regression probes, five selection seeds, per-class recall/confusion matrices, and a configured-full-pool baseline.

## Directory map

```text
aura/
  config.py             Configuration and validation
  provenance.py         Pipeline/source/runtime fingerprints
  ingest/               Dataset records, audio processing, persistence
  scoring/              DSP features and prototype quality score
  encoders/             Frozen Wav2Vec2 encoder
  cache/                Content-addressed diskcache wrapper
  selection/            Shared contracts and four selection strategies
configs/
  experiment.yaml       Original output locations
  experiment.smoke.yaml Small separate smoke outputs
  experiment.v2.yaml    Corrected pipeline with separate data/v2 outputs
scripts/
  prepare_data.py       Preparation, saved train statistics, split manifests
  inspect_dataset.py    Artifact diagnostics
  run_selection_demo.py Selection IDs and provenance
  evaluate_selection.py Validation tuning and final test benchmark
  study_representative_selection.py  New selectors + fresh-speaker confirmation
  study_quality_signals.py           Training-only quality validation
  plot_representative_study.py       Export benchmark curves
experiments/
  representative.py     Cluster exemplars and quality-gated cluster exemplars
tests/                  Unit, synthetic integration, and audit regressions
results/
  selection/            Original audited selection results
  benchmark_v2/         Corrected benchmark outputs when generated
AUDIT.md                Historical audit before repairs
```

## Setup and commands

Python 3.10+ is required. Install dependencies into a virtual environment:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

Prepare the corrected experiment, then inspect artifacts and run selection/evaluation:

```powershell
.venv/Scripts/python.exe scripts/prepare_data.py --config configs/experiment.v2.yaml
.venv/Scripts/python.exe scripts/inspect_dataset.py --config configs/experiment.v2.yaml
.venv/Scripts/python.exe scripts/run_selection_demo.py --config configs/experiment.v2.yaml --output-dir results/selection_v2
.venv/Scripts/python.exe scripts/evaluate_selection.py --config configs/experiment.v2.yaml --output-dir results/benchmark_v2
```

The default v2 experiment uses 2,000 training candidates, 500 validation examples, and all 4,074 ten-class test examples. The available ten-class training split contains 30,769 examples. A full-pool baseline in this experiment therefore means 2,000 candidates, not the complete available training split.

For the optional dataset-download test:

```powershell
$env:AURA_RUN_NETWORK_TESTS='1'
.venv/Scripts/python.exe -m pytest -q -k test_real_dataset_loader
```

## Quality and artifact contracts

The quality formula for active clips is:

```text
Q = 0.25 * normalized_energy_contrast
  + 0.25 * (1 - source_clipping_ratio)
  + 0.25 * activity_ratio
  + 0.25 * normalized_rms_stability
```

Min/max statistics are fitted on training features only and saved in `quality_norm_stats.json`. Evaluation-only reruns require matching saved training statistics and training metadata; missing or stale statistics fail explicitly.

Each split has metadata Parquet, a dense float32 embedding matrix, a row-index Parquet, and a manifest. Manifests record code fingerprints, package versions, resolved model revision, configuration, elapsed time, and artifact hashes. Evaluation verifies these before accepting inputs. Source or environment changes invalidate preparation cache keys; changing the quality formula requires re-preparation as well.

Feature caches store feature records only. Waveforms are processed on demand and retained in memory during a split, not stored in the v2 feature cache. The historical v1 cache contained processed waveform arrays; v2 uses separate paths and does not delete those historical files.

Selectors reject non-training rows, duplicate IDs, missing/nonfinite quality scores where required, and nonfinite/non-unit embeddings for diversity methods. The demo reads training artifacts only.

## Benchmark protocol and outputs

The default benchmark compares budgets 100, 200, 400, and 800 with seeds 0 through 4. Logistic-regression regularization C is selected from 0.1, 1, and 10. Combined alpha is selected from 0, 0.25, 0.5, 0.75, and 1. For each strategy/budget, the settings with best mean validation accuracy across seeds are locked before opening test data. Ties use validation macro-F1, then lower alpha/C.

Scaling is fitted only on the selected training examples. The deterministic quality strategy repeats the same subset across seeds; its zero seed variance is not evidence of broad robustness. The benchmark measures a frozen-embedding linear probe, not fine-tuning or an end-to-end raw-audio classifier.

Outputs include `validation_sweep.csv`, `validation_choices.json`, selected-ID JSONs, `runs.csv`, `summary.csv`, `per_class.json`, `experiment.json`, and `REPORT.md`. Completed experiment directories cannot be overwritten by the runner; use a new output directory for a new experiment.

Preparation and per-run fit/selection costs are recorded, but lower head-training cost alone does not establish total compute savings: the full-pool embedding pass must also be counted.

## Remaining work

1. Validate the packaged instructions in a separate clean environment and hardware session. Local source/result integrity, package compatibility, regression tests and saved independent-study checks are verified.
2. If accuracy-preserving compute savings are required, develop and independently confirm another budget/stopping rule. The completed FSDD confirmation measures speed savings but misses the one-point accuracy target.
3. Improve quality calibration only if a quality gate remains a goal; existing evidence favors the simpler unfiltered representative baseline.
4. For annotation efficiency or expensive fine-tuning savings, evaluate the actual learner and acquisition protocol under matching total label/compute budgets. Existing known-label linear probes do not establish those benefits.
5. Design retrieval objectives and evaluation before adding a vector database, API, or RAG interface. See `COMPLETION.md` for completion boundaries and separate extensions.

## Independent validation and duplicate control

The next frozen study transfers the existing settings to [AudioMNIST](https://github.com/soerenab/AudioMNIST), revision `630d7dab4c040882834de6fa21baf9a60372accd`, distributed under the repository's MIT license. Cite Becker et al., *AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark*, DOI `10.1016/j.jfranklin.2023.11.038`. The downloaded repository preserves its license and citation.

A declared seed-20261018 split reserves 15 of 60 speakers: 22,500 candidate recordings and 7,500 held-out recordings. No byte-identical recordings cross roles. This is a custom speaker split, not the authors' benchmark split. Digits provide another task and recording environment while remaining within speech classification. The frozen pipeline resamples the 48 kHz recordings to 16 kHz.

For a clean reproduction, obtain the pinned dataset revision before running the study:

```powershell
git init data/raw/AudioMNIST
git -C data/raw/AudioMNIST remote add origin https://github.com/soerenab/AudioMNIST.git
git -C data/raw/AudioMNIST fetch --depth 1 origin 630d7dab4c040882834de6fa21baf9a60372accd
git -C data/raw/AudioMNIST checkout --detach FETCH_HEAD
```

Random, deduplicated random, representatives, and quality-filtered representatives use budgets 100, 200, 400, 800, and 1,600 with five seeds. Settings transfer from Speech Commands v3 without tuning on AudioMNIST. Deduplicated random uses random's classifier C and samples unique raw-byte contents uniformly within each class. Every model is checkpointed before held-out audio is prepared. A full-candidate head provides a reference.

```powershell
.venv/Scripts/python.exe scripts/study_independent_dataset.py --control
.venv/Scripts/python.exe scripts/report_independent_study.py --control
.venv/Scripts/python.exe scripts/study_independent_dataset.py
.venv/Scripts/python.exe scripts/report_independent_study.py
.venv/Scripts/python.exe scripts/verify_independent_study.py
```

The duplicate control on the already observed v4 holdout is complete. Representatives exceed deduplicated random by +8.31, +3.10, and +1.68 percentage points at budgets 100, 200, and 400; paired speaker-bootstrap intervals exclude zero at those budgets. At 800 and 1,600, the differences are -0.64 and -0.60 points with intervals spanning zero. These post-hoc diagnostics narrow the supported budget range and do not count as fresh confirmation. See `results/duplicate_control_v5/REPORT.md`.

The runners save immutable protocols, selected IDs, models, manifests, timing, predictions, summary tables, and paired speaker-bootstrap comparisons under `results/audiomnist_v5/` and `results/duplicate_control_v5/`. Preparation remains resumable in `data/audiomnist_v5/`. `pytest.ini` restricts discovery to project tests, avoiding traversal of downloaded datasets and upstream code.

The completed independent study reaches 85.61% at budget 100 with representatives versus 76.92% with deduplicated random: +8.69 points, exploratory paired speaker-bootstrap 95% interval [+7.39, +9.77]. Positive mean gains occur at all tested budgets; the interval at 1,600 barely excludes zero and should not be treated as strong evidence after multiple comparisons. The full 22,500-recording head reaches 97.01%; the best tested subset reaches 94.73%, still 2.29 points lower. Quality gating does not establish a consistent improvement over unfiltered representatives across both datasets. No annotation or total compute savings have been demonstrated. See `results/audiomnist_v5/INTERPRETATION.md` and `COMPLETION.md`. The test suite has 81 passing tests and one skipped optional network test.

See `AUDIT.md` for historical findings. It describes the pre-repair code and should not be read as the current implementation status.

## Acquisition cost development study

The v6 cost study uses only the previous AudioMNIST candidate training pool, partitioned into 35 candidate speakers (17,500 clips) and 10 development speakers (5,000 clips). It never opens the previously evaluated 15-speaker holdout. This is development evidence, not another independent confirmation set.

The study compares class-balanced random selection, cluster exemplars in the frozen wav2vec2 representation, and cluster exemplars using 26 cheap source-audio MFCC summaries. MFCC scaling fits candidate rows only. Random and MFCC subsets are saved before this run performs neural encoding. All classifiers use C=1; no classifier or descriptor search is performed. Budgets are 100, 400 and 1,600, with extra random controls at 3,200 and 6,400 and a full-pool reference; five seeds give 56 model jobs.

Persistent feature/embedding caches are bypassed for the timed passes. An actual full-pool raw preprocessing/encoding pass supplies the shared expensive baseline component. Random/MFCC recipes have separate selected-only raw encoding replays at seed 0. Candidate MFCC extraction, normalization, selection, encoder setup/warm-up, raw encoding and head fitting are accounted for. Common evaluation preparation, downloads, integrity checks and checkpoint serialization are excluded. Runtime libraries and OS caches are warm; the experiment does not measure a cold machine or energy consumption. Costs are sums of measured components, with one timing replay per configuration; accuracy is averaged over five fixed models.

```powershell
$env:HF_HUB_OFFLINE='1' # after obtaining the cached checkpoint
.venv/Scripts/python.exe scripts/study_acquisition_cost.py
.venv/Scripts/python.exe scripts/report_acquisition_cost.py
```

The reporter verifies input hashes, content/speaker separation, fresh-versus-frozen representations, selections, model hashes, predictions and component sums. It writes `accuracy_cost.csv`, uncertainty comparisons, a plot, `REPORT.md`, and final manifests under `results/acquisition_cost_v6/`. The original interrupted timing pass is retained under `results/acquisition_cost_v6_preflight/` and excluded from analysis because regression tests briefly overlapped with it.

A second pass stopped before scoring because strict coordinate equality with stored embeddings failed. Equal-length B1/B16 controls reproduce numerical differences without padding, consistent with [PyTorch's numerical-accuracy notes](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html). The stopped protocol and runner are preserved under `results/acquisition_cost_v6_strict_check/`. The amended protocol fixes all primary selection/accuracy comparisons to canonical frozen embeddings. Fresh timed outputs receive recorded drift and cosine/ID/unit-norm checks, followed by untimed seed-zero fresh-vector shadow fits/reselection. This makes the distinction between fixed-representation accuracy and measured component costs explicit; it does not assert exact numerical identity. See `NUMERICAL_CONTROL.md`, replay JSONs and `numerical_sensitivity.csv` in the final results directory.

The completed study finds random at 6,400 recordings reaches 96.71% versus full-pool 97.52%, at 37.55 versus 104.99 measured component seconds. The mean loss is 0.812 points, with exploratory speaker-bootstrap interval [0.440, 1.192]; a one-point noninferiority claim is not established. MFCC and wav2vec2 curation are dominated on this tested grid when recording budget may vary. Fixed-budget data-efficiency benefits remain a different objective. Fresh-vector wav2vec2 exemplar Jaccards are 0.64/0.29/0.21 at budgets 100/400/1,600, exposing a numerical selection-stability limitation despite smaller aggregate accuracy changes. See `results/acquisition_cost_v6/INTERPRETATION.md`. Verification covers all 56 jobs and 22,500 source hashes; the current suite has 87 passing tests and one skipped optional network test.

## Independent cost-rule confirmation

The v7 study transfers v6's random retention fraction, 64/175, to previously unused FSDD without tuning. FSDD is pinned to revision `26eb9aaf76e81b692f806f9140c2d2777410d7a1`; Jakobovski and contributors declare CC BY-SA 4.0 in its README. Six speaker-disjoint folds each train on 2,500 recordings and evaluate on 500. The transferred subset has 920 recordings, balanced across ten known digit labels. This custom leave-one-speaker-out protocol differs from the official index-based split.

All 60 heads are locked before evaluation predictions: six full-pool primary heads, 30 subset primary heads (five seeds), and 24 extra fits for three paired timing repeats per speaker. Actual warm raw-to-classifier training is timed, including uncached encoding; AB/BA ordering is counterbalanced. Startup/warm-up, inventory, serialization and evaluation are excluded. All numerical/runtime settings are saved. Same-layout repeated embeddings and predictions are identical in 24 comparisons; this does not resolve cross-batch exemplar sensitivity.

```powershell
$env:HF_HUB_OFFLINE='1' # after obtaining the cached checkpoint
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
.venv/Scripts/python.exe scripts/study_cost_confirmation.py
.venv/Scripts/python.exe scripts/report_cost_confirmation.py
```

Run timed experiments without competing tests or other heavy workloads. The runner resumes intact checkpoints and rejects changed artifacts. The reporter rechecks raw/source hashes, folds, selections, timing pairs, saved-model predictions and metrics without retraining. Preserve the v6 decision artifacts: the v7 protocol pins their hashes.

The independent result misses the chosen accuracy goal: full accuracy is 69.03%, subset mean 66.71%, a 2.327-point loss with exploratory speaker-bootstrap interval [1.020, 3.687]. Median warm training times are 19.87 and 7.83 seconds; median paired saving is 61.78%, with the subset faster in 18/18 pairs. These findings support measured speed savings on this workload, not a transferable one-point accuracy guarantee. Six speakers and overlapping training folds limit statistical conclusions. Do not tune on these observed scores and call it fresh confirmation.

See `results/cost_confirmation_v7/INTERPRETATION.md`, `REPORT.md`, `DATASET_NOTICE.md` and `verification.json`. All 60 models and 3,000 recordings are verified; the current suite passes 95 tests with one optional network test skipped. Remaining work is the final reproducibility handoff or a separately defined optimization extension; see `COMPLETION.md`.

## Representative-selection and quality studies

The follow-up studies keep the corrected v2 preparation code and embeddings frozen. Experimental selectors live in `experiments/` so changing an acquisition algorithm does not invalidate expensive unchanged audio artifacts.

`RepresentativeSelector` fits class-wise KMeans clusters and chooses an actual recording nearest each assigned cluster's centre. `QualityRepresentativeSelector` first retains a configurable top fraction by the existing quality proxy, then applies the same procedure. It retains enough candidates to fill each allocated class budget. These are experimental baselines, not established novel algorithms.

The study compares random, quality, farthest-point diversity, original combined selection, representative selection, and quality-gated representatives. It uses the original four budgets and five seeds, selects classifier C/combined alpha/retained fraction using the existing tuning validation set, and locks every setting before processing a fresh confirmation cohort.

The confirmation cohort consists of all official-validation recordings whose speakers occur in neither the training candidate pool nor the existing 500-example tuning set. It contains 372 recordings. This small, naturally imbalanced cohort is a new check; its scores must not be compared directly with the earlier 4,074-example test scores. The old test set is not opened by this study.

```powershell
.venv/Scripts/python.exe scripts/study_quality_signals.py
.venv/Scripts/python.exe scripts/study_representative_selection.py
.venv/Scripts/python.exe scripts/analyze_confirmation_uncertainty.py
.venv/Scripts/python.exe scripts/plot_representative_study.py
```

Quality diagnostics use 100 class-balanced training recordings with controlled noise, clipping, gain, silence, and prefix truncation. They also measure associations between quality components and losses from five speaker-disjoint classifier folds on the 2,000 training candidates. No quality weights are fitted and no validation/test examples enter those diagnostics.

Results are stored separately in `results/quality_study_v3/` and `results/representative_v3/`. The representative study saves its predeclared protocol, selected IDs, locked settings, complete validation sweep, fresh confirmation cohort/artifacts, per-class metrics, coverage/isolation diagnostics, and provenance. The plotting script exports PNG and SVG curves. Once confirmation results are visible, further algorithm development should not tune against that cohort.

The first follow-up found that quality-filtered representatives beat random in mean confirmation accuracy at all four budgets. The strongest evidence is at budget 100: 56.83% versus 51.18%, a +5.65 percentage-point difference with an exploratory speaker-bootstrap interval of [+2.51, +8.79] points. Larger-budget intervals include zero. This is a 372-recording, 59-speaker confirmation study, not a general superiority claim. See `results/representative_v3/INTERPRETATION.md` and `UNCERTAINTY.md`.

## Frozen full-corpus scaling study

The scaling study inventories all 30,769 official-training recordings in the ten configured classes. It reserves 250 speakers absent from the original 2,000-example candidate pool, giving **28,755 candidate training recordings and 2,014 fresh holdout recordings**. Speakers and raw-byte content are disjoint across these roles. This is an explicitly reserved holdout from the official training corpus, not the official test split or an independent dataset.

Settings transfer from the earlier v3 tuning choices at budgets 100, 200, 400, and 800. Budget 1,600 uses the previously tuned budget-800 settings. The full-pool head uses the earlier full-pool C. No new parameter search is performed. The original combined, quality, diversity, representative, quality-filtered representative, and random selectors retain their implementations.

```powershell
# This script deliberately uses the full corpus, overriding dev sample limits.
.venv/Scripts/python.exe scripts/prepare_scaling_study.py
# May be declared before preparation finishes; no holdout data are processed.
.venv/Scripts/python.exe scripts/benchmark_scaling_study.py --declare-only
# Once candidate preparation finishes, fit all heads, then prepare/evaluate holdout.
.venv/Scripts/python.exe scripts/benchmark_scaling_study.py
.venv/Scripts/python.exe scripts/plot_scaling_study.py
```

Preparation saves validated 500-recording chunks under `data/scaling_v4/chunks/` and resumes them on interruption. Normalization statistics are fitted across the complete candidate pool, excluding the holdout. Completed classifier jobs are also saved with hashes and can be resumed before final evaluation. The benchmark protocol is immutable after declaration; changed settings require a new results directory.

Quality scores inside staging chunks are provisional. Use final metadata from `data/scaling_v4/metadata/`, where scores have been recomputed with complete-candidate normalization statistics.

For parallel execution, stop the sequential benchmark first, then run `scripts/precompute_scaling_jobs.py --workers 4` with two BLAS/OpenMP threads per process. This executes only pending jobs from the same frozen protocol and writes compatible hashed checkpoints. Once it finishes, resume `scripts/benchmark_scaling_study.py` to verify/load every checkpoint and evaluate the holdout. The controller must not run alongside the sequential benchmark. Parallel scheduling changes resource contention, so the observed wall times are not an isolated algorithm-speed benchmark. The actual execution log is saved in `parallel_execution.json`.

Partition/source hashes, the frozen protocol, per-seed selections and models, preparation/selection/fit timings, predictions, paired speaker-bootstrap differences, and final reports are stored under `data/scaling_v4/` and `results/scaling_v4/`. Earlier experiments and their evaluation cohorts remain intact.

After completion, `scripts/analyze_scaling_selections.py` reports selected-content duplicates and the recording-count fraction/gap to full-pool accuracy. Recording-count reduction alone does not demonstrate annotation savings because the present selectors already know every candidate's class.

The completed scaling study found a +7.74 percentage-point representative-selection gain over random at budget 100 (59.42% versus 51.69%). Positive mean gains persist through budget 800, then disappear at 1,600. The earlier fixed quality gate does not improve consistently at scale. The full 28,755-candidate head reaches 86.79%; no tested subset is within one point of that reference. See `results/scaling_v4/INTERPRETATION.md`, `REPORT.md`, and `EFFICIENCY.md`. The suite now has 76 passing tests plus one skipped network test.
