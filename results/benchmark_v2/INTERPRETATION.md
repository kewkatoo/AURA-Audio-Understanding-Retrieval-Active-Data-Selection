# What the first corrected benchmark establishes

Run date: 18 October 2026. This evaluates frozen Wav2Vec2 embeddings with a standardized multinomial logistic-regression head on the existing 2,000-example, ten-class Speech Commands candidate pool. Validation contains 500 examples; the official ten-class test split contains 4,074. The complete available training split has 30,769 examples and has not yet been evaluated.

## Outcome

**There is no consistent advantage for the combined AURA selector over class-balanced random selection in this experiment.** The pipeline is now capable of measuring the claim, but the current quality/diversity objective has not demonstrated it reliably.

| Training budget | Random accuracy, mean ± seed std | Quality | Diversity, mean ± seed std | Combined | Combined minus random |
|---|---:|---:|---:|---:|---:|
| 100 | 56.36% ± 2.97 pp | 57.88% | 40.48% ± 2.67 pp | 57.88% | +1.52 pp |
| 200 | 66.13% ± 1.55 pp | 64.92% | 56.93% ± 1.63 pp | 64.92% | -1.21 pp |
| 400 | 75.36% ± 0.69 pp | 72.78% | 71.30% ± 0.71 pp | 74.82% | -0.54 pp |
| 800 | 80.99% ± 0.55 pp | 78.94% | 81.27% ± 0.57 pp | 81.15% | +0.16 pp |

The **full configured candidate pool (2,000 examples) achieves 85.00%** test accuracy and 0.8500 macro-F1. Combined selection at budget 800 uses 40% of those examples and is 3.85 percentage points below that baseline. It does not meet a within-one-percentage-point full-pool target.

Quality and combined selection produced the same subset across seeds at their chosen settings; their test seed standard deviation is zero. This is algorithmic determinism, not five independent confirmations of robustness. Random and diversity subsets vary across seeds. All results share the same test recordings, so seed standard deviations are not uncertainty estimates over new datasets or deployment domains.

## Protocol

The run fitted **483 candidate heads** across four budgets, five selection seeds, three regularization strengths (C = 0.1, 1, 10), and five combined weights (alpha = 0, 0.25, 0.5, 0.75, 1). It produced **81 final test evaluations**, including one deterministic full-pool baseline.

For each strategy and budget, C and combined alpha were selected using mean validation accuracy across seeds. Ties were resolved using validation macro-F1, then lower alpha/C. Every choice was saved before the evaluator opened test artifacts. Standardization was fitted on the selected training examples only.

Validation chose alpha **1.0** at budgets 100 and 200, reducing combined selection to quality-only selection. It chose alpha **0.25** at budgets 400 and 800. An alpha of 0.25 means a nominal 25% quality weight in the existing formula; it does not mean a measured 25% causal contribution to results.

The source, configuration, package versions, model revision, artifact hashes, and input split manifests are saved in `experiment.json`. Selections are saved in `selections/`; all validation candidates are in `validation_sweep.csv`; final per-run metrics and timings are in `runs.csv`; per-class recall and confusion matrices are in `per_class.json`. `paired_differences.csv` contains per-seed percentage-point differences relative to random selection.

## Correctness repairs supporting this run

- Equal-length, unpadded encoder batches avoid padding-induced convolutional normalization changes. A real pretrained-model check gave cosine similarity approximately 1.0 and zero observed element-wise difference for a short clip alone versus beside a longer clip. The audit's earlier diagnostic gave similarity 0.449.
- Acoustic diagnostics are measured before peak normalization/trimming, preserving source clipping and background context.
- Activity uses a noise-floor threshold with a margin and absolute floor. Training activity-ratio standard deviation is now 0.1467 rather than 0.00123; it is no longer effectively fixed by the median threshold.
- Silence has zero RMS stability and zero composite quality; the short-silence detector bug was corrected.
- Training normalization statistics are persisted and verified for evaluation-only reruns.
- Selectors enforce training-only data, unique IDs, required quality values, and finite/unit-normalized embeddings where needed.
- Source/runtime/model fingerprints invalidate stale preparation cache values. V2 feature caches contain DSP feature records, not waveform arrays.
- Original data and selection outputs remain available; corrected data use `data/v2` and corrected demonstration selections use `results/selection_v2`.

Verification: **62 tests passed, one dataset-download test skipped**. Saved corrected artifacts passed metadata/index, finite-value, normalization, and referenced-file checks. Raw-byte content hashing found no cross-split content overlaps in the evaluated pool; sample IDs and speaker IDs are also disjoint.

A controlled synthetic noise diagnostic reduced the active/inactive energy contrast from 42.71 dB to 22.72 dB as added noise amplitude increased from 0.001 to 0.01. At noise amplitude 0.1, no activity was detected. This is a narrow sanity check, not perceptual or general task-quality validation.

## Cost and limitations

On this machine (RTX 4050 Laptop GPU), split preparation took approximately 70.30 seconds for train, 17.17 for validation, and 107.42 for test: **194.89 seconds total**. The complete validation-tuning and test benchmark took **122.62 seconds**. These are single-run measurements, not controlled performance benchmarks.

At budget 800, combined selection averaged about 0.202 seconds for selection and 0.390 seconds for fitting the chosen head. Fitting the full-pool head took 0.439 seconds. These measurements do **not** demonstrate a compute saving from combined selection for this cheap classifier. They exclude any claim about expensive raw-audio fine-tuning, repeated large-model training, or annotation cost.

The activity detector remains heuristic and can reject useful low-energy/noisy speech. RMS stability is not a validated proxy for training usefulness. The experiment uses one candidate-pool sample, one task/dataset, one encoder/pooling choice, one head family, and a small validation set. No significance or broad generalization claim is warranted from these results.

## Recommended next research step

Preserve this run as the baseline. Use validation or a fresh experiment to compare a modest quality gate followed by representative selection against the current weighted sum. Examine density/representative clustering, speaker coverage, and which examples farthest-point selection favors; its poor small-budget performance is consistent with an outlier-selection problem, but that explanation has not been established here.

Validate each quality component using controlled corruptions and task utility, then repeat on a larger candidate pool and another task. Because these test results are now visible, further design/tuning decisions should not be optimized against this test set. Use validation for development and an additional held-out evaluation for stronger claims.
