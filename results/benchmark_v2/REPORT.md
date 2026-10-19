# Corrected-pipeline selection benchmark

Candidate pool: 2000; validation: 500; test: 4074.

StandardScaler + multinomial logistic regression, fitted only on each training subset. C and combined alpha were selected by mean validation accuracy across selection seeds; ties use macro-F1, then smaller alpha/C. Test data were loaded after choices were saved.

| Strategy | Budget | Test accuracy ± seed std | Macro-F1 | Alpha | C |
|---|---:|---:|---:|---:|---:|
| combined | 100 | 57.88% ± 0.00 pp | 0.5724 | 1.0 | 1 |
| combined | 200 | 64.92% ± 0.00 pp | 0.6488 | 1.0 | 1 |
| combined | 400 | 74.82% ± 0.00 pp | 0.7476 | 0.25 | 1 |
| combined | 800 | 81.15% ± 0.00 pp | 0.8109 | 0.25 | 0.1 |
| diversity | 100 | 40.48% ± 2.67 pp | 0.3857 | — | 1 |
| diversity | 200 | 56.93% ± 1.63 pp | 0.5640 | — | 1 |
| diversity | 400 | 71.30% ± 0.71 pp | 0.7098 | — | 1 |
| diversity | 800 | 81.27% ± 0.57 pp | 0.8119 | — | 1 |
| full_pool | 2000 | 85.00% ± n/a pp | 0.8500 | — | 10 |
| quality | 100 | 57.88% ± 0.00 pp | 0.5724 | — | 1 |
| quality | 200 | 64.92% ± 0.00 pp | 0.6488 | — | 1 |
| quality | 400 | 72.78% ± 0.00 pp | 0.7299 | — | 1 |
| quality | 800 | 78.94% ± 0.00 pp | 0.7897 | — | 1 |
| random | 100 | 56.36% ± 2.97 pp | 0.5584 | — | 1 |
| random | 200 | 66.13% ± 1.55 pp | 0.6588 | — | 0.1 |
| random | 400 | 75.36% ± 0.69 pp | 0.7527 | — | 1 |
| random | 800 | 80.99% ± 0.55 pp | 0.8095 | — | 0.1 |

Standard deviations describe selection-seed variation, not independent test datasets. Quality-only and the classifier optimizer are deterministic. The full-pool baseline is the configured candidate pool, not all available training recordings.

Preparation timings are stored per split in the input manifests; per-run selection/head fit timings are in runs.csv. These measurements do not establish a total compute saving over training a model directly on raw audio.