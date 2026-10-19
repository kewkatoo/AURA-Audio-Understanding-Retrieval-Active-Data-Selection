# Representative selection study

Training candidates: 2000. Tuning: 500. Fresh confirmation: 372 recordings from 59 unused speakers.

All settings were locked using tuning validation before confirmation preparation. The previously inspected test set was not opened. This small confirmation set is not comparable directly to the previous 4,074-recording test set.

| Strategy | Budget | Tuning accuracy | Confirmation accuracy ± seed std | Setting |
|---|---:|---:|---:|---|
| combined | 100 | 53.60% | 52.42% ± 0.00 pp | alpha=1; C=1 |
| combined | 200 | 63.80% | 61.83% ± 0.00 pp | alpha=1; C=1 |
| combined | 400 | 73.00% | 70.97% ± 0.00 pp | alpha=0.25; C=1 |
| combined | 800 | 80.20% | 76.61% ± 0.00 pp | alpha=0.25; C=0.1 |
| diversity | 100 | 39.52% | 42.80% ± 3.15 pp | default; C=1 |
| diversity | 200 | 52.84% | 53.60% ± 2.11 pp | default; C=1 |
| diversity | 400 | 67.80% | 69.35% ± 1.72 pp | default; C=1 |
| diversity | 800 | 79.32% | 78.12% ± 1.44 pp | default; C=1 |
| full_pool | 2000 | 84.60% | 79.57% ± n/a | default; C=10 |
| quality | 100 | 53.60% | 52.42% ± 0.00 pp | default; C=1 |
| quality | 200 | 63.80% | 61.83% ± 0.00 pp | default; C=1 |
| quality | 400 | 68.00% | 66.94% ± 0.00 pp | default; C=1 |
| quality | 800 | 76.40% | 71.77% ± 0.00 pp | default; C=1 |
| quality_representative | 100 | 61.48% | 56.83% ± 1.85 pp | retain=0.75; C=1 |
| quality_representative | 200 | 67.88% | 63.01% ± 0.77 pp | retain=0.9; C=1 |
| quality_representative | 400 | 74.36% | 71.72% ± 1.58 pp | retain=0.75; C=1 |
| quality_representative | 800 | 79.88% | 77.31% ± 0.45 pp | retain=0.75; C=1 |
| random | 100 | 56.08% | 51.18% ± 2.21 pp | default; C=1 |
| random | 200 | 65.64% | 61.24% ± 2.24 pp | default; C=0.1 |
| random | 400 | 72.40% | 69.41% ± 2.25 pp | default; C=1 |
| random | 800 | 79.44% | 76.40% ± 0.86 pp | default; C=0.1 |
| representative | 100 | 58.72% | 56.18% ± 1.25 pp | default; C=1 |
| representative | 200 | 66.64% | 62.31% ± 0.82 pp | default; C=1 |
| representative | 400 | 75.08% | 70.70% ± 1.78 pp | default; C=1 |
| representative | 800 | 79.32% | 75.32% ± 0.58 pp | default; C=1 |

Geometry diagnostics are computed only from the training pool. Lower average coverage distance means the pool is nearer to selected representatives. Higher selected kNN distance means more isolated selections. Neither is a substitute for classifier performance.

Variation across selection seeds is not uncertainty over new datasets. Deterministic algorithms can yield identical subsets and zero seed variation. Development used one small candidate pool and one previously used tuning set. No settings were adjusted after confirmation scores appeared.