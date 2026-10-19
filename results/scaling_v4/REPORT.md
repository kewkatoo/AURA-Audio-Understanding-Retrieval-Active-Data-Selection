# Full-corpus scaling benchmark

Available official-training recordings: 30769. Candidate pool: 28755. Fresh speaker holdout: 2014 recordings from 250 speakers. Duplicate exclusions: 0.

All selection/head settings transferred from the earlier v3 tuning choices. New larger budgets use the largest earlier budget's settings. No parameter search was performed. Prior evaluation sets were not loaded. The full-pool baseline excludes this newly reserved holdout.

| Strategy | Budget | Holdout accuracy ± seed std | Macro-F1 | Selection time |
|---|---:|---:|---:|---:|
| combined | 100 | 42.35% ± 0.00 pp | 0.4127 | 1.06s |
| combined | 200 | 51.34% ± 0.00 pp | 0.5072 | 1.10s |
| combined | 400 | 61.07% ± 0.00 pp | 0.6076 | 1.44s |
| combined | 800 | 74.48% ± 0.00 pp | 0.7403 | 2.18s |
| combined | 1600 | 81.58% ± 0.00 pp | 0.8141 | 2.47s |
| diversity | 100 | 27.73% ± 2.47 pp | 0.2707 | 0.43s |
| diversity | 200 | 37.65% ± 0.38 pp | 0.3664 | 0.47s |
| diversity | 400 | 55.72% ± 0.92 pp | 0.5532 | 0.59s |
| diversity | 800 | 70.69% ± 0.41 pp | 0.7050 | 1.06s |
| diversity | 1600 | 79.56% ± 0.54 pp | 0.7942 | 1.31s |
| full_pool | 28755 | 86.79% ± n/a | 0.8674 | 0.01s |
| quality | 100 | 42.35% ± 0.00 pp | 0.4127 | 0.75s |
| quality | 200 | 51.34% ± 0.00 pp | 0.5072 | 0.85s |
| quality | 400 | 61.57% ± 0.00 pp | 0.6118 | 0.96s |
| quality | 800 | 67.68% ± 0.00 pp | 0.6742 | 1.27s |
| quality | 1600 | 75.82% ± 0.00 pp | 0.7558 | 1.20s |
| quality_representative | 100 | 59.30% ± 1.11 pp | 0.5905 | 8.78s |
| quality_representative | 200 | 63.87% ± 1.59 pp | 0.6359 | 16.73s |
| quality_representative | 400 | 69.50% ± 1.42 pp | 0.6918 | 33.49s |
| quality_representative | 800 | 75.39% ± 0.74 pp | 0.7518 | 63.93s |
| quality_representative | 1600 | 79.42% ± 0.45 pp | 0.7925 | 112.56s |
| random | 100 | 51.69% ± 0.68 pp | 0.5140 | 0.16s |
| random | 200 | 62.92% ± 1.99 pp | 0.6272 | 0.18s |
| random | 400 | 70.51% ± 1.26 pp | 0.7042 | 0.19s |
| random | 800 | 75.78% ± 0.98 pp | 0.7566 | 0.25s |
| random | 1600 | 81.04% ± 0.73 pp | 0.8096 | 0.26s |
| representative | 100 | 59.42% ± 1.15 pp | 0.5916 | 10.90s |
| representative | 200 | 65.58% ± 0.98 pp | 0.6532 | 17.82s |
| representative | 400 | 72.28% ± 0.64 pp | 0.7205 | 44.86s |
| representative | 800 | 77.19% ± 0.95 pp | 0.7705 | 83.99s |
| representative | 1600 | 80.39% ± 0.79 pp | 0.8022 | 154.36s |

Speaker-bootstrap differences versus random are in speaker_bootstrap.csv. These exploratory intervals condition on this candidate pool and transferred settings and are not adjusted for multiple comparisons. Scores cannot be directly compared with earlier evaluation cohorts.

This remains one dataset/task with a frozen embedding linear probe. It does not establish annotation savings, raw-audio fine-tuning savings, or independent-dataset generalization.