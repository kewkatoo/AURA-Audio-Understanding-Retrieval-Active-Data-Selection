# Redundancy and data-efficiency diagnostics

These post-study diagnostics do not change model choices or selections. Candidate fractions describe recording-count reduction, not annotation savings.

| Strategy | Budget | Candidate fraction | Gap to full-pool accuracy | Duplicate contents selected, mean |
|---|---:|---:|---:|---:|
| combined | 100 | 0.35% | 44.44 pp | 7.0 |
| combined | 200 | 0.70% | 35.45 pp | 11.0 |
| combined | 400 | 1.39% | 25.72 pp | 0.0 |
| combined | 800 | 2.78% | 12.31 pp | 0.0 |
| combined | 1600 | 5.56% | 5.21 pp | 0.0 |
| diversity | 100 | 0.35% | 59.07 pp | 0.0 |
| diversity | 200 | 0.70% | 49.15 pp | 0.0 |
| diversity | 400 | 1.39% | 31.07 pp | 0.0 |
| diversity | 800 | 2.78% | 16.11 pp | 0.0 |
| diversity | 1600 | 5.56% | 7.23 pp | 0.0 |
| quality | 100 | 0.35% | 44.44 pp | 7.0 |
| quality | 200 | 0.70% | 35.45 pp | 11.0 |
| quality | 400 | 1.39% | 25.22 pp | 20.0 |
| quality | 800 | 2.78% | 19.12 pp | 36.0 |
| quality | 1600 | 5.56% | 10.97 pp | 75.0 |
| quality_representative | 100 | 0.35% | 27.49 pp | 0.0 |
| quality_representative | 200 | 0.70% | 22.92 pp | 0.0 |
| quality_representative | 400 | 1.39% | 17.29 pp | 0.0 |
| quality_representative | 800 | 2.78% | 11.40 pp | 0.0 |
| quality_representative | 1600 | 5.56% | 7.37 pp | 0.0 |
| random | 100 | 0.35% | 35.10 pp | 0.0 |
| random | 200 | 0.70% | 23.87 pp | 0.0 |
| random | 400 | 1.39% | 16.29 pp | 0.6 |
| random | 800 | 2.78% | 11.01 pp | 1.6 |
| random | 1600 | 5.56% | 5.75 pp | 3.8 |
| representative | 100 | 0.35% | 27.37 pp | 0.0 |
| representative | 200 | 0.70% | 21.21 pp | 0.0 |
| representative | 400 | 1.39% | 14.51 pp | 0.0 |
| representative | 800 | 2.78% | 9.60 pp | 0.0 |
| representative | 1600 | 5.56% | 6.41 pp | 0.0 |

Raw-content duplicates can partly explain differences between selectors. A future deduplicated-random control is needed to isolate representation coverage from simple deduplication. Do not tune that control against this now-observed holdout.

Selection and fit times include a mixture of sequential and four-worker execution. They are observed wall times with contention, not a controlled algorithm-speed comparison. Preparation costs must also be counted before claiming compute savings.