# Confirmation uncertainty

Each locked model was refitted and its original confirmation accuracy reproduced before analysis. No setting or selected ID was changed. Intervals resample speakers with replacement and count all their recordings together; seed models are averaged within each recording.

| Strategy | Budget | Difference from random | 95% speaker-bootstrap interval |
|---|---:|---:|---:|
| combined | 100 | +1.24 pp | [-2.37, +4.89] pp |
| combined | 200 | +0.59 pp | [-3.17, +4.42] pp |
| combined | 400 | +1.56 pp | [-1.99, +5.14] pp |
| combined | 800 | +0.22 pp | [-2.80, +3.30] pp |
| diversity | 100 | -8.39 pp | [-11.92, -4.38] pp |
| diversity | 200 | -7.63 pp | [-11.67, -3.40] pp |
| diversity | 400 | -0.05 pp | [-3.43, +3.44] pp |
| diversity | 800 | +1.72 pp | [-0.33, +3.91] pp |
| quality | 100 | +1.24 pp | [-2.37, +4.89] pp |
| quality | 200 | +0.59 pp | [-3.17, +4.42] pp |
| quality | 400 | -2.47 pp | [-6.28, +1.15] pp |
| quality | 800 | -4.62 pp | [-8.56, -0.84] pp |
| quality_representative | 100 | +5.65 pp | [+2.51, +8.79] pp |
| quality_representative | 200 | +1.77 pp | [-0.66, +4.35] pp |
| quality_representative | 400 | +2.31 pp | [-0.60, +4.90] pp |
| quality_representative | 800 | +0.91 pp | [-1.75, +3.68] pp |
| random | 100 | +0.00 pp | [+0.00, +0.00] pp |
| random | 200 | +0.00 pp | [+0.00, +0.00] pp |
| random | 400 | +0.00 pp | [+0.00, +0.00] pp |
| random | 800 | +0.00 pp | [+0.00, +0.00] pp |
| representative | 100 | +5.00 pp | [+2.18, +8.15] pp |
| representative | 200 | +1.08 pp | [-1.73, +3.88] pp |
| representative | 400 | +1.29 pp | [-1.49, +4.18] pp |
| representative | 800 | -1.08 pp | [-3.46, +1.30] pp |

These exploratory intervals condition on the existing training/tuning experiment. They do not account for hyperparameter search, multiple comparisons, or a new dataset/domain. They are not a reason to tune further against this confirmation cohort.