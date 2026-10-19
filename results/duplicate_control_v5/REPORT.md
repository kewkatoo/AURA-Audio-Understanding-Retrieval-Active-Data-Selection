# Duplicate-control results

Mean held-out accuracy across five fixed selection seeds:

| Method | Budget | Accuracy |
|---|---:|---:|
| deduplicated_random | 100 | 51.11% |
| deduplicated_random | 200 | 62.48% |
| deduplicated_random | 400 | 70.61% |
| deduplicated_random | 800 | 77.83% |
| deduplicated_random | 1600 | 80.98% |
| quality_representative | 100 | 59.30% |
| quality_representative | 200 | 63.87% |
| quality_representative | 400 | 69.50% |
| quality_representative | 800 | 75.39% |
| quality_representative | 1600 | 79.42% |
| random | 100 | 51.69% |
| random | 200 | 62.92% |
| random | 400 | 70.51% |
| random | 800 | 75.78% |
| random | 1600 | 81.04% |
| representative | 100 | 59.42% |
| representative | 200 | 65.58% |
| representative | 400 | 72.28% |
| representative | 800 | 77.19% |
| representative | 1600 | 80.39% |

Representative minus deduplicated random (percentage points; paired speaker bootstrap):

| Budget | Difference | Exploratory 95% interval |
|---:|---:|---:|
| 100 | +8.31 | [+6.83, +9.90] |
| 200 | +3.10 | [+1.74, +4.45] |
| 400 | +1.68 | [+0.43, +3.08] |
| 800 | -0.64 | [-1.66, +0.39] |
| 1600 | -0.60 | [-1.59, +0.36] |

This is a post-hoc diagnostic on the already observed v4 holdout. It is not fresh confirmation.
Deduplication samples uniformly among unique raw-byte contents, retaining the smallest ID per content.
This also changes content sampling weights and the seed-to-subset mapping; it is not a pure paired removal of duplicates.
Exact-byte matching does not identify all acoustically similar recordings.

Intervals resample held-out speakers and condition on the five fixed models. They do not include hyperparameter/search uncertainty, have no multiplicity adjustment, and should be treated as exploratory.
Comparisons evaluate frozen selector-plus-classifier recipes: previously tuned C can differ by method/budget, so all differences cannot be attributed solely to selection. At budget 100, random, deduplicated random, and representatives share C=1.
Selection uses frozen wav2vec2 embeddings for every candidate. Head-training reductions alone do not establish total compute savings.
