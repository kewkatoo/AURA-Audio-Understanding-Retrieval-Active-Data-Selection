# Independent AudioMNIST results

Mean held-out accuracy across five fixed selection seeds:

| Method | Budget | Accuracy |
|---|---:|---:|
| deduplicated_random | 100 | 76.92% |
| deduplicated_random | 200 | 85.13% |
| deduplicated_random | 400 | 88.46% |
| deduplicated_random | 800 | 91.50% |
| deduplicated_random | 1600 | 93.83% |
| full_pool | 22500 | 97.01% |
| quality_representative | 100 | 85.06% |
| quality_representative | 200 | 87.69% |
| quality_representative | 400 | 91.56% |
| quality_representative | 800 | 93.46% |
| quality_representative | 1600 | 94.73% |
| random | 100 | 76.63% |
| random | 200 | 85.58% |
| random | 400 | 88.61% |
| random | 800 | 91.44% |
| random | 1600 | 93.82% |
| representative | 100 | 85.61% |
| representative | 200 | 88.14% |
| representative | 400 | 91.42% |
| representative | 800 | 93.39% |
| representative | 1600 | 94.55% |

Representative minus deduplicated random (percentage points; paired speaker bootstrap):

| Budget | Difference | Exploratory 95% interval |
|---:|---:|---:|
| 100 | +8.69 | [+7.39, +9.77] |
| 200 | +3.01 | [+2.17, +3.81] |
| 400 | +2.96 | [+1.93, +4.04] |
| 800 | +1.90 | [+1.10, +2.78] |
| 1600 | +0.73 | [+0.02, +1.45] |

Dataset: [AudioMNIST](https://github.com/soerenab/AudioMNIST), revision 630d7dab4c040882834de6fa21baf9a60372accd (MIT).
Citation: Becker et al., AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark, DOI 10.1016/j.jfranklin.2023.11.038.
A declared 45/15 speaker split uses 22,500 training and 7,500 held-out recordings before any content exclusions. See partition.json for exact accepted counts.
This custom split is not directly comparable with the authors' published benchmark. Digits and a different recording environment test transfer, but remain speech classification.
All classifier and selector settings transfer from Speech Commands; no new dataset tuning is performed. Full-pool C=10 and subset C values differ by the previously locked choices.
Class labels are already known during selection. This measures subset training accuracy, not label-acquisition savings.

Intervals resample held-out speakers and condition on the five fixed models. They do not include hyperparameter/search uncertainty, have no multiplicity adjustment, and should be treated as exploratory.
Comparisons evaluate frozen selector-plus-classifier recipes: previously tuned C can differ by method/budget, so all differences cannot be attributed solely to selection. At budget 100, random, deduplicated random, and representatives share C=1.
Selection uses frozen wav2vec2 embeddings for every candidate. Head-training reductions alone do not establish total compute savings.
