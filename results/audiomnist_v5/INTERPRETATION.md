# Independent validation: what the result supports

The low-budget representative-selection benefit transfers from Speech Commands to AudioMNIST. It survives a control that samples uniformly from unique raw-byte contents. This strengthens the empirical case for class-balanced cluster exemplars when the downstream classifier can use only a small labeled subset.

It does not establish a new clustering algorithm, annotation savings, a universal ranking of selectors, or a reduction in total computation.

## Independent evaluation

The [AudioMNIST repository](https://github.com/soerenab/AudioMNIST), revision `630d7dab4c040882834de6fa21baf9a60372accd`, provides the dataset under its MIT license. Cite Becker et al., *AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark*, DOI [10.1016/j.jfranklin.2023.11.038](https://doi.org/10.1016/j.jfranklin.2023.11.038).

The predeclared seed-20261018 partition assigns 45 speakers / 22,500 recordings to candidate training and 15 speakers / 7,500 recordings to evaluation. Speakers and exact contents are disjoint across roles. There are two redundant exact copies within the candidate pool. The original 48 kHz mono recordings pass through the same frozen 16 kHz preprocessing and wav2vec2-base mean-pooling pipeline used earlier.

All selector and classifier settings transfer from the earlier Speech Commands v3 choices. There is no new dataset tuning. Every one of the 101 models was fitted and checkpointed before held-out audio was prepared. This is a custom speaker split, not the authors' published benchmark, and the task remains speech classification.

## Results

Accuracy is averaged across five fixed selection seeds, except the single full-pool reference.

| Selected recordings | Random | Deduplicated random | Representatives | Quality-filtered representatives |
|---:|---:|---:|---:|---:|
| 100 | 76.63% | 76.92% | 85.61% | 85.06% |
| 200 | 85.58% | 85.13% | 88.14% | 87.69% |
| 400 | 88.61% | 88.46% | 91.42% | 91.56% |
| 800 | 91.44% | 91.50% | 93.39% | 93.46% |
| 1,600 | 93.82% | 93.83% | 94.55% | 94.73% |

Full pool, 22,500 recordings: **97.01%**.

The predeclared primary comparison is representatives minus deduplicated random at budget 100: **+8.69 percentage points**, exploratory paired speaker-bootstrap 95% interval **[+7.39, +9.77]**. At budgets 200/400/800/1,600, differences are +3.01/+2.96/+1.90/+0.73 points. The last interval is [+0.02, +1.45]; its proximity to zero and the number of comparisons make it weak evidence for a larger-budget advantage.

Intervals resample 15 held-out speakers and condition on the fixed seed models. They do not include all training/search uncertainty and have no multiplicity adjustment. Previously tuned classifier C differs across some method/budget pairs, so these comparisons evaluate frozen recipes. At budget 100, random, deduplicated random and representatives all use C=1.

The best tested subset, quality-filtered representatives at 1,600, uses 7.11% of candidate recordings but remains 2.29 points below the full reference. No tested subset is within one accuracy point of full-pool performance. A small retained fraction must be presented together with this accuracy loss.

## What changed after the duplicate control

On the already observed Speech Commands v4 cohort, representatives exceed deduplicated random by +8.31/+3.10/+1.68 points at budgets 100/200/400, with exploratory intervals above zero. At 800/1,600 the differences are -0.64/-0.60 points with intervals spanning zero. This post-hoc control narrows the cross-dataset finding to the smallest budgets; it is not another fresh confirmation cohort.

Deduplication changes content sampling weights and the seed-to-subset mapping, not only the chance of repeated copies. Exact-byte matching does not detect every acoustically similar recording. AudioMNIST's very low duplicate count provides an additional check that substantial representative gains can occur without much exact redundancy.

Quality-filtered representatives perform similarly to unfiltered representatives on AudioMNIST. Combined with their weaker Speech Commands scaling results and the earlier non-monotonic corruption diagnostics, this does not justify making the current quality gate essential to the main method.

## Benefits and costs

The demonstrated benefit is better held-out accuracy for the same small retained labeled subset. This could be useful when storage, downstream training, or repeated reuse limits the usable subset, provided the measured accuracy loss is acceptable. Those deployment benefits remain conditional on actual task costs.

Candidate preparation took 536.18 seconds; held-out preparation took 193.52 seconds. The full-pool head took 5.03 seconds to fit. Mean selection plus head-fit times were approximately 12.26 + 0.13 seconds for 100 representatives and 124.52 + 0.48 seconds for 1,600. Four compute workers shared the CPU, so these are observed execution times rather than controlled speed benchmarks. Nevertheless, this run supplies no evidence of total compute savings: all candidate embeddings are prepared and clustering is substantial relative to this cheap classifier.

All candidate class labels are known during class-balanced selection. This cannot demonstrate label-acquisition savings. A cheaper representation that selects before expensive encoding, or an expensive downstream learner, is the next practical cost experiment. Active annotation acquisition needs a separate protocol that does not access labels before selection.

## Verification and next work

Verification checks all 30,000 raw hashes, source and dataset declarations, identical encoders across roles, matrix/metadata alignment, speaker/content separation, all 101 selected subsets and model checkpoints, exact class budgets, and metric reproduction from saved predictions. Package versions, execution context and all report/artifact hashes are saved alongside the results. The project suite passes 81 tests with one optional network test skipped.

The research prototype now has evidence from two speech corpora. Remaining priorities are a cost experiment, a clear decision on quality calibration, and a final reproducible research handoff. Broader modality claims, novel algorithm claims, annotation acquisition and retrieval/product components need their own methods and evaluation. Do not use the now-observed AudioMNIST holdout to tune a new method and then describe that same cohort as fresh confirmation.

See `REPORT.md`, `speaker_bootstrap.csv`, `observed_costs.csv`, `verification.json`, `experiment.json`, and the root `COMPLETION.md` for supporting artifacts and remaining acceptance criteria.
