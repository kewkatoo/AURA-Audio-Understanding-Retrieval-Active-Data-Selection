# Development accuracy and acquisition cost

This study uses only the former AudioMNIST candidate training pool: 35 candidate speakers and 10 development speakers. Earlier holdout audio, representations and predictions are not used. Parent split declarations are read for provenance. These are development results, not fresh independent confirmation.

All heads use C=1. MFCC selection uses 13 source MFCC means and 13 standard deviations, candidate-only standardization, unit normalization, and the same class-wise KMeans exemplars. Five seeds estimate accuracy; cost replays use seed 0 only. Random subsets at 3,200 and 6,400 provide additional cost controls.

| Method | Recordings | Mean accuracy | Total seconds | Gap to full (points) | Dominated on tested mean/cost grid |
|---|---:|---:|---:|---:|---|
| full_pool | 17500 | 97.52% | 104.99 | 0.00 | False |
| mfcc_representative | 100 | 79.02% | 82.33 | 18.50 | True |
| mfcc_representative | 400 | 90.41% | 84.18 | 7.11 | True |
| mfcc_representative | 1600 | 94.50% | 95.11 | 3.02 | True |
| random | 100 | 79.27% | 2.85 | 18.25 | False |
| random | 400 | 88.78% | 4.28 | 8.74 | False |
| random | 1600 | 94.26% | 11.82 | 3.26 | False |
| random | 3200 | 95.88% | 20.00 | 1.64 | False |
| random | 6400 | 96.71% | 37.55 | 0.81 | False |
| wav2vec_representative | 100 | 85.18% | 108.41 | 12.34 | True |
| wav2vec_representative | 400 | 91.62% | 120.11 | 5.90 | True |
| wav2vec_representative | 1600 | 94.94% | 164.46 | 2.58 | True |

Full-pool reference: 97.52% at 104.99 measured component seconds.
Cheapest tested subset within one percentage point of the full mean: random, budget 6400, 96.71%, 37.55 seconds (0.358 times the full-pool component total). This development choice needs fresh confirmation.
Its full-minus-subset accuracy gap is 0.812 points; exploratory paired speaker-bootstrap 95% interval [0.440, 1.192]. Check the interval against the one-point margin; the mean criterion alone does not establish noninferiority.

## What the timing includes

One sequential process with two CPU threads and CUDA encoding. Fresh raw decoding, resampling/preprocessing, feature computation and neural inference bypass persistent feature/embedding caches. MFCC and random selections are locked before any candidate neural encoding in this run. Fresh neural outputs are checked against the frozen representations.
A recipe total adds model setup/warm-up, required descriptor extraction/normalization, required raw preprocessing plus encoding, selection, and head fitting. Full-pool and wav2vec selection share one measured all-candidate primitive; random and MFCC have separate selected-only raw encoding replays. Accuracy uses canonical frozen training/development embeddings for consistency. These are measured component totals, not independent complete pipeline replications.
Python/library initialization, inventory/provenance checks, construction/comparison of benchmark-only canonical reference matrices, checkpoint serialization, common evaluation preparation/prediction, and previously completed downloads are excluded. OS file caches and loaded library kernels are warm. Model weights come from the local cache; setup and a synthetic neural warm-up are explicitly included in each recipe total. Costs are not energy, hardware rental prices, or cold-machine startup measurements.
Timing has one replay per subset configuration and no timing uncertainty interval. Component order, warm caches and thermal state can affect the observations. A globally optimal cost frontier is not established by this finite grid.

The reported frontier allows different retained-recording budgets. A method dominated on that grid can still be relevant when there is an additional hard limit on retained recordings. Fixed-budget accuracy and cross-budget cost are different decision criteria.

## Numerical audit

Before development scoring, a strict per-coordinate fresh/reference check failed. A candidate-only control reproduced differences between the same clip encoded alone and in an equal-length batch, while repeating the same single-clip call was identical. [PyTorch documents that batch shapes can change floating-point results](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html). This is not the earlier padding defect: the control uses identical lengths and no padding.
The protocol was amended before any development predictions. The stopped pass and original runner are preserved in acquisition_cost_v6_strict_check. Current accuracy/selection uses canonical frozen embeddings. Fresh uncached timed vectors are checked for shape, ID alignment, finiteness, unit norm and a predeclared minimum cosine of 0.99 to the corresponding canonical vector; coordinate and cosine drift are recorded, not treated as numerical identity.
Full-pass maximum coordinate difference: 0.014161; minimum cosine: 0.993659. See full_neural.json and replay JSONs for all measured differences.
After all timing, seed-zero shadow fits use fresh training vectors and, for wav2vec representatives, fresh-vector reselection. They are evaluated on the same canonical development embeddings. numerical_sensitivity.csv reports accuracy changes, prediction disagreement and subset overlap. This is a sensitivity diagnostic, not independent full-pipeline evaluation. Precision/backend settings should enter future artifact contracts when strict numerical reproduction is required.

## Limits and next decision

The same known candidate labels support class-balanced selection in every method. No annotation-acquisition savings are established. A cost improvement at lower accuracy must be stated together with that accuracy loss. Cross-budget comparisons use means across five fixed seeds while the timed replay uses seed 0.
Speaker-bootstrap differences condition on the fixed models, resample only the 10 development speakers, and have no multiplicity correction. The within-one-point check compares observed means and is not a statistical noninferiority test. All development scores are now observed; method changes and model choices require new confirmation data before a general performance claim.
See accuracy_cost.csv, costs.csv, speaker_bootstrap.csv, timing_context.json, protocol.json and verification.json for the measured components and provenance.
