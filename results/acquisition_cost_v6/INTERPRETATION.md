# What the cost experiment changes

For this frozen-embedding linear-probe task, smaller curated subsets do not automatically reduce computation. The tested MFCC and wav2vec2 curation recipes are dominated by simpler alternatives on the measured accuracy–cost grid when the retained subset size may vary.

The promising development baseline is a larger random subset. It retains a useful accuracy/cost trade-off without scanning every candidate with a selection representation. This is not an algorithmic novelty claim or fresh independent confirmation.

## Design and evidence

Only the previous AudioMNIST candidate pool is used: 35 speakers / 17,500 clips for training candidates and 10 speakers / 5,000 clips for development evaluation. Earlier holdout audio, representations and predictions are not used. All 56 classifiers use C=1. Five seeds estimate subset accuracy; seed 0 supplies one timing replay per configuration. Protocol choices were fixed before development predictions.

MFCC selection uses 13 source-audio coefficient means plus 13 standard deviations, candidate-only standardization, L2 normalization, and the existing class-wise KMeans exemplar algorithm. Random and MFCC selections are locked before neural encoding. The budget grid is 100/400/1,600 for each selector, with additional random controls at 3,200/6,400 and a full-pool head.

| Recipe | Retained recordings | Mean development accuracy | Measured component total |
|---|---:|---:|---:|
| Full pool | 17,500 | 97.52% | 104.99 s |
| Random | 6,400 | 96.71% | 37.55 s |
| Random | 3,200 | 95.88% | 20.00 s |
| Random | 1,600 | 94.26% | 11.82 s |
| MFCC representatives | 1,600 | 94.50% | 95.11 s |
| wav2vec2 representatives | 1,600 | 94.94% | 164.46 s |
| Random | 100 | 79.27% | 2.85 s |
| MFCC representatives | 100 | 79.02% | 82.33 s |
| wav2vec2 representatives | 100 | 85.18% | 108.41 s |

Random at 6,400 uses 36.57% of candidate recordings and approximately 64.23% less measured component time than the full-pool recipe. Its mean accuracy loss is 0.812 percentage points. The exploratory paired speaker-bootstrap interval for that loss is **[0.440, 1.192] points**. Although the observed mean meets the predeclared one-point criterion, the interval extends above one point. Statistical noninferiority and independent confirmation are not established.

## Why the cheaper selector did not win

Extracting MFCCs for all 17,500 candidates took 78.88 seconds. The full raw-preprocessing/wav2vec2 pass took 100.55 seconds on this CPU/GPU configuration. The smaller representation makes clustering fast, but scanning the entire pool still costs substantial time.

At 400 recordings, MFCC representatives improve over same-budget random by +1.632 points, exploratory interval [+0.948, +2.504]. At 100 and 1,600, differences are -0.252 and +0.244 points, with intervals spanning zero. Larger random subsets achieve higher accuracy for less measured time on the tested cross-budget grid.

The earlier small-budget wav2vec2 representative accuracy benefit remains: at 100 clips it reaches 85.18% versus random's 79.27%. But it first encodes every candidate and then clusters them. A cheap linear head cannot recover that selection overhead through reduced head-training time.

These findings answer different operational questions. If at most 100 clips can be retained, the stronger wav2vec2 subset may be useful despite its preparation cost. If acquisition/training compute is the main limit and more clips may be retained, larger random subsets are the stronger baseline here. The finite grid does not prove global optimality or a result for expensive fine-tuning.

## What the timing does and does not measure

The experiment uses one sequential process, two CPU threads and an RTX 4050 Laptop GPU. Timed passes bypass persistent feature/embedding caches and perform actual raw decoding, resampling, preprocessing and inference. Model setup/warm-up, required descriptors/normalization, selection and head fitting enter the recipe totals. Full-pool and wav2vec2 recipes share one measured all-candidate primitive; random/MFCC recipes have separate selected-only encoding replays.

Accuracy uses canonical frozen embeddings. These totals sum measured components; they are not independently repeated end-to-end pipeline timings. Library startup, downloads, benchmark-only reference-matrix construction/validation, provenance checks, checkpoint writes and common evaluation work are excluded. OS caches and runtime kernels are warm. There is one timing replay and no timing uncertainty interval; energy and hardware rental cost are not measured. The runner's slow benchmark-only reference-building step affects wall time but is excluded from the application component model.

## Numerical stability finding

A strict fresh/reference coordinate check stopped an earlier pass before model fitting or development scoring. An equal-length control reproduced differences between encoding the same clip alone and in a batch of identical copies, with no padding. This is consistent with [PyTorch's documented floating-point behavior](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html); the control does not isolate a specific kernel as the cause.

The amended full pass has minimum corresponding-vector cosine 0.993659 and median cosine approximately 0.999999966; 77 of 17,500 rows exceed a maximum coordinate difference of 0.001. This is approximate reproduction, not numerical identity.

After timing, fresh-vector seed-zero fits and wav2vec2 reselection show that small numerical differences can produce substantially different exemplar IDs. Jaccard overlap with the canonical wav2vec2 selections is 0.64/0.29/0.21 at budgets 100/400/1,600. Accuracy changes are +0.24/+0.82/-0.20 points, and prediction disagreement reaches 7.8%. Aggregate accuracy stability does not imply stable individual selections or predictions.

The main tables remain reproducible from saved canonical vectors, selected IDs and models. Future raw-to-subset reproducibility work should record numerical/backend settings and test selection stability; recording only model revision and random seed is insufficient for strict equality. Existing frozen results are not replaced by the sensitivity runs.

## Completion and next work

Verification passes for all 56 model jobs, all 22,500 development-source raw hashes, speaker/content separation, selected class budgets, fresh-vector audits, checkpoint hashes, saved metrics, sensitivity baselines and cost components. The project suite passes **87 tests**, with one optional network test skipped. Earlier interrupted/strict-check passes are retained separately and excluded from final performance measurements.

The supported project claim is currently data efficiency under a fixed retained-subset budget, with evidence on two speech corpora. The cost experiment supplies a stronger random baseline and limits compute-saving claims for the tested curation methods. It does not establish annotation savings or a new acquisition algorithm.

Next priorities are fresh confirmation and repeated timing of a chosen practical baseline, a decision on whether expensive downstream training is actually part of the intended use case, numerical-provenance/stability work, and the final reproducible research handoff. Quality calibration, unlabeled active acquisition and retrieval/product components remain separate work with their own evaluation requirements.

Supporting files: `REPORT.md`, `accuracy_cost.csv`, `costs.csv`, `speaker_bootstrap.csv`, `full_gap_bootstrap.csv`, `numerical_sensitivity.csv`, `NUMERICAL_CONTROL.md`, `verification.json` and `experiment.json`.
