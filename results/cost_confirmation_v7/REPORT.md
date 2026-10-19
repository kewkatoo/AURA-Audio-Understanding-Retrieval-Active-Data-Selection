# Independent cost-rule confirmation

The fixed random-retention rule is tested on the previously unused Free Spoken Digit Dataset (FSDD). This study does not tune a new fraction, classifier or selector. It evaluates transfer of a development choice, not a novel acquisition algorithm.

Dataset revision: `26eb9aaf76e81b692f806f9140c2d2777410d7a1`. Six speakers, 3,000 recordings, 8 kHz mono. Each speaker is evaluated on all 500 recordings using models trained on the other five speakers. Training sizes after content exclusions: [2500]; transferred random budgets: [920]. The rule retains 64/175 of candidate rows, rounded up to a class-count multiple; C=1 throughout.
This custom speaker-disjoint protocol differs from the dataset's official index-based 0–4 test split. Speakers occur in training for other folds, so fold models are dependent. Every configuration and repeat is fitted before any evaluation prediction; no held-out score chooses the rule.

## Accuracy

Full-pool pooled accuracy: **69.03%**. Random-rule accuracy, averaged over five subset seeds: **66.71%**.
Full-minus-random loss: **2.327 percentage points**, exploratory paired speaker-bootstrap 95% interval **[1.020, 3.687]**.
Observed mean meets the one-point criterion: **False**. Exploratory upper interval below one point: **False**.
The bootstrap resamples six held-out speaker blocks and conditions on the fitted heads. It omits training-population uncertainty and does not remove dependence between folds' overlapping training sets. This small corpus is not a powered statistical noninferiority study. Passing an exploratory check would support this corpus only; failing it would show that this fixed fraction does not preserve the target margin here.

## Repeated timing

Three paired repeats per held-out speaker give **18 pairs**. Full-pool median warm end-to-end training time: **19.87 s**. Subset median: **7.83 s**.
Median paired random/full ratio: **0.382**, range **[0.325, 0.417]**. Median paired saving: **61.78%**. Random is faster in **18/18 pairs**.
These are actual sequential wall times for selection, raw decoding/resampling/VAD/normalization, uncached wav2vec2 inference, matrix assembly, selected-only scaling and logistic-regression fitting. GPU work is synchronized. They are not sums of independently measured components. AB/BA order is counterbalanced over folds/repeats, and the timed seed-zero subset is held fixed within each fold.
One process, two CPU threads and a resident encoder are used. Startup/synthetic raw-audio and classifier warm-up were measured separately: [5.473167799995281] seconds across 1 session(s). Inventory/provenance, checkpoint writes, startup, evaluation encoding and evaluation prediction are excluded from trial times. OS caches are not flushed. No training feature cache is read. Repeats do not measure energy, hardware rental costs, cold-machine startup, or independent hardware sessions.

## Numerical repeatability

Identical embedding arrays across same-layout training repeats: **24/24 comparisons**. Maximum repeat embedding-coordinate difference: **0.00000000**. Maximum held-out prediction disagreement across repeats: **0.000%**.
Fresh training vectors are used directly, without a canonical training matrix or a relaxed reference-equality requirement. Evaluation vectors are computed once per held-out speaker and shared across its models. Same-layout repeatability does not establish invariance to different batch composition or hardware. Runtime records include model revision, dtype, attention backend, CUDA/cuDNN versions, TF32 settings and thread pools. The v6 cross-batch exemplar-instability finding remains separate.

## Scope and provenance

Class labels are known before class-balanced selection. This study establishes no annotation savings, no global optimality of a retention fraction, and no benefit for expensive fine-tuning. Absolute times and scores should not be compared directly with the different AudioMNIST cohort/hardware workload. Keep this now-observed corpus out of future tuning if claiming fresh confirmation.
All 60 fitted heads, 36 primary configurations, 36 timed trials, 18 timing pairs, 3,000 raw hashes, per-fold speaker/content separation, selected class budgets, embeddings, saved-model predictions and metrics are verified. Source declarations, model locks, session contexts and evaluation manifests are preserved.

Dataset: [FSDD, maintained by Jakobovski and contributors](https://github.com/Jakobovski/free-spoken-digit-dataset). Its README declares [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/). The source checkout and README are preserved locally; see DATASET_NOTICE.md. No dataset recordings were modified in the source checkout.
