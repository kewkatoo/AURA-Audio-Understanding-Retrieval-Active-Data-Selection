# Quality signal diagnostics

Controlled transforms on 100 class-balanced training recordings; 5 speaker-disjoint classifier folds over 2000 training examples.

| Transform | Mean quality | Change from clean | Fraction scoring lower than clean |
|---|---:|---:|---:|
| clean | 0.4892 | +0.0000 | 0.0% |
| clip_gain3 | 0.4935 | +0.0043 | 53.0% |
| clip_gain8 | 0.4781 | -0.0111 | 66.0% |
| clip_reference | 0.4905 | +0.0013 | 19.0% |
| gain_0.25 | 0.4575 | -0.0317 | 65.0% |
| noise_0db | 0.4362 | -0.0530 | 81.0% |
| noise_10db | 0.3813 | -0.1079 | 88.0% |
| noise_20db | 0.4095 | -0.0797 | 89.0% |
| retain_prefix_0.25 | 0.3654 | -0.1238 | 63.0% |
| retain_prefix_0.5 | 0.5064 | +0.0171 | 47.0% |
| silence | 0.0000 | -0.4892 | 99.0% |

Clipping transforms have a separate peak-scaled clean reference; compare the paired scores in clipping_pairs.csv rather than attributing all changes from original clean audio to clipping.

| Feature | Spearman correlation with held-out classifier loss |
|---|---:|
| quality_score | -0.1398 |
| snr | -0.1753 |
| speech_ratio | -0.0921 |
| clipping_ratio | +0.0236 |
| rms_stability | +0.1883 |

For quality, a negative correlation would support the hypothesis that higher quality predicts lower loss. These associations are descriptive, not causal or perceptual validation. No new quality weights were fitted. Truncation can remove the target word; it is a damage diagnostic, not a guaranteed label-preserving transform.

The activity detector can respond to absolute gain and can miss noisy speech. Controlled noise is additive float audio, not an automatically clipped PCM recording. Neither tuning validation nor the existing test/confirmation sets were used.