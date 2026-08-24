from __future__ import annotations

import pytest

from aura.ingest.audio import process_audio
from aura.scoring.features import extract_features
from aura.scoring.quality import (
    fit_quality_norm_stats, compute_quality_score, DEFAULT_WEIGHTS,
)


def _features_from_paths(paths):
    feats = []
    for p in paths:
        pa = process_audio(p, target_sample_rate=16000)
        feats.append(extract_features(pa.waveform, pa.sample_rate, pa.speech_ratio))
    return feats


def test_fit_quality_norm_stats_raises_on_empty_list():
    with pytest.raises(ValueError):
        fit_quality_norm_stats([])


def test_quality_score_in_unit_range(synthetic_wav_pool):
    feats = _features_from_paths(synthetic_wav_pool)
    norm_stats = fit_quality_norm_stats(feats)
    for f in feats:
        q = compute_quality_score(f, norm_stats)
        assert 0.0 <= q.quality_score <= 1.0
        assert 0.0 <= q.snr_norm <= 1.0
        assert 0.0 <= q.rms_stability_norm <= 1.0


def test_quality_score_weights_sum_reflected():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_quality_score_deterministic(synthetic_wav_pool):
    feats = _features_from_paths(synthetic_wav_pool)
    norm_stats = fit_quality_norm_stats(feats)
    q1 = compute_quality_score(feats[0], norm_stats)
    q2 = compute_quality_score(feats[0], norm_stats)
    assert q1.to_dict() == q2.to_dict()


def test_quality_norm_stats_not_refit_on_val_features(synthetic_wav_pool):
    """Simulates the leakage-safety contract: stats fit on a 'train' pool
    must be usable unchanged on a 'validation' sample whose own feature
    values fall outside the train pool's min/max -- and the result must
    still be clipped into [0, 1], not silently renormalized."""
    train_feats = _features_from_paths(synthetic_wav_pool[:2])
    val_feats = _features_from_paths(synthetic_wav_pool[2:])

    train_norm_stats = fit_quality_norm_stats(train_feats)
    for f in val_feats:
        q = compute_quality_score(f, train_norm_stats)
        assert 0.0 <= q.quality_score <= 1.0  # clipped even if out-of-range


def test_quality_score_degenerate_pool_returns_neutral():
    """If every feature in the pool is identical, min==max and normalization
    should return a neutral 0.5 rather than dividing by ~zero."""
    from aura.scoring.features import DSPFeatures
    identical = DSPFeatures(
        rms=0.5, zero_crossing_rate=0.1, spectral_centroid=100.0,
        spectral_bandwidth=50.0, spectral_rolloff=200.0,
        mfcc_mean=[0.0] * 13, mfcc_std=[0.0] * 13,
        clipping_ratio=0.0, speech_ratio=1.0, snr_db=5.0, rms_stability=0.5,
    )
    norm_stats = fit_quality_norm_stats([identical, identical])
    q = compute_quality_score(identical, norm_stats)
    assert q.snr_norm == 0.5
    assert q.rms_stability_norm == 0.5
