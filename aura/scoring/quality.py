"""
Prototype acoustic quality score.

IMPORTANT: this is NOT a perceptual audio-quality metric (not PESQ, not
MOS-correlated, not validated against human ratings). It is a simple,
transparent, reproducible combination of four measurable DSP signals,
intended as a defensible starting point for subset-selection experiments,
not a claim about how a clip actually sounds to a human listener.

Q(x) = 0.25 * SNR_norm
     + 0.25 * (1 - clipping_ratio)
     + 0.25 * speech_ratio
     + 0.25 * RMS_stability_norm

Normalization (min-max) for SNR and RMS_stability is fit ONLY on the
training pool's own feature distribution and then applied unchanged to
validation/test, to avoid leaking validation/test statistics into a score
that will later be used for training-set subset selection. clipping_ratio
and speech_ratio are already bounded in [0, 1] by construction and are used
directly, no fitted normalization needed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from aura.scoring.features import DSPFeatures

DEFAULT_WEIGHTS = {
    "snr": 0.25,
    "clipping": 0.25,
    "speech_ratio": 0.25,
    "rms_stability": 0.25,
}


@dataclass
class QualityNormStats:
    """Min/max fit on the training pool only. Persist and reuse these for
    validation/test scoring -- never refit on those splits."""
    snr_min: float
    snr_max: float
    rms_stability_min: float
    rms_stability_max: float

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "QualityNormStats":
        return QualityNormStats(**d)


@dataclass
class QualityScore:
    snr_norm: float
    clipping_component: float   # (1 - clipping_ratio)
    speech_ratio_component: float
    rms_stability_norm: float
    quality_score: float        # weighted combination

    def to_dict(self) -> dict:
        return asdict(self)


def fit_quality_norm_stats(features_list: list[DSPFeatures]) -> QualityNormStats:
    """Fit min/max normalization statistics on a list of DSPFeatures. Call
    this ONLY on the training-pool features."""
    if not features_list:
        raise ValueError("Cannot fit quality normalization stats on an empty feature list.")

    snr_values = np.array([f.snr_db for f in features_list], dtype=np.float64)
    stability_values = np.array([f.rms_stability for f in features_list], dtype=np.float64)

    return QualityNormStats(
        snr_min=float(np.min(snr_values)),
        snr_max=float(np.max(snr_values)),
        rms_stability_min=float(np.min(stability_values)),
        rms_stability_max=float(np.max(stability_values)),
    )


def _min_max_norm(value: float, vmin: float, vmax: float) -> float:
    if vmax - vmin < 1e-9:
        # Degenerate case: entire pool has (near-)identical value. Return
        # a neutral 0.5 rather than dividing by ~zero.
        return 0.5
    normalized = (value - vmin) / (vmax - vmin)
    return float(np.clip(normalized, 0.0, 1.0))


def compute_quality_score(
    features: DSPFeatures,
    norm_stats: QualityNormStats,
    weights: dict[str, float] = None,
) -> QualityScore:
    """Compute the prototype quality score for one sample, using
    normalization statistics fit on the training pool (norm_stats applies
    unchanged regardless of which split `features` came from -- this is
    what prevents validation/test leakage)."""
    weights = weights or DEFAULT_WEIGHTS

    snr_norm = _min_max_norm(features.snr_db, norm_stats.snr_min, norm_stats.snr_max)
    clipping_component = float(np.clip(1.0 - features.clipping_ratio, 0.0, 1.0))
    speech_ratio_component = float(np.clip(features.speech_ratio, 0.0, 1.0))
    rms_stability_norm = _min_max_norm(
        features.rms_stability, norm_stats.rms_stability_min, norm_stats.rms_stability_max
    )

    quality_score = (
        weights["snr"] * snr_norm
        + weights["clipping"] * clipping_component
        + weights["speech_ratio"] * speech_ratio_component
        + weights["rms_stability"] * rms_stability_norm
    )

    return QualityScore(
        snr_norm=snr_norm,
        clipping_component=clipping_component,
        speech_ratio_component=speech_ratio_component,
        rms_stability_norm=rms_stability_norm,
        quality_score=float(quality_score),
    )
