"""
Deterministic DSP feature extraction, run on the *processed* (mono,
resampled, normalized, VAD-trimmed) waveform produced by
aura/ingest/audio.py.

All outputs are plain Python floats / lists of floats so the resulting
dict is directly serializable to Parquet/JSON with no custom encoding.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import librosa


N_MFCC = 13  # standard default; recorded here so mfcc_mean/mfcc_std length
             # is documented and stable across the codebase


@dataclass
class DSPFeatures:
    rms: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    mfcc_mean: list[float]   # length N_MFCC
    mfcc_std: list[float]    # length N_MFCC
    clipping_ratio: float
    speech_ratio: float      # passed through from preprocessing, kept here
                              # too since it's a feature used by scoring
    snr_db: float
    rms_stability: float     # 1 - coefficient_of_variation(frame_rms), in [0,1]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _frame_rms(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(waveform) < frame_length:
        return np.array([np.sqrt(np.mean(waveform.astype(np.float64) ** 2) + 1e-12)])
    frames = librosa.util.frame(waveform, frame_length=frame_length, hop_length=hop_length)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=0) + 1e-12)


def compute_clipping_ratio(waveform: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples within `threshold` of full-scale amplitude
    (waveform is assumed peak-normalized to [-1, 1] already)."""
    if len(waveform) == 0:
        return 0.0
    return float(np.mean(np.abs(waveform) >= threshold))


def estimate_snr_db(
    waveform: np.ndarray,
    sample_rate: int,
    is_speech: np.ndarray | None,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> float:
    """Rough SNR estimate: ratio of mean energy in speech-active frames to
    mean energy in non-speech frames, in dB. Falls back to a fixed frame
    energy split if no VAD mask is supplied. This is explicitly a coarse
    proxy, not a calibrated SNR measurement -- documented as part of the
    "prototype acoustic quality score", not a perceptual metric.
    """
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))
    frame_energy = _frame_rms(waveform, frame_length, hop_length) ** 2

    if is_speech is None or len(is_speech) != len(frame_energy):
        # Recompute a local VAD mask if none is provided or lengths
        # mismatch (e.g. features computed on a differently-trimmed signal).
        median = np.median(frame_energy)
        is_speech = frame_energy > median

    speech_energy = frame_energy[is_speech]
    noise_energy = frame_energy[~is_speech]

    if len(speech_energy) == 0 or len(noise_energy) == 0:
        # Can't separate signal from noise (e.g. all-speech or all-silence
        # clip); return a neutral 0 dB rather than a fabricated value.
        return 0.0

    speech_power = float(np.mean(speech_energy))
    noise_power = float(np.mean(noise_energy))
    if noise_power < 1e-12:
        noise_power = 1e-12

    snr = 10.0 * np.log10(speech_power / noise_power)
    return float(snr)


def compute_rms_stability(frame_rms: np.ndarray) -> float:
    """1 - coefficient_of_variation(frame_rms), clipped to [0, 1]. Higher
    means more stable (consistent) energy across the utterance."""
    mean_rms = float(np.mean(frame_rms))
    std_rms = float(np.std(frame_rms))
    if mean_rms < 1e-9:
        return 0.0
    cv = std_rms / mean_rms
    stability = 1.0 - cv
    return float(np.clip(stability, 0.0, 1.0))


def extract_features(
    waveform: np.ndarray,
    sample_rate: int,
    speech_ratio: float,
    is_speech: np.ndarray | None = None,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> DSPFeatures:
    """Extract the full deterministic DSP feature set for one processed
    waveform. `speech_ratio` and `is_speech` are typically passed through
    from aura.ingest.audio preprocessing to avoid recomputing VAD twice,
    but `is_speech` is optional -- estimate_snr_db will recompute a local
    mask if not given.
    """
    frame_length = max(1, int(sample_rate * frame_ms / 1000))
    hop_length = max(1, int(sample_rate * hop_ms / 1000))

    if len(waveform) == 0:
        # Degenerate empty-audio guard: return a well-formed zero-valued
        # feature record rather than crashing the pipeline on one bad file.
        return DSPFeatures(
            rms=0.0, zero_crossing_rate=0.0, spectral_centroid=0.0,
            spectral_bandwidth=0.0, spectral_rolloff=0.0,
            mfcc_mean=[0.0] * N_MFCC, mfcc_std=[0.0] * N_MFCC,
            clipping_ratio=0.0, speech_ratio=0.0, snr_db=0.0, rms_stability=0.0,
        )

    rms_val = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2) + 1e-12))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
        waveform, frame_length=frame_length, hop_length=hop_length
    )))

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=waveform, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )))
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(
        y=waveform, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(
        y=waveform, sr=sample_rate, n_fft=frame_length, hop_length=hop_length
    )))

    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sample_rate, n_mfcc=N_MFCC,
        n_fft=frame_length, hop_length=hop_length,
    )
    mfcc_mean = mfcc.mean(axis=1).astype(float).tolist()
    mfcc_std = mfcc.std(axis=1).astype(float).tolist()

    clipping_ratio = compute_clipping_ratio(waveform)
    snr_db = estimate_snr_db(
        waveform, sample_rate, is_speech, frame_ms=frame_ms, hop_ms=hop_ms
    )

    frame_rms_arr = _frame_rms(waveform, frame_length, hop_length)
    rms_stability = compute_rms_stability(frame_rms_arr)

    return DSPFeatures(
        rms=rms_val,
        zero_crossing_rate=zcr,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
        clipping_ratio=clipping_ratio,
        speech_ratio=float(speech_ratio),
        snr_db=snr_db,
        rms_stability=rms_stability,
    )
