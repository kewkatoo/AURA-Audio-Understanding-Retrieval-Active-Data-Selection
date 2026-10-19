"""Cheap source-audio descriptors for selection before neural encoding."""
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

from aura.ingest.audio import load_audio, to_mono, verify_and_resample


def source_mfcc(path, sample_rate=16000):
    waveform, sr = load_audio(str(path))
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("MFCC input must be nonempty and finite")
    waveform, sr = verify_and_resample(to_mono(waveform), sr, sample_rate)
    coefficients = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13,
        n_fft=int(sample_rate * .025), hop_length=int(sample_rate * .010))
    return np.concatenate([coefficients.mean(axis=1), coefficients.std(axis=1)]).astype(np.float64)


def normalized_descriptors(training_features):
    """Fit scaling on the candidate pool only; return scaler and unit vectors."""
    x = np.asarray(training_features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 26 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite training descriptors with 26 columns")
    scaler = StandardScaler().fit(x)
    projected = scaler.transform(x)
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("A standardized descriptor has zero norm")
    return scaler, (projected / norms).astype(np.float32)
