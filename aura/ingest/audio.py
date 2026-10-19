"""
Audio loading and preprocessing.

Pipeline per sample: load -> mono -> verify/resample -> preserve source
diagnostics -> energy activity -> optional trim -> amplitude normalize.

The original (untrimmed, unnormalized-in-place) waveform is never mutated;
functions return new arrays. Nothing in this module writes to disk or
touches the cache -- caching happens one layer up (scripts/prepare_data.py)
since what's expensive and cacheable is the *downstream* features and
embeddings, not this preprocessing itself (which is cheap and needs to run
fresh at Wav2Vec2-input time, and re-running it is what verifies a cache
key's audio-identity hash).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa


@dataclass
class ProcessedAudio:
    waveform: np.ndarray          # float32, mono, shape (n_samples,)
    sample_rate: int
    original_duration_s: float
    processed_duration_s: float   # duration after optional trim
    speech_ratio: float           # fraction of frames classified as speech
    trimmed: bool
    source_waveform: np.ndarray | None = None  # mono/resampled, before gain or trimming
    source_activity: np.ndarray | None = None


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file. Returns (waveform, sample_rate). Does not mutate
    or resave anything -- the file on disk is the source of truth."""
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return waveform, sample_rate


def to_mono(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 1:
        return waveform
    # sf.read with always_2d=False already gives 1D for mono files; this
    # handles the case where the source file is genuinely multi-channel.
    return np.mean(waveform, axis=1).astype(np.float32)


def verify_and_resample(
    waveform: np.ndarray, sample_rate: int, target_sample_rate: int
) -> tuple[np.ndarray, int]:
    """Speech Commands is natively 16kHz, so for that dataset this is a
    no-op. Resampling only triggers if a source file genuinely differs
    from the configured target rate, per Phase 1 spec."""
    if sample_rate == target_sample_rate:
        return waveform, sample_rate
    resampled = librosa.resample(
        waveform.astype(np.float32), orig_sr=sample_rate, target_sr=target_sample_rate
    )
    return resampled.astype(np.float32), target_sample_rate


def amplitude_normalize(waveform: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Peak-normalize to [-1, 1]. No-op (safe) on silent/near-zero audio."""
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak < eps:
        return waveform.astype(np.float32)
    return (waveform / peak).astype(np.float32)


def energy_based_vad(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_percentile: float = 20.0,
    noise_margin_db: float = 6.0,
    min_rms: float = 0.001,
) -> np.ndarray:
    """Simple energy-threshold VAD. Returns a boolean array, one entry per
    frame, True where the frame is classified as speech.

    A low-percentile noise-floor estimate plus a dB margin, bounded below
    by an absolute RMS floor. This is an energy activity heuristic, not a
    validated speech detector. Apply before peak normalization.
    """
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))

    if not 0 <= energy_percentile <= 100 or noise_margin_db < 0 or min_rms <= 0:
        raise ValueError("Invalid activity threshold parameters")
    if len(waveform) == 0:
        return np.zeros(0, dtype=bool)
    if len(waveform) < frame_len:
        # Utterance shorter than one frame: treat the whole thing as a
        # single frame.
        rms = np.sqrt(np.mean(waveform.astype(np.float64) ** 2))
        return np.array([rms > min_rms], dtype=bool)

    frames = librosa.util.frame(
        waveform, frame_length=frame_len, hop_length=hop_len
    ).astype(np.float64)
    frame_rms = np.sqrt(np.mean(frames**2, axis=0))

    threshold = max(min_rms, np.percentile(frame_rms, energy_percentile) * 10 ** (noise_margin_db / 20))
    is_speech = frame_rms > threshold
    return is_speech


def trim_to_speech(
    waveform: np.ndarray,
    sample_rate: int,
    is_speech: np.ndarray,
    hop_ms: float = 10.0,
    padding_ms: float = 50.0,
    frame_ms: float = 25.0,
) -> np.ndarray:
    """Trim waveform to the [first, last] speech-active frame, with a small
    padding margin. If no frame is classified as speech, returns the
    original waveform unchanged (never returns an empty array)."""
    if not np.any(is_speech):
        return waveform

    hop_len = max(1, int(sample_rate * hop_ms / 1000))
    padding_samples = int(sample_rate * padding_ms / 1000)

    speech_frame_indices = np.where(is_speech)[0]
    start_sample = max(0, speech_frame_indices[0] * hop_len - padding_samples)
    end_sample = min(
        len(waveform), speech_frame_indices[-1] * hop_len
        + int(sample_rate * frame_ms / 1000) + padding_samples
    )
    if end_sample <= start_sample:
        return waveform
    return waveform[start_sample:end_sample]


def process_audio(
    path: str,
    target_sample_rate: int = 16000,
    do_mono: bool = True,
    do_normalize: bool = True,
    do_vad: bool = True,
    do_trim: bool = True,
    vad_frame_ms: float = 25.0,
    vad_hop_ms: float = 10.0,
    vad_energy_percentile: float = 20.0,
    vad_noise_margin_db: float = 6.0,
    vad_min_rms: float = 0.001,
) -> ProcessedAudio:
    """End-to-end preprocessing for one audio file. Returns a
    ProcessedAudio record; does not write anything to disk."""
    waveform, sample_rate = load_audio(path)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Audio must be nonempty and finite: {path}")
    original_duration_s = len(waveform) / sample_rate if waveform.ndim == 1 else waveform.shape[0] / sample_rate

    if do_mono:
        waveform = to_mono(waveform)
    if waveform.ndim != 1:
        raise ValueError("The audio pipeline requires mono input; enable mono conversion")

    waveform, sample_rate = verify_and_resample(waveform, sample_rate, target_sample_rate)

    source_waveform = waveform.copy()

    speech_ratio = 1.0
    trimmed = False
    if do_vad:
        is_speech = energy_based_vad(
            waveform,
            sample_rate,
            frame_ms=vad_frame_ms,
            hop_ms=vad_hop_ms,
            energy_percentile=vad_energy_percentile,
            noise_margin_db=vad_noise_margin_db,
            min_rms=vad_min_rms,
        )
        speech_ratio = float(np.mean(is_speech)) if len(is_speech) > 0 else 0.0

        if do_trim:
            trimmed_waveform = trim_to_speech(
                waveform, sample_rate, is_speech, hop_ms=vad_hop_ms, frame_ms=vad_frame_ms
            )
            if len(trimmed_waveform) != len(waveform):
                trimmed = True
            waveform = trimmed_waveform

    if do_normalize:
        waveform = amplitude_normalize(waveform)
    processed_duration_s = len(waveform) / sample_rate

    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_duration_s=float(original_duration_s),
        processed_duration_s=float(processed_duration_s),
        speech_ratio=speech_ratio,
        trimmed=trimmed,
        source_waveform=source_waveform,
        source_activity=is_speech if do_vad else None,
    )
