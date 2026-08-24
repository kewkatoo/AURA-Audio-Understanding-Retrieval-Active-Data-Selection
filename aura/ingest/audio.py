"""
Audio loading and preprocessing.

Pipeline per sample: load -> verify/resample sample rate -> mono ->
amplitude normalize -> energy-based VAD -> optional trim.

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
    peak = np.max(np.abs(waveform))
    if peak < eps:
        return waveform.astype(np.float32)
    return (waveform / peak).astype(np.float32)


def energy_based_vad(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_percentile: float = 50.0,
) -> np.ndarray:
    """Simple energy-threshold VAD. Returns a boolean array, one entry per
    frame, True where the frame is classified as speech.

    Method: compute per-frame RMS energy, threshold at the given percentile
    of the *utterance's own* frame-energy distribution. This is a
    deliberately simple, defensible baseline VAD -- not a learned model --
    consistent with "prototype acoustic quality score" framing elsewhere
    in the pipeline. It is stated as such, not sold as robust VAD.
    """
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop_len = max(1, int(sample_rate * hop_ms / 1000))

    if len(waveform) < frame_len:
        # Utterance shorter than one frame: treat the whole thing as a
        # single frame.
        rms = np.sqrt(np.mean(waveform.astype(np.float64) ** 2) + 1e-12)
        return np.array([rms > 0], dtype=bool)

    frames = librosa.util.frame(
        waveform, frame_length=frame_len, hop_length=hop_len
    ).astype(np.float64)
    frame_rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-12)

    threshold = np.percentile(frame_rms, energy_percentile)
    # Guard against degenerate all-equal-energy audio (e.g. pure silence):
    # if threshold==max, nothing would ever be "above" -> treat none as
    # speech, which is the correct behavior for silent clips.
    is_speech = frame_rms > threshold
    return is_speech


def trim_to_speech(
    waveform: np.ndarray,
    sample_rate: int,
    is_speech: np.ndarray,
    hop_ms: float = 10.0,
    padding_ms: float = 50.0,
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
        len(waveform), speech_frame_indices[-1] * hop_len + padding_samples
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
    vad_energy_percentile: float = 50.0,
) -> ProcessedAudio:
    """End-to-end preprocessing for one audio file. Returns a
    ProcessedAudio record; does not write anything to disk."""
    waveform, sample_rate = load_audio(path)
    original_duration_s = len(waveform) / sample_rate if waveform.ndim == 1 else waveform.shape[0] / sample_rate

    if do_mono:
        waveform = to_mono(waveform)

    waveform, sample_rate = verify_and_resample(waveform, sample_rate, target_sample_rate)

    if do_normalize:
        waveform = amplitude_normalize(waveform)

    speech_ratio = 1.0
    trimmed = False
    if do_vad:
        is_speech = energy_based_vad(
            waveform,
            sample_rate,
            frame_ms=vad_frame_ms,
            hop_ms=vad_hop_ms,
            energy_percentile=vad_energy_percentile,
        )
        speech_ratio = float(np.mean(is_speech)) if len(is_speech) > 0 else 0.0

        if do_trim:
            trimmed_waveform = trim_to_speech(
                waveform, sample_rate, is_speech, hop_ms=vad_hop_ms
            )
            if len(trimmed_waveform) != len(waveform):
                trimmed = True
            waveform = trimmed_waveform

    processed_duration_s = len(waveform) / sample_rate

    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_duration_s=float(original_duration_s),
        processed_duration_s=float(processed_duration_s),
        speech_ratio=speech_ratio,
        trimmed=trimmed,
    )
