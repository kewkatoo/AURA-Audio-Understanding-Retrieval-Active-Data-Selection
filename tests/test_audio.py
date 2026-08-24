from __future__ import annotations

import numpy as np

from aura.ingest.audio import (
    load_audio, to_mono, verify_and_resample, amplitude_normalize,
    energy_based_vad, trim_to_speech, process_audio,
)


def test_load_audio_returns_waveform_and_sample_rate(tmp_wav_file):
    path = tmp_wav_file()
    waveform, sr = load_audio(path)
    assert sr == 16000
    assert waveform.ndim == 1
    assert len(waveform) > 0


def test_to_mono_passthrough_for_1d():
    wav = np.random.randn(1000).astype(np.float32)
    result = to_mono(wav)
    assert np.array_equal(result, wav)


def test_to_mono_averages_channels():
    wav = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    result = to_mono(wav)
    assert result.shape == (100,)
    assert np.allclose(result, 0.5)


def test_verify_and_resample_noop_when_rate_matches():
    wav = np.random.randn(1000).astype(np.float32)
    out_wav, out_sr = verify_and_resample(wav, 16000, 16000)
    assert out_sr == 16000
    assert np.array_equal(out_wav, wav)


def test_verify_and_resample_changes_rate_when_mismatched():
    wav = np.random.randn(1000).astype(np.float32)
    out_wav, out_sr = verify_and_resample(wav, 8000, 16000)
    assert out_sr == 16000
    assert len(out_wav) != len(wav)


def test_amplitude_normalize_peak_is_one():
    wav = (0.3 * np.sin(np.linspace(0, 10, 1000))).astype(np.float32)
    normalized = amplitude_normalize(wav)
    assert np.isclose(np.max(np.abs(normalized)), 1.0, atol=1e-5)


def test_amplitude_normalize_handles_silence():
    wav = np.zeros(1000, dtype=np.float32)
    normalized = amplitude_normalize(wav)
    assert np.allclose(normalized, 0.0)


def test_energy_based_vad_returns_boolean_array():
    wav = np.concatenate([
        np.zeros(4000, dtype=np.float32),
        (0.8 * np.sin(np.linspace(0, 50, 8000))).astype(np.float32),
        np.zeros(4000, dtype=np.float32),
    ])
    is_speech = energy_based_vad(wav, 16000)
    assert is_speech.dtype == bool
    assert is_speech.any()
    assert not is_speech.all()  # should not classify pure silence as speech


def test_trim_to_speech_shrinks_silence_padded_clip():
    wav = np.concatenate([
        np.zeros(8000, dtype=np.float32),
        (0.8 * np.sin(np.linspace(0, 50, 8000))).astype(np.float32),
        np.zeros(8000, dtype=np.float32),
    ])
    is_speech = energy_based_vad(wav, 16000)
    trimmed = trim_to_speech(wav, 16000, is_speech)
    assert len(trimmed) < len(wav)
    assert len(trimmed) > 0


def test_trim_to_speech_returns_original_if_no_speech_detected():
    wav = np.zeros(1000, dtype=np.float32)
    is_speech = np.zeros(10, dtype=bool)
    trimmed = trim_to_speech(wav, 16000, is_speech)
    assert np.array_equal(trimmed, wav)


def test_process_audio_end_to_end(tmp_wav_file):
    path = tmp_wav_file(silence_s=0.3, tone_s=0.4)
    result = process_audio(path, target_sample_rate=16000)
    assert result.sample_rate == 16000
    assert result.original_duration_s > result.processed_duration_s  # trimmed
    assert 0.0 <= result.speech_ratio <= 1.0
    assert result.waveform.ndim == 1
    assert np.max(np.abs(result.waveform)) <= 1.0 + 1e-5


def test_process_audio_never_mutates_source_file(tmp_wav_file):
    path = tmp_wav_file()
    before, _ = load_audio(path)
    process_audio(path, target_sample_rate=16000)
    after, _ = load_audio(path)
    assert np.array_equal(before, after)


def test_process_audio_deterministic(tmp_wav_file):
    path = tmp_wav_file()
    r1 = process_audio(path, target_sample_rate=16000)
    r2 = process_audio(path, target_sample_rate=16000)
    assert np.array_equal(r1.waveform, r2.waveform)
    assert r1.speech_ratio == r2.speech_ratio
