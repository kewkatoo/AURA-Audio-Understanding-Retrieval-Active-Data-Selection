from __future__ import annotations

import numpy as np

from aura.ingest.audio import process_audio, energy_based_vad
from aura.scoring.features import extract_features, N_MFCC, compute_clipping_ratio, compute_rms_stability


def test_extract_features_shapes(tmp_wav_file):
    path = tmp_wav_file()
    pa = process_audio(path, target_sample_rate=16000)
    is_speech = energy_based_vad(pa.waveform, pa.sample_rate)
    feats = extract_features(pa.waveform, pa.sample_rate, pa.speech_ratio, is_speech=is_speech)

    assert len(feats.mfcc_mean) == N_MFCC
    assert len(feats.mfcc_std) == N_MFCC
    assert isinstance(feats.rms, float)
    assert isinstance(feats.snr_db, float)


def test_extract_features_serializable(tmp_wav_file):
    path = tmp_wav_file()
    pa = process_audio(path, target_sample_rate=16000)
    feats = extract_features(pa.waveform, pa.sample_rate, pa.speech_ratio)
    d = feats.to_dict()
    import json
    json.dumps(d)  # must not raise


def test_extract_features_deterministic(tmp_wav_file):
    path = tmp_wav_file()
    pa = process_audio(path, target_sample_rate=16000)
    f1 = extract_features(pa.waveform, pa.sample_rate, pa.speech_ratio)
    f2 = extract_features(pa.waveform, pa.sample_rate, pa.speech_ratio)
    assert f1.to_dict() == f2.to_dict()


def test_extract_features_empty_waveform_does_not_crash():
    feats = extract_features(np.array([], dtype=np.float32), 16000, speech_ratio=0.0)
    assert feats.rms == 0.0
    assert len(feats.mfcc_mean) == N_MFCC


def test_compute_clipping_ratio_detects_full_scale():
    wav = np.ones(1000, dtype=np.float32)
    ratio = compute_clipping_ratio(wav, threshold=0.99)
    assert ratio == 1.0


def test_compute_clipping_ratio_zero_for_quiet_signal():
    wav = (0.1 * np.sin(np.linspace(0, 10, 1000))).astype(np.float32)
    ratio = compute_clipping_ratio(wav, threshold=0.99)
    assert ratio == 0.0


def test_compute_rms_stability_bounds():
    frame_rms = np.array([0.5, 0.5, 0.5, 0.5])
    stability = compute_rms_stability(frame_rms)
    assert stability == 1.0  # zero variance -> perfectly stable

    frame_rms_noisy = np.array([0.1, 0.9, 0.2, 0.8])
    stability_noisy = compute_rms_stability(frame_rms_noisy)
    assert 0.0 <= stability_noisy <= 1.0
    assert stability_noisy < stability
