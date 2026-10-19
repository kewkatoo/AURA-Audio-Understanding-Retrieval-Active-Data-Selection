import numpy as np
import pytest
import soundfile as sf
from experiments.acoustic import source_mfcc, normalized_descriptors
from aura.scoring.features import extract_features


def test_cheap_features_match_frozen_source_mfcc(tmp_path):
    sr = 16000
    x = (.2 * np.sin(2 * np.pi * 320 * np.arange(sr) / sr)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, x, sr, subtype="FLOAT")
    actual = source_mfcc(path)
    frozen = extract_features(x, sr, 1)
    np.testing.assert_allclose(actual, frozen.mfcc_mean + frozen.mfcc_std, atol=1e-6)


def test_descriptor_scaling_is_candidate_only_and_unit_norm():
    x = np.random.default_rng(4).normal(size=(20, 26))
    scaler, matrix = normalized_descriptors(x)
    np.testing.assert_allclose(scaler.mean_, x.mean(axis=0))
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="26"):
        normalized_descriptors(x[:, :25])
    with pytest.raises(ValueError, match="zero norm"):
        normalized_descriptors(np.ones((2, 26)))
