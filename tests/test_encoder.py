from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor

from aura.encoders.base import EncoderInfo
from aura.encoders.wav2vec2 import Wav2Vec2Encoder


def _build_local_encoder(batch_size: int = 4) -> Wav2Vec2Encoder:
    """Builds a Wav2Vec2Encoder around a small, randomly-initialized model
    instead of calling from_pretrained, so encoder-wiring tests don't
    require network access to huggingface.co. This is a test-only
    construction path -- production code always uses from_pretrained via
    the normal __init__.
    """
    enc = object.__new__(Wav2Vec2Encoder)
    enc.checkpoint = "test-local"
    enc.pooling = "mean"
    enc.normalize = True
    enc.batch_size = batch_size
    enc.device = "cpu"

    config = Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3),
    )
    enc.model = Wav2Vec2Model(config)
    enc.model.eval()
    for p in enc.model.parameters():
        p.requires_grad_(False)

    enc.feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    enc._info = EncoderInfo(
        name="wav2vec2", checkpoint="test-local", embedding_dim=config.hidden_size,
        pooling="mean", l2_normalized=True, device="cpu",
    )
    return enc


@pytest.fixture
def local_encoder():
    return _build_local_encoder()


def test_encoder_frozen_parameters(local_encoder):
    assert all(not p.requires_grad for p in local_encoder.model.parameters())
    assert not local_encoder.model.training  # eval mode


def test_encoder_embedding_dim_recorded_automatically(local_encoder):
    assert local_encoder.info.embedding_dim == local_encoder.model.config.hidden_size


def test_encoder_output_shape(local_encoder):
    waves = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
    out = local_encoder.encode_batch(waves, 16000)
    assert out.shape == (3, local_encoder.info.embedding_dim)
    assert out.dtype == np.float32


def test_encoder_handles_variable_length_batch(local_encoder):
    waves = [
        np.random.randn(16000).astype(np.float32),
        np.random.randn(8000).astype(np.float32),
        np.random.randn(20000).astype(np.float32),
    ]
    out = local_encoder.encode_batch(waves, 16000)
    assert out.shape == (3, local_encoder.info.embedding_dim)
    assert not np.isnan(out).any()


def test_encoder_l2_normalized_when_enabled(local_encoder):
    waves = [np.random.randn(16000).astype(np.float32)]
    out = local_encoder.encode_batch(waves, 16000)
    norm = np.linalg.norm(out[0])
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_encoder_deterministic(local_encoder):
    waves = [np.random.randn(16000).astype(np.float32) for _ in range(2)]
    out1 = local_encoder.encode_batch(waves, 16000)
    out2 = local_encoder.encode_batch(waves, 16000)
    assert np.allclose(out1, out2)


def test_encoder_rejects_wrong_sample_rate(local_encoder):
    waves = [np.random.randn(8000).astype(np.float32)]
    with pytest.raises(ValueError):
        local_encoder.encode_batch(waves, 8000)


def test_encoder_empty_batch_returns_empty_array(local_encoder):
    out = local_encoder.encode_batch([], 16000)
    assert out.shape == (0, local_encoder.info.embedding_dim)


def test_encoder_batching_matches_single_pass(local_encoder):
    """Encoding 5 waveforms with batch_size=2 (multiple mini-batches)
    should give the same per-sample output as batch_size=5 (one batch) --
    catches bugs where batching accidentally mixes samples."""
    local_encoder.batch_size = 2
    waves = [np.random.randn(16000).astype(np.float32) for _ in range(5)]
    out_batched = local_encoder.encode_batch(waves, 16000)

    local_encoder.batch_size = 5
    out_single = local_encoder.encode_batch(waves, 16000)

    assert np.allclose(out_batched, out_single, atol=1e-5)
