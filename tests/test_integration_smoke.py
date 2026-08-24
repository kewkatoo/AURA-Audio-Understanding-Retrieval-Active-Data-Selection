"""
Integration smoke tests.

`test_pipeline_smoke_20_samples` exercises the full Phase 1 pipeline
(preprocess -> DSP features -> quality score with train-pool normalization
-> Wav2Vec2 embeddings -> metadata + embedding persistence) end-to-end on
20 synthetic samples across the 10 target classes, using
scripts.prepare_data.process_split directly with hand-built SampleRecords.

This deliberately does NOT go through aura.ingest.dataset.FilteredSpeechCommands
/ torchaudio's SPEECHCOMMANDS downloader, because that requires network
access to download.tensorflow.org, which is unavailable in sandboxed CI/dev
environments (this repo's own dev sandbox included). `test_real_dataset_loader`
below covers that path but is skipped unless AURA_RUN_NETWORK_TESTS=1 is set,
so it still runs in environments with real internet access (e.g. the
person's own machine) via `scripts/prepare_data.py` directly rather than duplicating
its logic here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.config import AuraConfig, PathsConfig
from aura.cache.store import CacheStore
from aura.ingest.dataset import SampleRecord
from aura.encoders.wav2vec2 import Wav2Vec2Encoder
from aura.ingest.metadata import save_metadata_parquet, save_embedding_matrix, load_metadata_parquet, load_embedding_matrix


CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _build_synthetic_pool(root: Path, n_samples: int = 20) -> list[SampleRecord]:
    """Builds n_samples synthetic wav files spread across the 10 classes,
    with varying amplitude/frequency/duration so quality/DSP features have
    nonzero variance (a pool of identical clips would make several tests
    degenerate, e.g. quality normalization)."""
    root.mkdir(parents=True, exist_ok=True)
    records = []
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        label = CLASSES[i % len(CLASSES)]
        speaker_id = f"speaker{i % 4}"
        amplitude = float(rng.uniform(0.2, 0.9))
        freq = float(rng.uniform(200, 600))
        duration_s = float(rng.uniform(0.5, 1.0))
        sr = 16000
        t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
        tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        pad = np.zeros(int(0.1 * sr), dtype=np.float32)
        wav = np.concatenate([pad, tone, pad])

        filename = f"{speaker_id}_nohash_{i}.wav"
        path = root / filename
        sf.write(str(path), wav, sr)

        records.append(SampleRecord(
            sample_id=f"{label}_{speaker_id}_{i}",
            audio_path=str(path),
            label=label,
            speaker_id=speaker_id,
            utterance_number=i,
            split="train",
        ))
    return records


def _local_wav2vec2_encoder(batch_size: int = 4) -> Wav2Vec2Encoder:
    from transformers import Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2FeatureExtractor
    from aura.encoders.base import EncoderInfo

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


def test_pipeline_smoke_20_samples(tmp_path):
    from scripts.prepare_data import process_split

    audio_root = tmp_path / "synthetic_audio"
    records = _build_synthetic_pool(audio_root, n_samples=20)
    assert len(records) == 20
    assert len(set(r.label for r in records)) == len(CLASSES)  # all classes present

    config = AuraConfig(
        paths=PathsConfig(
            metadata_dir=str(tmp_path / "metadata"),
            embeddings_dir=str(tmp_path / "embeddings"),
        ),
    )

    feature_cache = CacheStore(tmp_path / "cache" / "features")
    embedding_cache = CacheStore(tmp_path / "cache" / "embeddings")
    encoder = _local_wav2vec2_encoder()

    rows, embedding_matrix, sample_ids, norm_stats = process_split(
        records, split="train", config=config,
        feature_cache=feature_cache, embedding_cache=embedding_cache,
        encoder=encoder, norm_stats=None,
    )

    assert len(rows) == 20
    assert embedding_matrix.shape == (20, encoder.info.embedding_dim)
    assert len(sample_ids) == 20
    assert not np.isnan(embedding_matrix).any()

    for row in rows:
        assert 0.0 <= row["quality_score"] <= 1.0
        assert row["label"] in CLASSES

    metadata_path = tmp_path / "metadata" / "train.parquet"
    embedding_matrix_path = tmp_path / "embeddings" / "train.npy"
    embedding_index_path = tmp_path / "embeddings" / "train_index.parquet"
    save_metadata_parquet(rows, metadata_path)
    save_embedding_matrix(embedding_matrix, sample_ids, embedding_matrix_path, embedding_index_path)

    # Rerun through the SAME cache: every feature/embedding should now be a
    # cache hit, verifying the caching contract end-to-end (not recomputed).
    rows2, embedding_matrix2, sample_ids2, _ = process_split(
        records, split="train", config=config,
        feature_cache=feature_cache, embedding_cache=embedding_cache,
        encoder=encoder, norm_stats=norm_stats,
    )
    assert np.allclose(embedding_matrix, embedding_matrix2)
    assert [r["quality_score"] for r in rows] == [r["quality_score"] for r in rows2]

    # Reload persisted outputs and sanity check.
    df = load_metadata_parquet(metadata_path)
    matrix, index_df = load_embedding_matrix(embedding_matrix_path, embedding_index_path)
    assert len(df) == 20
    assert matrix.shape[0] == 20
    assert set(df["id"]) == set(index_df["sample_id"])

    feature_cache.close()
    embedding_cache.close()


@pytest.mark.skipif(
    os.environ.get("AURA_RUN_NETWORK_TESTS") != "1",
    reason=(
        "Requires network access to download.tensorflow.org to fetch the "
        "real Speech Commands archive. Set AURA_RUN_NETWORK_TESTS=1 to run "
        "this on a machine with internet access."
    ),
)
def test_real_dataset_loader(tmp_path):
    from aura.ingest.dataset import FilteredSpeechCommands

    ds = FilteredSpeechCommands(
        root=str(tmp_path / "raw"), classes=CLASSES, split="train", download=True,
    )
    records = ds.records(sample_limit=20)
    assert len(records) <= 20
    assert all(r.label in CLASSES for r in records)
    assert all(Path(r.audio_path).exists() for r in records)
