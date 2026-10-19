"""Regressions for the production-path failures found during the audit."""
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from aura.config import AuraConfig, PathsConfig
from aura.ingest.audio import energy_based_vad, process_audio, trim_to_speech
from aura.ingest.dataset import FilteredSpeechCommands, SampleRecord
from aura.scoring.features import extract_features
from aura.scoring.quality import compute_quality_score, fit_quality_norm_stats
from aura.selection.combined import CombinedSelector
from aura.selection.diversity import DiversitySelector
from aura.selection.quality import QualitySelector
from aura.selection.random import RandomSelector
from aura.selection.utils import allocate_budget_per_class
from tests.test_encoder import _build_local_encoder


def test_production_style_encoder_is_invariant_to_companions_and_order():
    encoder = _build_local_encoder()
    encoder.feature_extractor.return_attention_mask = False
    rng = np.random.default_rng(5)
    waves = [rng.normal(size=n).astype(np.float32) for n in [3200, 8000, 3200]]
    individual = np.stack([encoder.encode_one(w, 16000) for w in waves])
    encoder.batch_size = 3
    mixed = encoder.encode_batch(waves, 16000)
    reordered = encoder.encode_batch(waves[::-1], 16000)[::-1]
    np.testing.assert_allclose(individual, mixed, atol=1e-5)
    np.testing.assert_allclose(mixed, reordered, atol=1e-5)


def test_silence_has_no_activity_or_stability_and_zero_quality():
    for n in [100, 16000]:
        wave = np.zeros(n, dtype=np.float32)
        assert not energy_based_vad(wave, 16000).any()
        feats = extract_features(wave, 16000, 0)
        assert feats.rms_stability == 0
        assert compute_quality_score(feats, fit_quality_norm_stats([feats])).quality_score == 0


def test_activity_tracks_occupancy_instead_of_fixed_median():
    ratios = []
    for fraction in [0.2, 0.6]:
        n = int(16000 * fraction)
        wave = np.zeros(16000, dtype=np.float32)
        wave[:n] = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / 16000)
        ratios.append(energy_based_vad(wave, 16000).mean())
    assert ratios[1] - ratios[0] > 0.3


def test_source_clipping_survives_normalization_and_trim(tmp_path):
    wave = np.zeros(16000, dtype=np.float32)
    wave[4000:8000] = 0.2 * np.sin(np.arange(4000) / 10)
    path = tmp_path / "quiet.wav"
    sf.write(path, wave, 16000, subtype="FLOAT")
    audio = process_audio(str(path))
    assert np.max(np.abs(audio.waveform)) == pytest.approx(1)
    features = extract_features(audio.source_waveform, 16000, audio.speech_ratio, audio.source_activity)
    assert features.clipping_ratio == 0
    clipped = np.clip(wave * 20, -1, 1)
    assert extract_features(clipped, 16000, audio.speech_ratio).clipping_ratio > 0.1


def test_trim_retains_complete_last_active_frame():
    wave = np.ones(16000, dtype=np.float32)
    mask = np.zeros(98, dtype=bool)
    mask[10] = True
    trimmed = trim_to_speech(wave, 16000, mask, padding_ms=0)
    assert len(trimmed) == 400


def pool():
    return pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["x", "x", "y", "y"],
                         "split": ["train"] * 4, "quality_score": [0.1, 0.9, 0.2, 0.8]})


@pytest.mark.parametrize("selector", [RandomSelector(), QualitySelector(), DiversitySelector(), CombinedSelector()])
def test_selection_capacity_determinism_and_train_boundary(selector):
    meta = pool()
    embeddings = dict(zip(meta.id, np.eye(4, dtype=np.float32)))
    first = selector.select(meta, embeddings, 3, 0)
    assert first.selected_ids == selector.select(meta, embeddings, 3, 0).selected_ids
    assert len(first.selected_ids) == len(set(first.selected_ids)) == 3
    assert sorted(first.class_counts.values()) == [1, 2]
    assert selector.select(meta, embeddings, 10, 0).actual_budget == 4
    meta.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="training"):
        selector.select(meta, embeddings, 3, 0)


def test_budget_redistribution_and_invalid_inputs():
    assert allocate_budget_per_class(["x", "y", "z"], 8, {"x": 1, "y": 5, "z": 5}) == {"x": 1, "y": 4, "z": 3}
    for budget in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            RandomSelector().select(pool(), {}, budget, 0)
    with pytest.raises(ValueError, match="unique"):
        RandomSelector().select(pd.concat([pool(), pool()]), {}, 2, 0)
    with pytest.raises(ValueError, match="unit-normalized"):
        DiversitySelector().select(pool(), {sid: np.ones(4) for sid in pool().id}, 2, 0)


def test_combined_quality_endpoint():
    meta = pool()
    embeddings = dict(zip(meta.id, np.eye(4, dtype=np.float32)))
    assert CombinedSelector(1).select(meta, embeddings, 2, 0).selected_ids == ["b", "d"]


def test_dataset_limit_fills_remainder_and_short_classes():
    ds = object.__new__(FilteredSpeechCommands)
    ds._records = [SampleRecord(f"{c}{i}", "unused", c, "speaker", i, "train")
                   for c in ["a", "b", "c"] for i in range(10 if c != "a" else 1)]
    assert len(ds.records(8)) == 8
    assert len(ds.records(50)) == 21
    with pytest.raises(ValueError):
        ds.records(0)


def test_evaluation_requires_matching_persisted_train_stats(tmp_path):
    from scripts.prepare_data import load_training_stats, quality_signature
    from aura.cache.store import file_content_hash
    config = AuraConfig(paths=PathsConfig(metadata_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="missing"):
        load_training_stats(config)
    train = tmp_path / "train.parquet"
    train.write_bytes(b"training artifact")
    payload = {"signature": quality_signature(config), "train_metadata_hash": file_content_hash(train),
               "stats": {"snr_min": 0, "snr_max": 10, "rms_stability_min": 0, "rms_stability_max": 1}}
    (tmp_path / "quality_norm_stats.json").write_text(json.dumps(payload))
    assert load_training_stats(config).snr_max == 10
    train.write_bytes(b"different pool")
    with pytest.raises(ValueError, match="match"):
        load_training_stats(config)


def test_process_split_rejects_missing_stats_before_computing():
    from scripts.prepare_data import process_split
    with pytest.raises(ValueError, match="training quality"):
        process_split([], "test", AuraConfig(), None, None, None, None)
