#!/usr/bin/env python
"""
End-to-end Phase 1 pipeline:

  dataset -> filter to classes -> preprocess audio -> DSP features
  -> quality score -> Wav2Vec2 embeddings -> persisted metadata + embeddings

Safe to rerun: every expensive step (DSP features, embeddings) is cached
by content hash + config fingerprint, so a rerun with unchanged inputs and
config only reads from cache. Changing the encoder checkpoint or
preprocessing config automatically invalidates the relevant cache entries
without touching the other ones.

Usage:
    python scripts/prepare_data.py --config configs/experiment.yaml
    python scripts/prepare_data.py --config configs/experiment.yaml --splits train validation
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.config import AuraConfig
from aura.cache.store import CacheStore, file_content_hash, config_fingerprint, make_cache_key
from aura.ingest.dataset import FilteredSpeechCommands, SampleRecord
from aura.ingest.audio import process_audio, energy_based_vad
from aura.scoring.features import extract_features, DSPFeatures
from aura.scoring.quality import fit_quality_norm_stats, compute_quality_score, QualityNormStats
from aura.encoders.wav2vec2 import Wav2Vec2Encoder
from aura.ingest.metadata import (
    build_metadata_row, save_metadata_parquet,
    save_embedding_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def audio_preprocess_config_dict(config: AuraConfig) -> dict:
    return {
        "target_sample_rate": config.audio.sample_rate,
        "mono": config.audio.mono,
        "normalize": config.audio.normalize,
        "vad": config.audio.vad,
        "vad_frame_ms": config.audio.vad_frame_ms,
        "vad_hop_ms": config.audio.vad_hop_ms,
        "vad_energy_percentile": config.audio.vad_energy_percentile,
    }


def encoder_config_dict(config: AuraConfig) -> dict:
    return {
        "name": config.encoder.name,
        "checkpoint": config.encoder.checkpoint,
        "pooling": config.encoder.pooling,
        "normalize": config.encoder.normalize,
    }


def process_split(
    records: list[SampleRecord],
    split: str,
    config: AuraConfig,
    feature_cache: CacheStore,
    embedding_cache: CacheStore,
    encoder: Wav2Vec2Encoder,
    norm_stats: QualityNormStats | None,
) -> tuple[list[dict], np.ndarray, list[str], QualityNormStats]:
    """Processes one split. If norm_stats is None (train split), fits it
    from this split's own features; otherwise reuses the passed-in stats
    (validation/test) to avoid leakage."""
    preprocess_cfg = audio_preprocess_config_dict(config)
    preprocess_cfg_hash = config_fingerprint(preprocess_cfg)
    enc_cfg = encoder_config_dict(config)
    enc_cfg_hash = config_fingerprint(enc_cfg)

    all_features: list[DSPFeatures] = []
    all_processed = []  # (record, processed_audio) pairs, needed twice (norm fit + row build)
    cache_hits_features = 0
    cache_misses_features = 0

    t0 = time.time()
    for i, record in enumerate(records):
        audio_hash = file_content_hash(record.audio_path)
        feature_key = make_cache_key(audio_hash, preprocess_cfg_hash, namespace="features")

        cached = feature_cache.get(feature_key)
        if cached is not None:
            processed_audio, features = cached
            cache_hits_features += 1
        else:
            processed_audio = process_audio(
                record.audio_path,
                target_sample_rate=preprocess_cfg["target_sample_rate"],
                do_mono=preprocess_cfg["mono"],
                do_normalize=preprocess_cfg["normalize"],
                do_vad=preprocess_cfg["vad"],
                do_trim=preprocess_cfg["vad"],
                vad_frame_ms=preprocess_cfg["vad_frame_ms"],
                vad_hop_ms=preprocess_cfg["vad_hop_ms"],
                vad_energy_percentile=preprocess_cfg["vad_energy_percentile"],
            )
            is_speech = None
            if preprocess_cfg["vad"] and len(processed_audio.waveform) > 0:
                is_speech = energy_based_vad(
                    processed_audio.waveform, processed_audio.sample_rate,
                    frame_ms=preprocess_cfg["vad_frame_ms"],
                    hop_ms=preprocess_cfg["vad_hop_ms"],
                    energy_percentile=preprocess_cfg["vad_energy_percentile"],
                )
            features = extract_features(
                processed_audio.waveform, processed_audio.sample_rate,
                speech_ratio=processed_audio.speech_ratio, is_speech=is_speech,
                frame_ms=preprocess_cfg["vad_frame_ms"], hop_ms=preprocess_cfg["vad_hop_ms"],
            )
            feature_cache.set(feature_key, (processed_audio, features))
            cache_misses_features += 1

        all_features.append(features)
        all_processed.append((record, processed_audio))

        if (i + 1) % 500 == 0:
            logger.info(f"[{split}] features: {i + 1}/{len(records)}")

    logger.info(
        f"[{split}] feature extraction done in {time.time() - t0:.1f}s "
        f"(cache hits={cache_hits_features}, misses={cache_misses_features})"
    )

    if norm_stats is None:
        norm_stats = fit_quality_norm_stats(all_features)
        logger.info(f"[{split}] fit quality norm stats: {norm_stats.to_dict()}")

    # Embeddings: batch through the encoder, cached per-sample.
    embeddings_list: list[np.ndarray] = []
    sample_ids: list[str] = []
    rows: list[dict] = []

    to_encode_waveforms = []
    to_encode_records = []  # (record, processed_audio, features) needing fresh encoding
    cached_embeddings: dict[str, np.ndarray] = {}

    t0 = time.time()
    for record, processed_audio in all_processed:
        audio_hash = file_content_hash(record.audio_path)
        emb_key = make_cache_key(audio_hash, f"{preprocess_cfg_hash}:{enc_cfg_hash}", namespace="embeddings")
        cached_emb = embedding_cache.get(emb_key)
        if cached_emb is not None:
            cached_embeddings[record.sample_id] = cached_emb
        else:
            to_encode_waveforms.append(processed_audio.waveform)
            to_encode_records.append((record, processed_audio, emb_key))

    logger.info(
        f"[{split}] embeddings: {len(cached_embeddings)} cached, "
        f"{len(to_encode_waveforms)} to compute"
    )

    freshly_computed: dict[str, np.ndarray] = {}
    batch_size = config.encoder.batch_size
    for start in range(0, len(to_encode_waveforms), batch_size):
        batch_waveforms = to_encode_waveforms[start:start + batch_size]
        batch_records = to_encode_records[start:start + batch_size]
        batch_embeddings = encoder.encode_batch(batch_waveforms, config.audio.sample_rate)
        for (record, _, emb_key), emb in zip(batch_records, batch_embeddings):
            embedding_cache.set(emb_key, emb)
            freshly_computed[record.sample_id] = emb
        if (start // batch_size + 1) % 20 == 0:
            logger.info(f"[{split}] embeddings: {start + len(batch_waveforms)}/{len(to_encode_waveforms)}")

    logger.info(f"[{split}] embedding computation done in {time.time() - t0:.1f}s")

    dataset_version = config.dataset.name
    for (record, processed_audio), features in zip(all_processed, all_features):
        q_score = compute_quality_score(features, norm_stats)
        emb = cached_embeddings.get(record.sample_id)
        if emb is None:
            emb = freshly_computed.get(record.sample_id)
        if emb is None:
            raise RuntimeError(f"Missing embedding for sample {record.sample_id}")

        embedding_matrix_path = str(Path(config.paths.embeddings_dir) / f"{split}.npy")
        row = build_metadata_row(
            sample_id=record.sample_id,
            audio_path=record.audio_path,
            label=record.label,
            speaker_id=record.speaker_id,
            processed_audio=processed_audio,
            dsp_features=features,
            quality_score=q_score,
            embedding_path=embedding_matrix_path,
            dataset_version=dataset_version,
            encoder_name=config.encoder.name,
            encoder_checkpoint=config.encoder.checkpoint,
            split=split,
        )
        rows.append(row)
        embeddings_list.append(emb)
        sample_ids.append(record.sample_id)

    embedding_matrix = np.stack(embeddings_list, axis=0) if embeddings_list else np.zeros((0, encoder.info.embedding_dim), dtype=np.float32)
    return rows, embedding_matrix, sample_ids, norm_stats


def main():
    parser = argparse.ArgumentParser(description="AURA Phase 1 data preparation pipeline.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--sample-limit-override", type=int, default=None,
        help="Override sample_limit for ALL requested splits (mainly for smoke tests).",
    )
    args = parser.parse_args()

    config = AuraConfig.from_yaml(args.config)
    logger.info(f"Loaded config from {args.config}")

    feature_cache = CacheStore(Path(config.cache.path) / "features")
    embedding_cache = CacheStore(Path(config.cache.path) / "embeddings")

    logger.info(f"Loading encoder: {config.encoder.checkpoint} (device={config.encoder.device})")
    encoder = Wav2Vec2Encoder(
        checkpoint=config.encoder.checkpoint,
        pooling=config.encoder.pooling,
        normalize=config.encoder.normalize,
        device=config.encoder.device,
        batch_size=config.encoder.batch_size,
    )
    logger.info(f"Encoder ready: {encoder.info}")

    sample_limits = {
        "train": config.dataset.sample_limit.train,
        "validation": config.dataset.sample_limit.validation,
        "test": config.dataset.sample_limit.test,
    }
    if args.sample_limit_override is not None:
        sample_limits = {k: args.sample_limit_override for k in sample_limits}

    norm_stats: QualityNormStats | None = None
    # Process train first (if requested) since validation/test quality
    # normalization depends on train's fitted stats. If train isn't in
    # args.splits (e.g. a validation-only rerun), norm_stats stays None and
    # each processed split fits its own stats -- logged as a warning so
    # this doesn't silently happen in a full run.
    ordered_splits = [s for s in ["train", "validation", "test"] if s in args.splits]
    if "train" not in ordered_splits:
        logger.warning(
            "Processing without 'train' split: quality-score normalization "
            "stats will be fit per-split rather than reused from train. "
            "This is fine for isolated smoke tests but NOT for a real run."
        )

    for split in ordered_splits:
        logger.info(f"=== Processing split: {split} ===")
        dataset = FilteredSpeechCommands(
            root=config.dataset.root,
            classes=config.dataset.classes,
            split=split,
            download=True,
        )
        records = dataset.records(sample_limit=sample_limits[split])
        logger.info(f"[{split}] {len(records)} records selected (of {len(dataset)} available)")

        split_norm_stats = norm_stats if split != "train" else None
        rows, embedding_matrix, sample_ids, fitted_stats = process_split(
            records, split, config, feature_cache, embedding_cache, encoder, split_norm_stats
        )
        if split == "train":
            norm_stats = fitted_stats

        metadata_path = Path(config.paths.metadata_dir) / f"{split}.parquet"
        save_metadata_parquet(rows, metadata_path)
        logger.info(f"[{split}] saved metadata: {metadata_path} ({len(rows)} rows)")

        embedding_matrix_path = Path(config.paths.embeddings_dir) / f"{split}.npy"
        embedding_index_path = Path(config.paths.embeddings_dir) / f"{split}_index.parquet"
        save_embedding_matrix(embedding_matrix, sample_ids, embedding_matrix_path, embedding_index_path)
        logger.info(
            f"[{split}] saved embeddings: {embedding_matrix_path} "
            f"shape={embedding_matrix.shape}"
        )

    feature_cache.close()
    embedding_cache.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
