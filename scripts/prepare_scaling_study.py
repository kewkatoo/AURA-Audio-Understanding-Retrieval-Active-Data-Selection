#!/usr/bin/env python
"""Prepare the full available corpus with a fresh training-speaker holdout.

Uses frozen v2 preprocessing and caches, processing at most --chunk-size
waveforms at a time. Quality statistics fit candidate training rows only.
The holdout is materialized only after the benchmark protocol is locked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from aura.cache.store import CacheStore, config_fingerprint, file_content_hash
from aura.config import AuraConfig
from aura.encoders.wav2vec2 import Wav2Vec2Encoder
from aura.ingest.dataset import FilteredSpeechCommands, SampleRecord
from aura.ingest.metadata import save_embedding_matrix
from aura.provenance import source_fingerprint, runtime_versions
from aura.scoring.features import DSPFeatures
from aura.scoring.quality import QualityNormStats, compute_quality_score
from experiments.scaling import partition_corpus
from scripts.prepare_data import process_split


def create_plan(config, root, previous_metadata, held_out_speakers, seed):
    previous = pd.read_parquet(previous_metadata)
    previous_hashes = {file_content_hash(path) for path in previous.audio_path}
    records = [dict(asdict(r), audio_hash=file_content_hash(r.audio_path))
               for r in FilteredSpeechCommands(config.dataset.root, config.dataset.classes,
                                               "train", download=False).records()]
    partition = partition_corpus(records, set(previous.speaker_id), previous_hashes, held_out_speakers, seed)
    plan = {"config": asdict(config), "source_fingerprint": source_fingerprint(),
            "previous_metadata_hash": file_content_hash(previous_metadata),
            "held_out_speakers": held_out_speakers, "seed": seed,
            "available_recordings": len(records), **partition}
    path = root / "partition.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Existing partition differs; use a new data root")
    else:
        path.write_text(json.dumps(plan, indent=2))
    return plan


def features_from_row(row):
    return DSPFeatures(rms=row.rms, zero_crossing_rate=row.zero_crossing_rate,
                       spectral_centroid=row.spectral_centroid, spectral_bandwidth=row.spectral_bandwidth,
                       spectral_rolloff=row.spectral_rolloff, mfcc_mean=list(row.mfcc_mean), mfcc_std=list(row.mfcc_std),
                       clipping_ratio=row.clipping_ratio, speech_ratio=row.speech_ratio,
                       snr_db=row.snr, rms_stability=row.rms_stability)


def prepare_role(config, root, plan, role, chunk_size, protocol):
    if role == "holdout" and (protocol is None or not Path(protocol).exists()):
        raise ValueError("A locked benchmark protocol is required before holdout preparation")
    if role == "holdout":
        declaration = json.loads(Path(protocol).read_text())
        if declaration.get("partition_hash") != file_content_hash(root / "partition.json"):
            raise ValueError("Protocol does not match the planned cohort")
    final_manifest = root / "metadata" / f"{role}_manifest.json"
    if final_manifest.exists():
        from experiments.scaling import load_scaling_artifact
        load_scaling_artifact(root, role)
        print(f"Verified existing {role} artifacts", flush=True)
        return
    started = time.perf_counter()
    raw_records = plan[role]
    split = "train" if role == "train" else "validation"
    signature = config_fingerprint({"role": role, "records": raw_records, "source": source_fingerprint(),
                                    "chunk_size": chunk_size, "encoder": asdict(config.encoder)})
    if role == "train":
        # Temporary scores are replaced AFTER fitting on the complete candidate
        # pool. No holdout feature contributes to normalization statistics.
        stats = QualityNormStats(0, 1, 0, 1)
    else:
        payload = json.loads((root / "metadata/quality_norm_stats.json").read_text())
        if payload["train_metadata_hash"] != file_content_hash(root / "metadata/train.parquet"):
            raise ValueError("Stale candidate quality statistics")
        stats = QualityNormStats.from_dict(payload["stats"])
    encoder = Wav2Vec2Encoder(checkpoint=config.encoder.checkpoint, pooling=config.encoder.pooling,
                              normalize=config.encoder.normalize, device=config.encoder.device,
                              batch_size=config.encoder.batch_size)
    features = CacheStore(Path(config.cache.path) / "features")
    embeddings = CacheStore(Path(config.cache.path) / "embeddings")
    frames, matrices, all_ids, manifests = [], [], [], []
    try:
        for start in range(0, len(raw_records), chunk_size):
            subset = raw_records[start:start + chunk_size]
            directory = root / "chunks" / role / f"{start:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            chunk_manifest = directory / "manifest.json"
            if chunk_manifest.exists():
                saved = json.loads(chunk_manifest.read_text())
                if saved["signature"] != signature or saved["encoder"] != asdict(encoder.info):
                    raise ValueError("Chunk configuration or encoder revision differs")
                for path, expected in saved["artifacts"].items():
                    if file_content_hash(path) != expected:
                        raise ValueError(f"Changed chunk: {path}")
                frame = pd.read_parquet(directory / "metadata.parquet")
                matrix = np.load(directory / "embeddings.npy")
            else:
                records = [SampleRecord(**{k: v for k, v in r.items() if k != "audio_hash" and k != "split"},
                                        split=split) for r in subset]
                rows, matrix, ids, _ = process_split(records, split, config, features, embeddings, encoder, stats)
                frame = pd.DataFrame(rows)
                frame.to_parquet(directory / "metadata.parquet", index=False)
                np.save(directory / "embeddings.npy", matrix)
                saved = {"signature": signature, "encoder": asdict(encoder.info),
                         "artifacts": {str(p.resolve()): file_content_hash(p) for p in
                                       [directory / "metadata.parquet", directory / "embeddings.npy"]}}
                chunk_manifest.write_text(json.dumps(saved, indent=2))
            if frame.id.tolist() != [r["sample_id"] for r in subset] or len(matrix) != len(frame):
                raise ValueError("Chunk rows differ from the planned order")
            frames.append(frame)
            matrices.append(matrix)
            all_ids.extend(frame.id.tolist())
            manifests.append(file_content_hash(chunk_manifest))
            print(f"{role}: {min(start + chunk_size, len(raw_records))}/{len(raw_records)} ready", flush=True)
    finally:
        features.close()
        embeddings.close()
    frame = pd.concat(frames, ignore_index=True)
    matrix = np.concatenate(matrices)
    if role == "train":
        stats = QualityNormStats(float(frame.snr.min()), float(frame.snr.max()),
                                 float(frame.rms_stability.min()), float(frame.rms_stability.max()))
    frame["quality_score"] = [compute_quality_score(features_from_row(row), stats).quality_score
                              for row in frame.itertuples()]
    metadata_path = root / "metadata" / f"{role}.parquet"
    matrix_path = root / "embeddings" / f"{role}.npy"
    index_path = root / "embeddings" / f"{role}_index.parquet"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    frame["embedding_path"] = str(matrix_path)
    frame.to_parquet(metadata_path, index=False)
    save_embedding_matrix(matrix, all_ids, matrix_path, index_path)
    stats_path = root / "metadata/quality_norm_stats.json"
    if role == "train":
        stats_path.write_text(json.dumps({"stats": stats.to_dict(),
            "train_metadata_hash": file_content_hash(metadata_path), "partition_hash": file_content_hash(root / "partition.json")}, indent=2))
    final_manifest.write_text(json.dumps({
        "role": role, "source_fingerprint": source_fingerprint(), "runtime_versions": runtime_versions(),
        "encoder": asdict(encoder.info), "records": len(frame), "speakers": int(frame.speaker_id.nunique()),
        "elapsed_seconds": time.perf_counter() - started, "chunk_manifests": manifests,
        "script_hash": file_content_hash(__file__), "signature": signature,
        "artifacts": {str(p.resolve()): file_content_hash(p) for p in
                      [metadata_path, matrix_path, index_path, stats_path, root / "partition.json"]},
    }, indent=2))
    print(f"Saved {role}: {len(frame)} records from {frame.speaker_id.nunique()} speakers", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.v2.yaml")
    parser.add_argument("--data-root", default="data/scaling_v4")
    parser.add_argument("--previous-metadata", default="data/v2/metadata/train.parquet")
    parser.add_argument("--holdout-speakers", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--role", choices=["plan", "train", "holdout"], default="train")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)
    config = AuraConfig.from_yaml(args.config)
    plan = create_plan(config, root, args.previous_metadata, args.holdout_speakers, args.seed)
    print(f"Available={plan['available_recordings']}, candidates={len(plan['train'])}, "
          f"fresh holdout={len(plan['holdout'])}, duplicate exclusions={len(plan['excluded_duplicates'])}", flush=True)
    if args.role != "plan":
        prepare_role(config, root, plan, args.role, args.chunk_size, args.protocol)


if __name__ == "__main__":
    main()
