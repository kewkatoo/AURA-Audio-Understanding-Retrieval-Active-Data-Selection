"""Planning and artifact validation for a frozen full-corpus scaling study."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aura.cache.store import config_fingerprint, file_content_hash
from aura.ingest.metadata import load_embedding_matrix
from aura.provenance import source_fingerprint


def partition_corpus(records, previous_speakers, previous_hashes, held_out_speakers=250, seed=20261018):
    """Predeclare speaker holdout, then remove byte-identical cross-role audio.

    Inputs are dictionaries including audio_hash. Only IDs/hashes/speakers are
    used for the partition; no model outcomes inform the choice.
    """
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Corpus IDs must be unique")
    fresh = sorted({r["speaker_id"] for r in records} - set(previous_speakers))
    if held_out_speakers < 1 or held_out_speakers > len(fresh):
        raise ValueError("Insufficient previously unseen speakers")
    rng = np.random.default_rng(seed)
    held_out = set(rng.choice(fresh, held_out_speakers, replace=False).tolist())
    holdout = [r for r in records if r["speaker_id"] in held_out and r["audio_hash"] not in previous_hashes]
    heldout_hashes = {r["audio_hash"] for r in holdout}
    train = [r for r in records if r["speaker_id"] not in held_out and r["audio_hash"] not in heldout_hashes]
    accepted = {r["sample_id"] for r in train + holdout}
    excluded = [r for r in records if r["sample_id"] not in accepted]
    if not train or not holdout or {r["label"] for r in train} != {r["label"] for r in holdout}:
        raise ValueError("Both roles must be nonempty and cover identical classes")
    return {"train": sorted(train, key=lambda r: r["sample_id"]),
            "holdout": sorted(holdout, key=lambda r: r["sample_id"]),
            "excluded_duplicates": sorted(excluded, key=lambda r: r["sample_id"]),
            "reserved_speakers": sorted(held_out)}


def load_scaling_artifact(root, role):
    if role not in {"train", "holdout"}:
        raise ValueError("Unknown scaling artifact role")
    root = Path(root)
    manifest = json.loads((root / "metadata" / f"{role}_manifest.json").read_text())
    if manifest["source_fingerprint"] != source_fingerprint():
        raise ValueError("Frozen preparation pipeline changed")
    for path, expected in manifest["artifacts"].items():
        if file_content_hash(path) != expected:
            raise ValueError(f"Changed artifact: {path}")
    metadata = pd.read_parquet(root / "metadata" / f"{role}.parquet")
    matrix, index = load_embedding_matrix(root / "embeddings" / f"{role}.npy",
                                          root / "embeddings" / f"{role}_index.parquet")
    if (metadata.id.duplicated().any() or metadata.id.tolist() != index.sample_id.tolist()
            or index.row_index.tolist() != list(range(len(metadata)))
            or matrix.shape != (len(metadata), manifest["encoder"]["embedding_dim"])
            or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)):
        raise ValueError("Invalid scaling metadata/embedding alignment or values")
    expected_split = "train" if role == "train" else "validation"
    if not metadata.split.eq(expected_split).all():
        raise ValueError("Invalid scaling split labels")
    return metadata, matrix, manifest
