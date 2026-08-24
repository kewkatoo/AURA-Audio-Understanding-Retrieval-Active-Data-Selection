#!/usr/bin/env python
"""
Inspect the outputs of scripts/prepare_data.py: sample counts per class,
durations, quality-score statistics, embedding dimensionality, and a check
for missing/corrupt files referenced by metadata.

Usage:
    python scripts/inspect_dataset.py --config configs/experiment.yaml
    python scripts/inspect_dataset.py --config configs/experiment.yaml --splits train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.config import AuraConfig


def inspect_split(config: AuraConfig, split: str) -> None:
    metadata_path = Path(config.paths.metadata_dir) / f"{split}.parquet"
    embedding_matrix_path = Path(config.paths.embeddings_dir) / f"{split}.npy"
    embedding_index_path = Path(config.paths.embeddings_dir) / f"{split}_index.parquet"

    print(f"\n{'=' * 60}\nSPLIT: {split}\n{'=' * 60}")

    if not metadata_path.exists():
        print(f"  [MISSING] metadata not found at {metadata_path} -- run prepare_data.py first.")
        return

    df = pd.read_parquet(metadata_path)
    print(f"Samples: {len(df)}")

    print("\nSamples per class:")
    counts = df["label"].value_counts().sort_index()
    for label in config.dataset.classes:
        n = int(counts.get(label, 0))
        print(f"  {str(label):>8}: {n}")
    missing_classes = set(config.dataset.classes) - set(counts.index)
    if missing_classes:
        print(f"  WARNING: classes with zero samples in this split: {sorted(missing_classes)}")

    total_duration = df["processed_duration"].sum()
    mean_duration = df["processed_duration"].mean()
    print(f"\nTotal duration (processed): {total_duration:.1f}s ({total_duration / 60:.1f} min)")
    print(f"Mean duration (processed): {mean_duration:.3f}s")
    print(f"Mean duration (original):  {df['original_duration'].mean():.3f}s")
    print(f"Mean speech_ratio: {df['speech_ratio'].mean():.3f}")

    print("\nQuality score statistics:")
    print(df["quality_score"].describe().to_string())

    print("\nDSP feature summary (mean values):")
    for col in ["rms", "zero_crossing_rate", "spectral_centroid",
                "spectral_bandwidth", "spectral_rolloff", "clipping_ratio",
                "snr", "rms_stability"]:
        print(f"  {col:>20s}: {df[col].mean():.4f}")

    # Embedding checks
    if embedding_matrix_path.exists() and embedding_index_path.exists():
        matrix = np.load(embedding_matrix_path)
        index_df = pd.read_parquet(embedding_index_path)
        print(f"\nEmbedding matrix shape: {matrix.shape}")
        print(f"Embedding index rows: {len(index_df)}")
        if matrix.shape[0] != len(df):
            print(
                f"  WARNING: embedding matrix row count ({matrix.shape[0]}) "
                f"!= metadata row count ({len(df)})"
            )
        if len(set(index_df['sample_id'])) != len(index_df):
            print("  WARNING: duplicate sample_ids in embedding index.")
        norms = np.linalg.norm(matrix, axis=1)
        print(f"Embedding L2-norm mean/std: {norms.mean():.4f} / {norms.std():.4f}")
        nan_count = int(np.isnan(matrix).sum())
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN values found in embedding matrix.")
    else:
        print(f"\n  [MISSING] embeddings not found at {embedding_matrix_path}")

    # Missing/corrupt audio file check
    print("\nChecking referenced audio files exist on disk...")
    missing_files = []
    for path in df["audio_path"]:
        if not Path(path).exists():
            missing_files.append(path)
    if missing_files:
        print(f"  WARNING: {len(missing_files)} referenced audio files are missing.")
        for p in missing_files[:5]:
            print(f"    {p}")
        if len(missing_files) > 5:
            print(f"    ... and {len(missing_files) - 5} more")
    else:
        print("  All referenced audio files present.")

    dup_ids = df["id"].duplicated().sum()
    if dup_ids > 0:
        print(f"  WARNING: {dup_ids} duplicate sample ids in metadata.")

    null_counts = df.isnull().sum()
    nonzero_nulls = null_counts[null_counts > 0]
    if len(nonzero_nulls) > 0:
        print(f"  WARNING: null values found in columns:\n{nonzero_nulls.to_string()}")


def main():
    parser = argparse.ArgumentParser(description="Inspect AURA Phase 1 pipeline outputs.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"],
        choices=["train", "validation", "test"],
    )
    args = parser.parse_args()

    config = AuraConfig.from_yaml(args.config)
    for split in args.splits:
        inspect_split(config, split)


if __name__ == "__main__":
    main()
