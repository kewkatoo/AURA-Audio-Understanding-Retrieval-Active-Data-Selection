"""
Metadata + embedding persistence.

Design decision (per Phase 1 spec, section "Embeddings"):
Embeddings are stored as ONE dense numpy matrix per split
(data/embeddings/{split}.npy, shape (n_samples, embedding_dim)) plus a
row-index Parquet file mapping row_index -> sample_id
(data/embeddings/{split}_index.parquet). This avoids both (a) thousands of
tiny per-sample files, which is slow on most filesystems and painful to
version, and (b) storing raw embedding vectors inline in the main metadata
Parquet, which would bloat that file and duplicate data if embeddings are
regenerated independently of metadata (e.g. swapping encoders in a later
phase). The main metadata Parquet instead stores `embedding_path` (the
split-level .npy path) and each row's `sample_id`, which is joined against
the index Parquet to find its row.

Memory note: at 768-dim float32, 2000 samples = ~6MB, and the full ~10-class
dataset (~10 * ~3000 = ~30k samples) is under 100MB. This comfortably fits
in memory for the dataset sizes in scope here; if this pipeline is later
pointed at a much larger corpus, the alternative would be a memory-mapped
.npy (np.load(..., mmap_mode='r')) or a chunked format like Parquet with
list-of-float32 columns -- flagging this now rather than pre-building it,
since it isn't needed at current scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METADATA_COLUMNS = [
    "id", "audio_path", "label", "speaker_id",
    "sample_rate", "original_duration", "processed_duration", "speech_ratio",
    "rms", "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth",
    "spectral_rolloff", "mfcc_mean", "mfcc_std", "clipping_ratio", "snr",
    "rms_stability", "quality_score", "embedding_path",
    "dataset_version", "encoder_name", "encoder_checkpoint", "split",
]


def build_metadata_row(
    sample_id: str,
    audio_path: str,
    label: str,
    speaker_id: str | None,
    processed_audio,     # aura.ingest.audio.ProcessedAudio
    dsp_features,         # aura.scoring.features.DSPFeatures
    quality_score,         # aura.scoring.quality.QualityScore
    embedding_path: str,
    dataset_version: str,
    encoder_name: str,
    encoder_checkpoint: str,
    split: str,
) -> dict:
    return {
        "id": sample_id,
        "audio_path": audio_path,
        "label": label,
        "speaker_id": speaker_id,
        "sample_rate": processed_audio.sample_rate,
        "original_duration": processed_audio.original_duration_s,
        "processed_duration": processed_audio.processed_duration_s,
        "speech_ratio": processed_audio.speech_ratio,
        "rms": dsp_features.rms,
        "zero_crossing_rate": dsp_features.zero_crossing_rate,
        "spectral_centroid": dsp_features.spectral_centroid,
        "spectral_bandwidth": dsp_features.spectral_bandwidth,
        "spectral_rolloff": dsp_features.spectral_rolloff,
        "mfcc_mean": dsp_features.mfcc_mean,
        "mfcc_std": dsp_features.mfcc_std,
        "clipping_ratio": dsp_features.clipping_ratio,
        "snr": dsp_features.snr_db,
        "rms_stability": dsp_features.rms_stability,
        "quality_score": quality_score.quality_score,
        "embedding_path": embedding_path,
        "dataset_version": dataset_version,
        "encoder_name": encoder_name,
        "encoder_checkpoint": encoder_checkpoint,
        "split": split,
    }


def save_metadata_parquet(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    df.to_parquet(path, index=False)


def load_metadata_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_embedding_matrix(
    embeddings: np.ndarray,
    sample_ids: list[str],
    matrix_path: str | Path,
    index_path: str | Path,
) -> None:
    """Persists the dense (n_samples, dim) embedding matrix and its
    row_index -> sample_id mapping. Row order in the matrix is exactly the
    order of `sample_ids`."""
    matrix_path = Path(matrix_path)
    index_path = Path(index_path)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(matrix_path, embeddings.astype(np.float32))

    index_df = pd.DataFrame({
        "row_index": np.arange(len(sample_ids)),
        "sample_id": sample_ids,
    })
    index_df.to_parquet(index_path, index=False)


def load_embedding_matrix(
    matrix_path: str | Path, index_path: str | Path, mmap: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = np.load(matrix_path, mmap_mode="r" if mmap else None)
    index_df = pd.read_parquet(index_path)
    return matrix, index_df


def get_embedding_for_sample(
    sample_id: str, matrix: np.ndarray, index_df: pd.DataFrame
) -> np.ndarray:
    row = index_df.loc[index_df["sample_id"] == sample_id]
    if row.empty:
        raise KeyError(f"sample_id {sample_id!r} not found in embedding index.")
    row_index = int(row.iloc[0]["row_index"])
    return matrix[row_index]
