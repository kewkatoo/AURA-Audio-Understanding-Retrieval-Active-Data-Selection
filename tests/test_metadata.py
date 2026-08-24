from __future__ import annotations

import numpy as np

from aura.ingest.metadata import (
    save_metadata_parquet, load_metadata_parquet,
    save_embedding_matrix, load_embedding_matrix, get_embedding_for_sample,
    METADATA_COLUMNS,
)


def test_metadata_parquet_round_trip(tmp_path):
    rows = [
        {col: (0.0 if col not in ("mfcc_mean", "mfcc_std") else [0.0] * 13) for col in METADATA_COLUMNS}
        for _ in range(3)
    ]
    for i, row in enumerate(rows):
        row["id"] = f"sample_{i}"
        row["label"] = "yes"

    path = tmp_path / "meta.parquet"
    save_metadata_parquet(rows, path)
    df = load_metadata_parquet(path)

    assert len(df) == 3
    assert list(df["id"]) == ["sample_0", "sample_1", "sample_2"]
    assert set(METADATA_COLUMNS).issubset(set(df.columns))


def test_embedding_matrix_round_trip(tmp_path):
    embeddings = np.random.randn(5, 8).astype(np.float32)
    sample_ids = [f"s{i}" for i in range(5)]

    matrix_path = tmp_path / "train.npy"
    index_path = tmp_path / "train_index.parquet"
    save_embedding_matrix(embeddings, sample_ids, matrix_path, index_path)

    loaded_matrix, loaded_index = load_embedding_matrix(matrix_path, index_path)
    assert np.allclose(loaded_matrix, embeddings)
    assert list(loaded_index["sample_id"]) == sample_ids


def test_get_embedding_for_sample_returns_correct_row(tmp_path):
    embeddings = np.arange(20, dtype=np.float32).reshape(5, 4)
    sample_ids = [f"s{i}" for i in range(5)]
    matrix_path = tmp_path / "train.npy"
    index_path = tmp_path / "train_index.parquet"
    save_embedding_matrix(embeddings, sample_ids, matrix_path, index_path)

    matrix, index_df = load_embedding_matrix(matrix_path, index_path)
    emb = get_embedding_for_sample("s3", matrix, index_df)
    assert np.array_equal(emb, embeddings[3])


def test_get_embedding_for_sample_missing_raises(tmp_path):
    embeddings = np.zeros((2, 4), dtype=np.float32)
    sample_ids = ["s0", "s1"]
    matrix_path = tmp_path / "train.npy"
    index_path = tmp_path / "train_index.parquet"
    save_embedding_matrix(embeddings, sample_ids, matrix_path, index_path)

    matrix, index_df = load_embedding_matrix(matrix_path, index_path)
    try:
        get_embedding_for_sample("does_not_exist", matrix, index_df)
        assert False, "expected KeyError"
    except KeyError:
        pass
