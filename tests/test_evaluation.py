import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from aura.cache.store import file_content_hash
from aura.ingest.metadata import save_embedding_matrix
from aura.provenance import PIPELINE_VERSION, source_fingerprint
from scripts import evaluate_selection as evaluation


def test_benchmark_locks_choices_before_opening_test(tmp_path, monkeypatch):
    metadata_dir, embedding_dir = tmp_path / "metadata", tmp_path / "embeddings"
    metadata_dir.mkdir()
    embedding_dir.mkdir()
    rng = np.random.default_rng(3)
    for split, n in [("train", 20), ("validation", 10), ("test", 10)]:
        labels = np.array(["a", "b"] * (n // 2))
        x = rng.normal(0, 0.1, (n, 4)).astype(np.float32)
        x[:, 0] += np.where(labels == "a", 1, -1)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        ids = [f"{split}-{i}" for i in range(n)]
        frame = pd.DataFrame({"id": ids, "label": labels, "split": split,
                              "speaker_id": ids, "quality_score": rng.uniform(size=n)})
        meta_path = metadata_dir / f"{split}.parquet"
        matrix_path, index_path = embedding_dir / f"{split}.npy", embedding_dir / f"{split}_index.parquet"
        frame.to_parquet(meta_path)
        save_embedding_matrix(x, ids, matrix_path, index_path)
        manifest = {"pipeline_version": PIPELINE_VERSION, "source_fingerprint": source_fingerprint(),
                    "encoder": {"checkpoint": "test"},
                    "artifacts": {str(p): file_content_hash(p) for p in [meta_path, matrix_path, index_path]}}
        (metadata_dir / f"{split}_manifest.json").write_text(json.dumps(manifest))
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump({"paths": {"metadata_dir": str(metadata_dir), "embeddings_dir": str(embedding_dir)}}))
    output = tmp_path / "benchmark"
    load_split = evaluation.load_split

    def guarded_load(config, split):
        if split == "test":
            assert (output / "validation_choices.json").exists()
        return load_split(config, split)

    monkeypatch.setattr(evaluation, "load_split", guarded_load)
    monkeypatch.setattr(sys, "argv", ["evaluate_selection", "--config", str(config_path),
        "--output-dir", str(output), "--budgets", "4", "--seeds", "0", "1",
        "--alphas", "0", "0.5", "1", "--c-values", "1"])
    evaluation.main()
    runs = pd.read_csv(output / "runs.csv")
    assert len(runs) == 9
    assert runs.test_accuracy.between(0, 1).all()
    assert set(runs.strategy) == {"random", "quality", "diversity", "combined", "full_pool"}
    assert (output / "REPORT.md").exists()
    with pytest.raises(ValueError, match="already complete"):
        evaluation.main()
