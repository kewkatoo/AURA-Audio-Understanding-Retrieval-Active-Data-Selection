import numpy as np
import pandas as pd
import pytest

from experiments.representative import RepresentativeSelector, QualityRepresentativeSelector, coverage_diagnostics


def clustered_pool():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal([1, 0], 0.005, (15, 2)),
                        rng.normal([0, 1], 0.005, (15, 2)), [[-1, -1]]]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = [f"s{i:02}" for i in range(len(x))]
    meta = pd.DataFrame({"id": ids, "label": "word", "split": "train",
                         "quality_score": np.linspace(0, 1, len(x))})
    return meta, dict(zip(ids, x))


def test_representatives_cover_dense_groups_before_single_outlier():
    meta, embeddings = clustered_pool()
    selected = RepresentativeSelector().select(meta, embeddings, 2, 4).selected_ids
    assert any(int(s[1:]) < 15 for s in selected)
    assert any(15 <= int(s[1:]) < 30 for s in selected)
    assert "s30" not in selected
    assert selected == RepresentativeSelector().select(meta.sample(frac=1, random_state=2), embeddings, 2, 4).selected_ids


def test_quality_gate_respects_retained_pool_and_capacity():
    meta, embeddings = clustered_pool()
    result = QualityRepresentativeSelector(0.5).select(meta, embeddings, 4, 0)
    assert len(set(result.selected_ids)) == 4
    assert all(int(s[1:]) >= 15 for s in result.selected_ids)
    assert len(QualityRepresentativeSelector(0.1).select(meta, embeddings, 25, 0).selected_ids) == 25
    assert len(RepresentativeSelector().select(meta, embeddings, 50, 0).selected_ids) == len(meta)


def test_duplicate_vectors_still_produce_exact_unique_budget():
    meta, _ = clustered_pool()
    embeddings = {sid: np.array([1, 0], dtype=np.float32) for sid in meta.id}
    result = RepresentativeSelector().select(meta, embeddings, 5, 0)
    assert len(set(result.selected_ids)) == 5


def test_representative_rejects_invalid_data_and_evaluation_rows():
    meta, embeddings = clustered_pool()
    embeddings[meta.id.iloc[0]] = np.zeros(2)
    with pytest.raises(ValueError, match="unit-normalized"):
        RepresentativeSelector().select(meta, embeddings, 2, 0)
    meta.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="training"):
        RepresentativeSelector().select(meta, embeddings, 2, 0)
    with pytest.raises(ValueError):
        RepresentativeSelector(float("nan"))


def test_full_pool_has_zero_coverage_distance():
    meta, embeddings = clustered_pool()
    full = coverage_diagnostics(meta, embeddings, meta.id.tolist())
    subset = coverage_diagnostics(meta, embeddings, meta.id.iloc[:2].tolist())
    assert full["mean_coverage_distance"] < 1e-6
    assert subset["mean_coverage_distance"] > full["mean_coverage_distance"]
