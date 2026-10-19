"""Class-balanced cluster exemplars, optionally after a modest quality gate.

These experimental methods reuse the frozen v2 embeddings. Unlike farthest
point selection, a selected exemplar is near a cluster centroid, so dense
regions can be represented without deliberately maximizing isolation.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from aura.selection.base import Selector


class RepresentativeSelector(Selector):
    name = "representative"

    def __init__(self, retain_fraction: float = 1.0):
        if not np.isfinite(retain_fraction) or not 0 < retain_fraction <= 1:
            raise ValueError("retain_fraction must be in (0, 1]")
        self.retain_fraction = float(retain_fraction)

    def _config_dict(self):
        return {"retain_fraction": self.retain_fraction, "n_init": 5,
                "algorithm": "kmeans_nearest_assigned_exemplar", "metric": "euclidean_on_unit_vectors"}

    def _select_within_class(self, class_label, candidate_ids, n_to_select, metadata, embeddings, rng):
        ids = list(candidate_ids)
        if self.retain_fraction < 1:
            if "quality_score" not in metadata or not np.isfinite(metadata.quality_score).all():
                raise ValueError("Quality gate requires finite quality scores")
            quality = metadata.set_index("id").quality_score.to_dict()
            keep = max(n_to_select, math.ceil(self.retain_fraction * len(ids)))
            ids = sorted(sorted(ids, key=lambda sid: (-quality[sid], sid))[:keep])
        try:
            matrix = np.stack([embeddings[sid] for sid in ids])
        except (KeyError, ValueError) as exc:
            raise ValueError("Missing or inconsistent embeddings") from exc
        if (matrix.ndim != 2 or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)):
            raise ValueError("Representative selection requires finite unit-normalized embeddings")
        if n_to_select >= len(ids):
            return ids[:n_to_select]
        clusters = KMeans(n_clusters=n_to_select, n_init=5,
                          random_state=int(rng.integers(0, 2**31 - 1)))
        with warnings.catch_warnings():
            # Duplicate vectors can produce fewer occupied clusters. Fill
            # the remaining slots deterministically instead of duplicating IDs.
            warnings.simplefilter("ignore", ConvergenceWarning)
            assignments = clusters.fit_predict(matrix)
        chosen = []
        for label in sorted(set(assignments)):
            members = np.flatnonzero(assignments == label)
            distances = np.sum((matrix[members] - clusters.cluster_centers_[label]) ** 2, axis=1)
            chosen.append(int(members[np.argmin(distances)]))
        chosen_set = set(chosen)
        if len(chosen) < n_to_select:
            chosen.extend(i for i in range(len(ids)) if i not in chosen_set)
        return [ids[i] for i in chosen[:n_to_select]]


class QualityRepresentativeSelector(RepresentativeSelector):
    name = "quality_representative"

    def __init__(self, retain_fraction: float = 0.75):
        super().__init__(retain_fraction)


def coverage_diagnostics(metadata, embeddings, selected_ids, neighbors=5):
    """Training-only geometry diagnostics, never used to rank validation models.

    Lower nearest-selected distance means better average pool coverage.
    Higher selected kNN distance means more isolated selected examples.
    Quadratic per-class matrices are acceptable for the 200/class study pool;
    this helper is not designed for a large unpartitioned corpus.
    """
    selected = set(selected_ids)
    distances_to_selection, selected_isolation = [], []
    for _, group in metadata.groupby("label"):
        ids = sorted(group.id)
        x = np.stack([embeddings[sid] for sid in ids])
        distances = np.clip(1 - x @ x.T, 0, 2)
        chosen = [i for i, sid in enumerate(ids) if sid in selected]
        if not chosen:
            continue
        distances_to_selection.extend(distances[:, chosen].min(axis=1).tolist())
        np.fill_diagonal(distances, np.inf)
        k = min(neighbors, len(ids) - 1)
        if k:
            isolation = np.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1)
            selected_isolation.extend(isolation[chosen].tolist())
    return {"mean_coverage_distance": float(np.mean(distances_to_selection)),
            "p95_coverage_distance": float(np.percentile(distances_to_selection, 95)),
            "selected_knn_distance": float(np.mean(selected_isolation)) if selected_isolation else 0.0}
