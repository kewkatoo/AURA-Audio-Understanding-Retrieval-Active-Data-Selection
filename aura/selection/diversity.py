"""
Diversity selector: k-center greedy / farthest-point sampling in
embedding space, run independently within each class.

Distance metric: embeddings are L2-normalized (Phase 1 guarantee), so
cosine similarity is a plain dot product and cosine distance is
`1 - dot`. We use this directly rather than computing Euclidean distance
separately -- for unit vectors the two are monotonically related
(||a-b||^2 = 2 - 2*dot(a,b)), so ranking by cosine distance and by
Euclidean distance select identically; cosine is cheaper (one dot product,
no sqrt) and is the natural metric for normalized embeddings.

Algorithm (per class):
  1. Pick a first center. Not centroid-nearest (which would bias toward
     the "average" sample) but chosen via the seeded RNG, as required --
     this keeps the diversity selector's outcome reproducible per seed
     while not hard-coding a single deterministic start that would make
     every seed produce the same diversity selection.
  2. Maintain min_dist[i] = distance from candidate i to its NEAREST
     already-selected center, for all not-yet-selected i.
  3. Repeat n_to_select-1 times: pick argmax(min_dist) (farthest from
     every current center = best coverage gain), add it as a new center,
     then update min_dist for all remaining candidates against ONLY the
     new center (min_dist[i] = min(min_dist[i], dist(i, new_center))).

Complexity: step 3 runs n_to_select-1 times, each doing one (n_remaining,
dim) x (dim,) matrix-vector product to get distances to the new center =
O(n * dim) per iteration, O(n * k * dim) total per class, where n = class
pool size and k = n_to_select. Memory is O(n) for min_dist plus the
(n, dim) embedding block already held in `embeddings` -- there is no O(n^2)
pairwise distance matrix at any point, which matters once n grows past a
few thousand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aura.selection.base import Selector


def _cosine_distance_to_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    """matrix: (n, dim), point: (dim,), both assumed L2-normalized.
    Returns (n,) array of cosine distances (1 - dot)."""
    similarities = matrix @ point
    return 1.0 - similarities


class DiversitySelector(Selector):
    name = "diversity"

    def _select_within_class(
        self,
        class_label: str,
        candidate_ids: list[str],
        n_to_select: int,
        metadata: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> list[str]:
        n = len(candidate_ids)
        if n_to_select >= n:
            # Degenerate case: selecting the whole (or more than the)
            # class pool -- diversity ordering is moot, return everything
            # requested in deterministic (sorted) order.
            return list(candidate_ids[:n_to_select])

        matrix = np.stack([embeddings[sid] for sid in candidate_ids], axis=0)  # (n, dim)

        remaining_mask = np.ones(n, dtype=bool)
        selected_local_indices: list[int] = []

        # Step 1: seeded first center.
        first_idx = int(rng.integers(0, n))
        selected_local_indices.append(first_idx)
        remaining_mask[first_idx] = False

        min_dist = _cosine_distance_to_point(matrix, matrix[first_idx])
        min_dist[first_idx] = -np.inf  # never re-selected

        # Steps 2-3: greedy farthest-point sampling.
        for _ in range(n_to_select - 1):
            # argmax over remaining candidates only.
            candidate_dist = np.where(remaining_mask, min_dist, -np.inf)
            next_idx = int(np.argmax(candidate_dist))
            selected_local_indices.append(next_idx)
            remaining_mask[next_idx] = False

            new_center_dist = _cosine_distance_to_point(matrix, matrix[next_idx])
            min_dist = np.minimum(min_dist, new_center_dist)
            min_dist[next_idx] = -np.inf

        return [candidate_ids[i] for i in selected_local_indices]