"""
Combined quality + diversity selector (AURA).

score(x) = alpha * quality_norm(x) + (1 - alpha) * diversity_gain(x)

This is a genuinely greedy, dynamically-updated trade-off, NOT "sort by
quality, then diversify the top-K" -- at every step of selection, EVERY
remaining candidate's diversity_gain (distance to its nearest already-
selected point) is recomputed, so a lower-quality point that has become
the most distinctive remaining option can outscore and be chosen over a
higher-quality but now-redundant point. A two-stage sort-then-diversify
approach could never make that trade at all; this can, every iteration.

Normalization (per class, per selection run):
- quality_norm: min-max normalized over the candidate pool for this class
  (NOT reusing Phase 1's train-pool-global normalization stats -- those
  were fit for the standalone quality score's own [0,1] range across the
  whole train split; here we re-normalize within the current class subset
  so quality and diversity are on comparable per-class scales for the
  combination). If the class pool has zero quality variance, quality_norm
  is a neutral 0.5 for every candidate (same degenerate-case convention as
  Phase 1's quality.py).
- diversity_gain: cosine distance to nearest selected point, in [0, 2] for
  L2-normalized embeddings, divided by 2 and clipped to [0, 1].

First-pick convention: before any point is selected, every candidate is
"maximally diverse" relative to the empty selected set, so diversity_gain
is treated as 1.0 for all candidates on the first pick -- meaning the
first pick reduces to argmax(quality_norm), letting alpha still control
outcome (alpha=1 -> pure quality first pick; alpha=0 -> all candidates
tie on diversity, tie broken by a seeded random permutation, matching the
diversity selector's own seeded-first-pick behavior).

DEFAULT_ALPHA = 0.5 (equal weight to quality and diversity) is a starting
point, not a claimed-optimal value -- alpha is fully configurable via the
CombinedSelector constructor, per the Phase 2 spec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aura.selection.base import Selector
from aura.selection.diversity import _cosine_distance_to_point

DEFAULT_ALPHA = 0.5


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax - vmin < 1e-9:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - vmin) / (vmax - vmin)


class CombinedSelector(Selector):
    name = "combined"

    def __init__(self, alpha: float = DEFAULT_ALPHA):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        self.alpha = alpha

    def _config_dict(self) -> dict:
        return {"alpha": self.alpha, "default_alpha": DEFAULT_ALPHA}

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
            return list(candidate_ids[:n_to_select])

        id_to_quality = dict(zip(metadata["id"], metadata["quality_score"]))
        quality_raw = np.array([float(id_to_quality[sid]) for sid in candidate_ids])
        quality_norm = _min_max_normalize(quality_raw)

        matrix = np.stack([embeddings[sid] for sid in candidate_ids], axis=0)  # (n, dim)

        alpha = self.alpha
        remaining_mask = np.ones(n, dtype=bool)
        selected_local_indices: list[int] = []

        # First pick: diversity_gain treated as 1.0 for everyone (see
        # module docstring), so score reduces to alpha*quality_norm, with
        # ties broken by a seeded random permutation for reproducibility
        # without hard-coding a fixed winner when alpha==0.
        first_scores = alpha * quality_norm + (1 - alpha) * 1.0
        tie_break = rng.permutation(n)  # deterministic given seed
        # Sort by (-score, tie_break) so highest score wins, ties broken
        # by the seeded permutation order.
        order = sorted(range(n), key=lambda i: (-first_scores[i], tie_break[i]))
        first_idx = order[0]
        selected_local_indices.append(first_idx)
        remaining_mask[first_idx] = False

        min_dist = _cosine_distance_to_point(matrix, matrix[first_idx])
        min_dist[first_idx] = -np.inf

        for _ in range(n_to_select - 1):
            diversity_gain = np.clip(min_dist / 2.0, 0.0, 1.0)
            score = alpha * quality_norm + (1 - alpha) * diversity_gain
            score = np.where(remaining_mask, score, -np.inf)
            next_idx = int(np.argmax(score))
            selected_local_indices.append(next_idx)
            remaining_mask[next_idx] = False

            new_center_dist = _cosine_distance_to_point(matrix, matrix[next_idx])
            min_dist = np.minimum(min_dist, new_center_dist)
            min_dist[next_idx] = -np.inf

        return [candidate_ids[i] for i in selected_local_indices]