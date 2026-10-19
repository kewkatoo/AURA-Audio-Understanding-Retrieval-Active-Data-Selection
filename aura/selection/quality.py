"""
Quality selector.

Ranks each class's candidates by the EXISTING Phase 1 `quality_score`
column (never recomputed here) descending, breaking ties deterministically
by sample_id ascending so the result is identical across runs/platforms
regardless of quality_score float-equality edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aura.selection.base import Selector


class QualitySelector(Selector):
    name = "quality"

    def _select_within_class(
        self,
        class_label: str,
        candidate_ids: list[str],
        n_to_select: int,
        metadata: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> list[str]:
        id_to_quality = dict(zip(metadata["id"], metadata["quality_score"]))
        # Sort by (-quality_score, sample_id): highest quality first,
        # sample_id ascending as a deterministic tiebreaker.
        ranked = sorted(
            candidate_ids,
            key=lambda sid: (-float(id_to_quality[sid]), sid),
        )
        return ranked[:n_to_select]