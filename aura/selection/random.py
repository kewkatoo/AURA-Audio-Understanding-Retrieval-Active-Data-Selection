"""
Random baseline selector.

Class-balanced by construction (via Selector.select's shared allocation
logic) and deterministic given a seed -- selection within each class is
np.random.Generator.choice without replacement, seeded through the single
RNG passed down from select(), never global np.random state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aura.selection.base import Selector


class RandomSelector(Selector):
    name = "random"

    def _select_within_class(
        self,
        class_label: str,
        candidate_ids: list[str],
        n_to_select: int,
        metadata: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> list[str]:
        # candidate_ids is already sorted (deterministic ordering) by the
        # caller; rng.choice over that fixed order is what makes the draw
        # reproducible for a given seed.
        indices = rng.choice(len(candidate_ids), size=n_to_select, replace=False)
        return [candidate_ids[i] for i in indices]