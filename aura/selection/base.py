"""
Selector interface shared by all Phase 2 strategies.

Deviation from the literal sketch in the Phase 2 spec
(`selector.select(...) -> selected_sample_ids`): `select()` here returns a
`SelectionResult` object rather than a bare list. `result.selected_ids` is
exactly that list, so nothing is lost -- but the spec's own
"REPRODUCIBILITY" section requires every run to record strategy, budget,
seed, alpha, class counts, and timestamp, and bolting that onto a bare
list return would mean either a second parallel return value or a mutable
out-parameter, both worse than one structured result object. This is
noted here explicitly rather than silently deviating from the sketch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from aura.selection.utils import validate_budget


@dataclass
class SelectionResult:
    strategy: str
    requested_budget: int
    actual_budget: int          # may be < requested_budget if the pool
                                 # couldn't supply it (see utils.allocate_budget_per_class)
    seed: int
    alpha: float | None         # only meaningful for the combined strategy
    selected_ids: list[str]
    class_counts: dict[str, int]
    per_class_requested: dict[str, int]
    timestamp: str
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Selector(ABC):
    """All selectors must be class-balanced, deterministic given a seed,
    and reject metadata containing evaluation rows at the API boundary.
    """

    name: str = "base"

    @abstractmethod
    def _select_within_class(
        self,
        class_label: str,
        candidate_ids: list[str],
        n_to_select: int,
        metadata: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> list[str]:
        """Strategy-specific per-class selection. Implemented by subclasses;
        `select()` handles budget allocation, looping over classes, and
        result-record assembly so this is the only method each strategy
        needs to write."""
        ...

    def select(
        self,
        metadata: pd.DataFrame,
        embeddings: dict[str, np.ndarray],
        budget: int,
        seed: int,
        **kwargs,
    ) -> SelectionResult:
        from aura.selection.utils import allocate_budget_per_class, get_class_pools, make_rng, class_counts as count_classes

        total_available = len(metadata)
        validate_budget(budget, total_available)
        required = {"id", "label", "split"}
        if not required.issubset(metadata.columns):
            raise ValueError(f"Metadata requires columns {sorted(required)}")
        if metadata[list(required)].isnull().any().any() or metadata["id"].duplicated().any():
            raise ValueError("Metadata IDs/labels/splits must be nonnull and IDs unique")
        if not metadata["split"].eq("train").all():
            raise ValueError("Selectors accept training rows only")
        if self.name in {"quality", "combined"}:
            if "quality_score" not in metadata or not np.isfinite(metadata["quality_score"]).all():
                raise ValueError("Finite quality scores are required")
        if self.name in {"diversity", "combined"}:
            vectors = []
            for sid in metadata["id"]:
                if sid not in embeddings:
                    raise ValueError(f"Missing embedding: {sid}")
                vectors.append(embeddings[sid])
            try:
                matrix = np.stack(vectors)
            except ValueError as exc:
                raise ValueError("Embedding dimensions must match") from exc
            if (matrix.ndim != 2 or not np.isfinite(matrix).all()
                    or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)):
                raise ValueError("Diversity selection requires finite unit-normalized embeddings")

        pools = get_class_pools(metadata)
        available_counts = {c: len(ids) for c, ids in pools.items()}
        per_class_budget = allocate_budget_per_class(
            classes=list(pools.keys()), budget=budget, available_counts=available_counts
        )

        rng = make_rng(seed)
        selected: list[str] = []
        for class_label in sorted(pools.keys()):
            n_to_select = per_class_budget.get(class_label, 0)
            if n_to_select <= 0:
                continue
            candidate_ids = sorted(pools[class_label])  # deterministic order in
            class_selected = self._select_within_class(
                class_label, candidate_ids, n_to_select, metadata, embeddings, rng
            )
            if len(class_selected) != n_to_select:
                raise RuntimeError(
                    f"{self.name} selector returned {len(class_selected)} ids for "
                    f"class {class_label!r}, expected {n_to_select}."
                )
            if len(set(class_selected)) != len(class_selected):
                raise RuntimeError(
                    f"{self.name} selector produced duplicate ids within class {class_label!r}."
                )
            selected.extend(class_selected)

        actual_budget = len(selected)
        result_class_counts = count_classes(selected, metadata)

        return SelectionResult(
            strategy=self.name,
            requested_budget=budget,
            actual_budget=actual_budget,
            seed=seed,
            alpha=getattr(self, "alpha", None),
            selected_ids=selected,
            class_counts=result_class_counts,
            per_class_requested=per_class_budget,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=self._config_dict(),
        )

    def _config_dict(self) -> dict:
        """Override in subclasses to record strategy-specific config
        (e.g. alpha for CombinedSelector) into SelectionResult.config."""
        return {}
