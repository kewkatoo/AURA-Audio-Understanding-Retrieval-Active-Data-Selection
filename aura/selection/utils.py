"""
Shared utilities for Phase 2 selection strategies.

Kept deliberately separate from any one strategy: budget allocation,
embedding lookup, and RNG construction are identical concerns across
random/quality/diversity/combined, so they live here once rather than
being reimplemented per-selector.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def make_rng(seed: int) -> np.random.Generator:
    """Single source of RNG construction for the whole selection package.
    Never use np.random.* module-level functions (uncontrolled global
    state) -- always go through this so a given seed is fully
    reproducible regardless of call order or other code running in the
    same process.
    """
    return np.random.default_rng(seed)


def validate_budget(budget: int, total_available: int) -> None:
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget <= 0:
        raise ValueError(f"budget must be a positive integer, got {budget}.")
    if total_available <= 0:
        raise ValueError("No samples available to select from (empty pool).")


def allocate_budget_per_class(
    classes: list[str],
    budget: int,
    available_counts: dict[str, int],
) -> dict[str, int]:
    """Splits `budget` approximately evenly across `classes`, deterministically.

    Algorithm:
    1. base = budget // n_classes for every class.
    2. Remainder (budget % n_classes) is distributed one-by-one to classes
       in sorted-name order -- deterministic, no RNG involved, so the same
       (classes, budget) always produces the same allocation regardless of
       seed.
    3. Cap each class's allocation at its available sample count (never
       request more than exists in that class).
    4. Any shortfall created by step 3 (e.g. a class has fewer samples than
       its base+remainder share) is redistributed to classes that still
       have spare capacity, again in sorted-name order, one unit at a time,
       repeating until either the shortfall is fully redistributed or no
       class has spare capacity left (in which case the returned total is
       less than `budget` -- this is the documented behavior for an
       insufficient-samples pool, not an error).

    Returns: dict[class_name -> allocated_count]. sum(result.values()) <=
    budget, with equality unless the pool as a whole can't supply `budget`
    samples.
    """
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if len(classes) != len(set(classes)) or any(available_counts.get(c, 0) < 0 for c in classes):
        raise ValueError("Classes must be unique and capacities nonnegative")
    classes_sorted = sorted(classes)
    n_classes = len(classes_sorted)
    if n_classes == 0:
        return {}

    base = budget // n_classes
    remainder = budget % n_classes

    alloc = {c: base for c in classes_sorted}
    for c in classes_sorted[:remainder]:
        alloc[c] += 1

    # Cap at availability, track shortfall.
    shortfall = 0
    for c in classes_sorted:
        avail = available_counts.get(c, 0)
        if alloc[c] > avail:
            shortfall += alloc[c] - avail
            alloc[c] = avail

    # Redistribute shortfall to classes with spare capacity, deterministic
    # round-robin over sorted class names, until either shortfall is
    # exhausted or no class has room left.
    while shortfall > 0:
        progressed = False
        for c in classes_sorted:
            if shortfall <= 0:
                break
            avail = available_counts.get(c, 0)
            if alloc[c] < avail:
                alloc[c] += 1
                shortfall -= 1
                progressed = True
        if not progressed:
            # No class has any spare capacity left; the pool genuinely
            # cannot supply the full budget. Stop here rather than loop
            # forever.
            break

    return alloc


def build_embedding_lookup(
    metadata: pd.DataFrame,
    embedding_matrix: np.ndarray,
    embedding_index: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Builds a sample_id -> embedding vector dict for exactly the sample
    ids present in `metadata`. Built once per selection run and passed to
    selectors, so selectors never deal with row-index bookkeeping directly
    (that's Phase 1's concern, encapsulated here at the boundary).

    Raises KeyError-style ValueError if a metadata sample_id has no
    corresponding embedding row, since a silent gap here would corrupt
    every downstream selector.
    """
    if embedding_index["sample_id"].duplicated().any() or metadata["id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in metadata or embedding index")
    rows = embedding_index["row_index"].to_numpy()
    if (embedding_matrix.ndim != 2 or len(embedding_index) != len(embedding_matrix)
            or not np.issubdtype(rows.dtype, np.integer)
            or set(rows) != set(range(len(embedding_matrix)))):
        raise ValueError("Embedding index must map each matrix row exactly once")
    id_to_row = dict(zip(embedding_index["sample_id"], rows))
    lookup: dict[str, np.ndarray] = {}
    missing = []
    for sample_id in metadata["id"]:
        row_idx = id_to_row.get(sample_id)
        if row_idx is None:
            missing.append(sample_id)
            continue
        lookup[sample_id] = np.asarray(embedding_matrix[row_idx])
    if missing:
        raise ValueError(
            f"{len(missing)} metadata sample_ids have no matching embedding "
            f"row (e.g. {missing[:5]}). Metadata and embeddings are misaligned."
        )
    return lookup


def class_counts(selected_ids: list[str], metadata: pd.DataFrame) -> dict[str, int]:
    """Counts how many of `selected_ids` fall into each class, per the
    metadata's `label` column."""
    id_to_label = dict(zip(metadata["id"], metadata["label"]))
    counts: dict[str, int] = {}
    for sid in selected_ids:
        label = id_to_label.get(sid, "<unknown>")
        counts[label] = counts.get(label, 0) + 1
    return counts


def get_class_pools(metadata: pd.DataFrame) -> dict[str, list[str]]:
    """Returns {class_label: [sample_id, ...]} restricted to whatever rows
    are present in `metadata`. Selector.select validates training membership."""
    pools: dict[str, list[str]] = {}
    for label, group in metadata.groupby("label"):
        pools[label] = list(group["id"])
    return pools
