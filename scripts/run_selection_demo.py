#!/usr/bin/env python
"""
Runs all four Phase 2 selection strategies (random, quality, diversity,
combined) across a set of budgets on the TRAIN split only, and persists
selected sample IDs + reproducibility metadata.

Validation/test are never loaded by this script -- only
config.paths.metadata_dir/train.parquet and embeddings_dir/train.npy are
read, so there is no code path here through which val/test could
participate in selection.

Usage:
    python scripts/run_selection_demo.py --config configs/experiment.yaml
    python scripts/run_selection_demo.py --config configs/experiment.yaml \
        --budgets 100 200 400 800 --seed 0 --alpha 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aura.config import AuraConfig
from aura.ingest.metadata import load_metadata_parquet, load_embedding_matrix
from aura.selection.utils import build_embedding_lookup
from aura.selection.random import RandomSelector
from aura.selection.quality import QualitySelector
from aura.selection.diversity import DiversitySelector
from aura.selection.combined import CombinedSelector


def print_class_counts(title: str, class_counts: dict[str, int]) -> None:
    print(f"  {title}:")
    for label in sorted(class_counts.keys()):
        print(f"    {label:>8s}: {class_counts[label]}")


def main():
    parser = argparse.ArgumentParser(description="AURA Phase 2 selection demo.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 400, 800])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight on quality in the combined (AURA) strategy.")
    parser.add_argument(
        "--output-dir", type=str, default="results/selection",
        help="Where to write per-run selection JSON files.",
    )
    args = parser.parse_args()

    config = AuraConfig.from_yaml(args.config)

    train_metadata_path = Path(config.paths.metadata_dir) / "train.parquet"
    train_matrix_path = Path(config.paths.embeddings_dir) / "train.npy"
    train_index_path = Path(config.paths.embeddings_dir) / "train_index.parquet"

    if not train_metadata_path.exists():
        raise FileNotFoundError(
            f"Train metadata not found at {train_metadata_path}. "
            "Run scripts/prepare_data.py first (Phase 1)."
        )

    print(f"Loading train metadata from {train_metadata_path}")
    metadata = load_metadata_parquet(train_metadata_path)
    metadata = metadata[metadata["split"] == "train"].reset_index(drop=True)
    print(f"Loaded {len(metadata)} train samples across {metadata['label'].nunique()} classes")

    print(f"Loading train embeddings from {train_matrix_path}")
    matrix, index_df = load_embedding_matrix(train_matrix_path, train_index_path)
    embeddings = build_embedding_lookup(metadata, matrix, index_df)
    print(f"Embedding lookup built: {len(embeddings)} vectors, dim={matrix.shape[1]}")

    selectors = {
        "random": RandomSelector(),
        "quality": QualitySelector(),
        "diversity": DiversitySelector(),
        "combined": CombinedSelector(alpha=args.alpha),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for budget in args.budgets:
        print(f"\n{'=' * 60}\nBUDGET: {budget} (seed={args.seed})\n{'=' * 60}")
        for strategy_name, selector in selectors.items():
            result = selector.select(metadata, embeddings, budget=budget, seed=args.seed)

            print(f"\n[{strategy_name}]")
            print(f"  requested_budget: {result.requested_budget}")
            print(f"  actual_budget:    {result.actual_budget}")
            if result.actual_budget != result.requested_budget:
                print(f"  NOTE: actual < requested -- pool could not fully supply this budget.")
            if result.alpha is not None:
                print(f"  alpha: {result.alpha}")
            print_class_counts("class_counts", result.class_counts)

            n_unique = len(set(result.selected_ids))
            print(f"  unique_ids: {n_unique} (== actual_budget: {n_unique == result.actual_budget})")

            out_path = output_dir / f"{strategy_name}_budget{budget}_seed{args.seed}.json"
            with open(out_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"  saved: {out_path}")

    print("\nDone. Validation/test were never loaded by this script.")


if __name__ == "__main__":
    main()