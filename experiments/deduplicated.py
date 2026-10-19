"""Content-uniform random control, with exact per-class budgets."""
import numpy as np

from aura.selection.random import RandomSelector


class DeduplicatedRandomSelector(RandomSelector):
    name = "deduplicated_random"

    def select(self, metadata, embeddings, budget, seed, **kwargs):
        if "audio_hash" not in metadata or metadata.audio_hash.isnull().any():
            raise ValueError("Content hashes are required")
        if metadata.groupby("audio_hash").label.nunique().gt(1).any():
            raise ValueError("Identical content has conflicting labels")
        # Validate the original pool before removing copies (including split/ID checks).
        if metadata.id.duplicated().any() or not metadata.split.eq("train").all():
            raise ValueError("Unique training IDs are required")
        unique = metadata.sort_values("id").drop_duplicates("audio_hash")
        if budget > len(unique):
            raise ValueError("Budget exceeds the unique-content pool")
        return super().select(unique, embeddings, budget, seed, **kwargs)
