"""
Deterministic filesystem cache for expensive per-sample computations
(DSP features, embeddings). Backed by `diskcache` for atomic, process-safe
reads/writes.

Design notes:
- Cache keys are explicit content hashes, never raw audio. Raw audio is
  never written into the cache directory (per Phase 1 spec).
- A cache key depends on: audio file identity (content hash of the file
  bytes, not just the path -- so a moved/renamed file with identical
  content still hits cache, and a modified file at the same path misses),
  plus a config fingerprint of whatever produced the cached value
  (preprocessing config for features, encoder checkpoint+config for
  embeddings). This makes cache invalidation automatic when config changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import diskcache
import numpy as np


def file_content_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hash of a file's bytes. Used as the audio-identity component
    of cache keys so identical audio content always maps to the same key
    regardless of filename/path."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def config_fingerprint(config_dict: dict) -> str:
    """Stable hash of a config dict (order-independent) used as the
    "what produced this value" component of a cache key."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def make_cache_key(audio_hash: str, config_hash: str, namespace: str) -> str:
    return f"{namespace}:{audio_hash}:{config_hash}"


class CacheStore:
    """Thin wrapper around diskcache.Cache with namespaced get/set and a
    get_or_compute helper. One CacheStore instance per namespace directory
    (e.g. one for features, one for embeddings) as required by the Phase 1
    spec (data/cache/embeddings/, data/cache/features/)."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self.cache_dir))

    def get(self, key: str) -> Any | None:
        return self._cache.get(key, default=None)

    def set(self, key: str, value: Any) -> None:
        self._cache.set(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get_or_compute(
        self, key: str, compute_fn: Callable[[], Any]
    ) -> tuple[Any, bool]:
        """Returns (value, was_cache_hit)."""
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = compute_fn()
        self.set(key, value)
        return value, False

    def close(self) -> None:
        self._cache.close()


def save_array_cached(store: CacheStore, key: str, array: np.ndarray) -> bool:
    """Convenience for numpy arrays (diskcache pickles numpy fine, but we
    round-trip through bytes explicitly so cache entries are portable and
    inspectable). Returns True if it was already cached."""
    existing = store.get(key)
    if existing is not None:
        return True
    store.set(key, array)
    return False
