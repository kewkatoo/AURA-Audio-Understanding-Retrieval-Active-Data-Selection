from __future__ import annotations

import numpy as np

from aura.cache.store import (
    CacheStore, file_content_hash, config_fingerprint, make_cache_key,
)


def test_file_content_hash_stable_for_identical_content(tmp_path):
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"  # different filename, same bytes
    p1.write_bytes(b"hello world")
    p2.write_bytes(b"hello world")
    assert file_content_hash(p1) == file_content_hash(p2)


def test_file_content_hash_differs_for_different_content(tmp_path):
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"hello world")
    p2.write_bytes(b"goodbye world")
    assert file_content_hash(p1) != file_content_hash(p2)


def test_config_fingerprint_order_independent():
    a = config_fingerprint({"x": 1, "y": 2})
    b = config_fingerprint({"y": 2, "x": 1})
    assert a == b


def test_config_fingerprint_differs_on_value_change():
    a = config_fingerprint({"x": 1})
    b = config_fingerprint({"x": 2})
    assert a != b


def test_make_cache_key_includes_namespace():
    k1 = make_cache_key("audiohash", "confighash", namespace="features")
    k2 = make_cache_key("audiohash", "confighash", namespace="embeddings")
    assert k1 != k2


def test_cache_store_hit_miss_cycle(tmp_path):
    store = CacheStore(tmp_path / "cache_ns")
    key = "somekey"
    assert store.get(key) is None
    assert key not in store

    calls = {"n": 0}
    def compute():
        calls["n"] += 1
        return np.array([1.0, 2.0, 3.0])

    value1, hit1 = store.get_or_compute(key, compute)
    assert hit1 is False
    assert calls["n"] == 1

    value2, hit2 = store.get_or_compute(key, compute)
    assert hit2 is True
    assert calls["n"] == 1  # not recomputed
    assert np.array_equal(value1, value2)
    store.close()


def test_cache_store_persists_across_instances(tmp_path):
    cache_dir = tmp_path / "cache_ns"
    store1 = CacheStore(cache_dir)
    store1.set("k", {"a": 1})
    store1.close()

    store2 = CacheStore(cache_dir)
    assert store2.get("k") == {"a": 1}
    store2.close()
