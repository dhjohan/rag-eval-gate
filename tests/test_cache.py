from __future__ import annotations

import numpy as np
import pytest

from rag_eval.cache.sqlite_cache import SqliteCache, canonical_json, sha256_text


def test_embedding_roundtrip(cache: SqliteCache) -> None:
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    digest = sha256_text("hello")
    cache.put_embedding(digest, "model-a", 3, vector)

    stored = cache.get_embedding(digest, "model-a", 3)

    assert stored is not None
    np.testing.assert_allclose(stored, vector)


def test_embedding_miss_on_different_dims(cache: SqliteCache) -> None:
    cache.put_embedding(sha256_text("hello"), "model-a", 3, [0.1, 0.2, 0.3])
    assert cache.get_embedding(sha256_text("hello"), "model-a", 4) is None


def test_embedding_miss_on_different_model(cache: SqliteCache) -> None:
    cache.put_embedding(sha256_text("hello"), "model-a", 3, [0.1, 0.2, 0.3])
    assert cache.get_embedding(sha256_text("hello"), "model-b", 3) is None


def test_put_embedding_rejects_wrong_dimensions(cache: SqliteCache) -> None:
    with pytest.raises(ValueError, match="expected 4"):
        cache.put_embedding(sha256_text("x"), "model-a", 4, [0.1, 0.2, 0.3])


def test_put_embedding_rejects_two_dimensional_input(cache: SqliteCache) -> None:
    with pytest.raises(ValueError, match="1-D"):
        cache.put_embedding(sha256_text("x"), "model-a", 3, [[0.1, 0.2, 0.3]])


def test_llm_response_roundtrip_preserves_actual_model(cache: SqliteCache) -> None:
    digest = sha256_text("prompt")
    cache.put_llm_response(
        digest,
        "Verdict",
        content='{"score": 1}',
        provider_used="openrouter_free",
        model_used="nvidia/nemotron-3.5-super-vl:free",
        usage_tokens=42,
        latency_ms=900,
    )

    cached = cache.get_llm_response(digest, "Verdict")

    assert cached is not None
    assert cached.model_used == "nvidia/nemotron-3.5-super-vl:free"
    assert cached.provider_used == "openrouter_free"
    assert cached.usage_tokens == 42
    assert cached.latency_ms == 900


def test_llm_response_is_scoped_by_target(cache: SqliteCache) -> None:
    digest = sha256_text("prompt")
    cache.put_llm_response(
        digest,
        "Verdict",
        content="{}",
        provider_used="a",
        model_used="m",
        usage_tokens=1,
        latency_ms=1,
    )
    assert cache.get_llm_response(digest, "freeform") is None


def test_quota_stops_at_limit(cache: SqliteCache) -> None:
    assert cache.try_consume_quota("gemini_lite", "2026-11-22", 2) is True
    assert cache.try_consume_quota("gemini_lite", "2026-11-22", 2) is True
    assert cache.try_consume_quota("gemini_lite", "2026-11-22", 2) is False
    assert cache.get_quota_used("gemini_lite", "2026-11-22") == 2


def test_quota_is_partitioned_by_window_date(cache: SqliteCache) -> None:
    cache.try_consume_quota("p", "2026-11-22", 1)
    assert cache.try_consume_quota("p", "2026-11-22", 1) is False
    assert cache.try_consume_quota("p", "2026-11-23", 1) is True


def test_unlimited_quota_never_blocks(cache: SqliteCache) -> None:
    for _ in range(50):
        assert cache.try_consume_quota("p", "2026-11-22", None) is True
    assert cache.get_quota_used("p", "2026-11-22") == 50


def test_release_quota_floors_at_zero(cache: SqliteCache) -> None:
    cache.release_quota("p", "2026-11-22")
    assert cache.get_quota_used("p", "2026-11-22") == 0
    cache.try_consume_quota("p", "2026-11-22", 5)
    cache.release_quota("p", "2026-11-22")
    assert cache.get_quota_used("p", "2026-11-22") == 0


def test_stats_counts_each_table(cache: SqliteCache) -> None:
    cache.put_embedding(sha256_text("a"), "m", 2, [1.0, 0.0])
    cache.put_llm_response(
        sha256_text("b"),
        "freeform",
        content="x",
        provider_used="p",
        model_used="m",
        usage_tokens=1,
        latency_ms=1,
    )
    cache.try_consume_quota("p", "2026-11-22", 5)

    stats = cache.stats()

    assert stats["embedding_cache_entries"] == 1
    assert stats["llm_cache_entries"] == 1
    assert stats["quota_rows"] == 1


def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_sha256_text_is_stable() -> None:
    assert sha256_text("hello") == sha256_text("hello")
    assert sha256_text("hello") != sha256_text("hello ")
