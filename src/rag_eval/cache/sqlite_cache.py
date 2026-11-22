"""SQLite-backed caches for embeddings, LLM prompts, and daily quota usage.

The embedding cache key is ``(content_sha256, model_id, dims)`` exactly as the
project rules require. Nothing here performs network I/O, which keeps the whole
module unit-testable offline.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_sha256 TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    dims           INTEGER NOT NULL,
    vector         BLOB NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (content_sha256, model_id, dims)
);

CREATE TABLE IF NOT EXISTS llm_responses (
    prompt_sha256 TEXT NOT NULL,
    target        TEXT NOT NULL,
    content       TEXT NOT NULL,
    provider_used TEXT NOT NULL,
    model_used    TEXT NOT NULL,
    usage_tokens  INTEGER NOT NULL,
    latency_ms    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (prompt_sha256, target)
);

CREATE TABLE IF NOT EXISTS quota_usage (
    provider    TEXT NOT NULL,
    window_date TEXT NOT NULL,
    used        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, window_date)
);
"""


def sha256_text(text: str) -> str:
    """Stable hash of arbitrary text, used as a cache key component."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic JSON so equivalent payloads hash identically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class CachedLLMResponse:
    content: str
    provider_used: str
    model_used: str
    usage_tokens: int
    latency_ms: int


class SqliteCache:
    """Thin, thread-safe wrapper around a single SQLite file."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SqliteCache:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- embeddings --------------------------------------------------------

    def get_embedding(self, content_sha256: str, model_id: str, dims: int) -> np.ndarray | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embedding_cache "
                "WHERE content_sha256 = ? AND model_id = ? AND dims = ?",
                (content_sha256, model_id, dims),
            ).fetchone()
        if row is None:
            return None
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        if vector.size != dims:
            # Corrupt or mismatched payload: treat as a miss rather than lying.
            return None
        return vector.copy()

    def put_embedding(
        self,
        content_sha256: str,
        model_id: str,
        dims: int,
        vector: Iterable[float] | np.ndarray,
    ) -> None:
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {array.shape}")
        if array.size != dims:
            raise ValueError(f"embedding has {array.size} dims, expected {dims}")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO embedding_cache "
                "(content_sha256, model_id, dims, vector, created_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(content_sha256, model_id, dims) DO UPDATE SET "
                "vector = excluded.vector, created_at = excluded.created_at",
                (content_sha256, model_id, dims, array.tobytes(), _utc_now()),
            )

    # -- LLM prompt cache --------------------------------------------------

    def get_llm_response(self, prompt_sha256: str, target: str) -> CachedLLMResponse | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT content, provider_used, model_used, usage_tokens, latency_ms "
                "FROM llm_responses WHERE prompt_sha256 = ? AND target = ?",
                (prompt_sha256, target),
            ).fetchone()
        if row is None:
            return None
        return CachedLLMResponse(
            content=row["content"],
            provider_used=row["provider_used"],
            model_used=row["model_used"],
            usage_tokens=row["usage_tokens"],
            latency_ms=row["latency_ms"],
        )

    def put_llm_response(
        self,
        prompt_sha256: str,
        target: str,
        *,
        content: str,
        provider_used: str,
        model_used: str,
        usage_tokens: int,
        latency_ms: int,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO llm_responses (prompt_sha256, target, content, provider_used, "
                "model_used, usage_tokens, latency_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(prompt_sha256, target) DO UPDATE SET "
                "content = excluded.content, provider_used = excluded.provider_used, "
                "model_used = excluded.model_used, usage_tokens = excluded.usage_tokens, "
                "latency_ms = excluded.latency_ms, created_at = excluded.created_at",
                (
                    prompt_sha256,
                    target,
                    content,
                    provider_used,
                    model_used,
                    usage_tokens,
                    latency_ms,
                    _utc_now(),
                ),
            )

    # -- daily quota -------------------------------------------------------

    def get_quota_used(self, provider: str, window_date: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT used FROM quota_usage WHERE provider = ? AND window_date = ?",
                (provider, window_date),
            ).fetchone()
        return int(row["used"]) if row is not None else 0

    def try_consume_quota(self, provider: str, window_date: str, limit: int | None) -> bool:
        """Atomically reserve one request against a provider's daily budget.

        Returns ``False`` without writing anything when the budget is exhausted,
        so callers can skip the provider instead of burning a failing request.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT used FROM quota_usage WHERE provider = ? AND window_date = ?",
                (provider, window_date),
            ).fetchone()
            used = int(row["used"]) if row is not None else 0
            if limit is not None and used >= limit:
                return False
            if row is None:
                self._conn.execute(
                    "INSERT INTO quota_usage (provider, window_date, used) VALUES (?, ?, 1)",
                    (provider, window_date),
                )
            else:
                self._conn.execute(
                    "UPDATE quota_usage SET used = used + 1 "
                    "WHERE provider = ? AND window_date = ?",
                    (provider, window_date),
                )
            return True

    def release_quota(self, provider: str, window_date: str) -> None:
        """Undo a reservation, e.g. when a request never left the process."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE quota_usage SET used = MAX(used - 1, 0) "
                "WHERE provider = ? AND window_date = ?",
                (provider, window_date),
            )

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict[str, int]:
        with self._lock:
            embeddings = self._conn.execute(
                "SELECT COUNT(*) AS n FROM embedding_cache"
            ).fetchone()["n"]
            llm = self._conn.execute("SELECT COUNT(*) AS n FROM llm_responses").fetchone()["n"]
            quota_rows = self._conn.execute("SELECT COUNT(*) AS n FROM quota_usage").fetchone()["n"]
        return {
            "embedding_cache_entries": int(embeddings),
            "llm_cache_entries": int(llm),
            "quota_rows": int(quota_rows),
        }

    def iter_quota(self) -> Iterator[tuple[str, str, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT provider, window_date, used FROM quota_usage "
                "ORDER BY provider, window_date"
            ).fetchall()
        for row in rows:
            yield row["provider"], row["window_date"], int(row["used"])

    def clear(self) -> None:
        with self._lock, self._conn:
            for table in ("embedding_cache", "llm_responses", "quota_usage"):
                self._conn.execute(f"DELETE FROM {table}")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
