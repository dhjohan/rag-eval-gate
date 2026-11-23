from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import EmbeddingsConfig, LLMConfig

PROVIDER_ENV_VARS = (
    "NVIDIA_API_KEY",
    "GEMINI_API_KEY",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "OPENROUTER_API_KEY",
    "JINA_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real .env/keys out of the offline test run."""
    for name in PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def cache() -> Iterator[SqliteCache]:
    with SqliteCache(":memory:") as instance:
        yield instance


@pytest.fixture
def provider() -> Callable[..., dict[str, Any]]:
    def _make(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "enabled": True,
            "base_url": "https://provider.test/v1",
            "model": "test/model",
            "api_key_env": "TEST_API_KEY",
            "rpm": 600,
            "rpd": 10_000,
            "rpd_timezone": "UTC",
            "structured_output": "json_schema",
            "timeout_s": 5,
        }
        base.update(overrides)
        return base

    return _make


@pytest.fixture
def llm_config() -> Callable[..., LLMConfig]:
    def _make(
        providers: dict[str, Any],
        order: list[str] | None = None,
        **overrides: Any,
    ) -> LLMConfig:
        payload: dict[str, Any] = {
            "order": order if order is not None else list(providers),
            "providers": providers,
            "max_retries": 1,
            "backoff_base_s": 0.01,
            "backoff_cap_s": 0.05,
        }
        payload.update(overrides)
        return LLMConfig.model_validate(payload)

    return _make


@pytest.fixture
def embedding_provider() -> Callable[..., dict[str, Any]]:
    def _make(**overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "enabled": True,
            "base_url": "https://embed.test/v1",
            "model": "gemini-embedding-2",
            "api_key_env": "TEST_API_KEY",
            "dims": 64,
            "task_type": "RETRIEVAL_DOCUMENT",
            "timeout_s": 5,
        }
        base.update(overrides)
        return base

    return _make


@pytest.fixture
def embeddings_config() -> Callable[..., EmbeddingsConfig]:
    def _make(
        providers: dict[str, Any],
        order: list[str] | None = None,
        fallback: bool = False,
    ) -> EmbeddingsConfig:
        return EmbeddingsConfig.model_validate(
            {
                "order": order if order is not None else list(providers),
                "providers": providers,
                "fallback_enabled": fallback,
            }
        )

    return _make
