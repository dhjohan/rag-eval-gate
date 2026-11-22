from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag_eval.config import (
    ConfigError,
    LLMConfig,
    expand_env,
    load_providers_config,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "providers.yaml"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "providers.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_shipped_config_matches_verified_provider_catalog() -> None:
    config = load_providers_config(CONFIG_PATH, env={"CLOUDFLARE_ACCOUNT_ID": "acct123"})

    # Chain is NVIDIA -> Cloudflare -> OpenRouter; Gemini chat models were
    # dropped because their free-tier quotas are too strict / unpublished.
    assert config.llm.order == [
        "nvidia_super",
        "nvidia_ultra",
        "cloudflare_gpt_oss",
        "openrouter_free",
    ]
    assert config.llm.providers["nvidia_super"].model == "nvidia/nemotron-3.5-super-vl-120b-a12b"
    assert config.llm.providers["nvidia_ultra"].structured_output == "none"
    assert config.llm.providers["cloudflare_gpt_oss"].model == "@cf/openai/gpt-oss-120b"
    assert config.llm.providers["openrouter_free"].model == "openrouter/free"

    # Gemini chat providers must not be in the active chain.
    assert not any(name.startswith("gemini_") for name in config.llm.providers)

    # The retired Kimi entry must not creep back in.
    assert not any("kimi" in cfg.model.lower() for cfg in config.llm.providers.values())


def test_shipped_config_resolves_cloudflare_account_id() -> None:
    config = load_providers_config(CONFIG_PATH, env={"CLOUDFLARE_ACCOUNT_ID": "acct123"})
    cloudflare = config.llm.providers["cloudflare_gpt_oss"]
    assert cloudflare.base_url == (
        "https://api.cloudflare.com/client/v4/accounts/acct123/ai/v1"
    )
    assert cloudflare.is_configured is True


def test_unresolved_env_is_flagged_not_fatal() -> None:
    config = load_providers_config(CONFIG_PATH, env={})
    cloudflare = config.llm.providers["cloudflare_gpt_oss"]
    assert cloudflare.unresolved_env == ("CLOUDFLARE_ACCOUNT_ID",)
    assert cloudflare.is_configured is False
    assert "${CLOUDFLARE_ACCOUNT_ID}" in cloudflare.base_url


def test_embedding_models_are_768_dimension_safe() -> None:
    config = load_providers_config(CONFIG_PATH, env={"CLOUDFLARE_ACCOUNT_ID": "x"})
    google = config.embeddings.providers["google"]
    jina = config.embeddings.providers["jina"]
    assert google.model == "gemini-embedding-2"
    assert google.dims == 768
    # v5-text-small cannot be truncated to 768; nano is natively 768.
    assert jina.model == "jina-embeddings-v5-text-nano"
    assert jina.dims == 768


def test_rerank_uses_current_model() -> None:
    config = load_providers_config(CONFIG_PATH, env={"CLOUDFLARE_ACCOUNT_ID": "x"})
    assert config.rerank.model == "jina-reranker-v3.5"


def test_embedding_fallback_is_off_by_default() -> None:
    config = load_providers_config(CONFIG_PATH, env={"CLOUDFLARE_ACCOUNT_ID": "x"})
    assert config.embeddings.fallback_enabled is False
    assert config.embeddings.ordered_providers == ["google"]


def test_expand_env_reports_missing_variables() -> None:
    value, missing = expand_env("https://x/${A}/y/${B}", {"A": "1"})
    assert value == "https://x/1/y/${B}"
    assert missing == ("B",)


def test_order_referencing_unknown_provider_is_rejected(provider: Any) -> None:
    with pytest.raises(ValueError, match="unknown providers"):
        LLMConfig.model_validate(
            {"order": ["a", "ghost"], "providers": {"a": provider()}}
        )


def test_duplicate_order_entries_are_rejected(provider: Any) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        LLMConfig.model_validate(
            {"order": ["a", "a"], "providers": {"a": provider()}}
        )


def test_enabled_provider_missing_from_order_is_rejected(provider: Any) -> None:
    with pytest.raises(ValueError, match=r"missing from llm\.order"):
        LLMConfig.model_validate(
            {"order": ["a"], "providers": {"a": provider(), "b": provider()}}
        )


def test_non_https_base_url_is_rejected(provider: Any) -> None:
    with pytest.raises(ValueError, match="https"):
        LLMConfig.model_validate(
            {"order": ["a"], "providers": {"a": provider(base_url="http://insecure.test/v1")}}
        )


def test_reasoning_effort_requires_capability(provider: Any) -> None:
    with pytest.raises(ValueError, match="supports_reasoning"):
        LLMConfig.model_validate(
            {
                "order": ["a"],
                "providers": {"a": provider(reasoning_effort="low", supports_reasoning=False)},
            }
        )


def test_unknown_timezone_is_rejected(provider: Any) -> None:
    with pytest.raises(ValueError, match="timezone"):
        LLMConfig.model_validate(
            {"order": ["a"], "providers": {"a": provider(rpd_timezone="Mars/Olympus")}}
        )


def test_unknown_structured_output_mode_is_rejected(provider: Any) -> None:
    with pytest.raises(ValueError, match="structured_output"):
        LLMConfig.model_validate(
            {"order": ["a"], "providers": {"a": provider(structured_output="yaml")}}
        )


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_providers_config(tmp_path / "absent.yaml", env={})


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "llm: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_providers_config(path, env={})
