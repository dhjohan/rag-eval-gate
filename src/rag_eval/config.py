"""Typed configuration for free-tier providers.

The YAML file is the source of truth for *which* models exist and how they are
rate limited. Secrets never live in YAML: every provider names an environment
variable via ``api_key_env`` and the value is resolved at call time.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("configs/providers.yaml")
DEFAULT_CACHE_PATH = Path(".cache/rag_eval.sqlite3")

ENV_PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

StructuredOutput = Literal["json_schema", "json_object", "none"]
REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})

_API_KEY_PATTERN = r"^[A-Z][A-Z0-9_]*$"


class ConfigError(ValueError):
    """Raised when the provider configuration is missing or invalid."""


def expand_env(value: str, env: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    """Substitute ``${VAR}`` references, reporting any that are unresolved."""
    missing: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = env.get(name, "").strip()
        if resolved:
            return resolved
        missing.append(name)
        return match.group(0)

    expanded = ENV_PLACEHOLDER.sub(_replace, value)
    return expanded, tuple(dict.fromkeys(missing))


def _expand_tree(obj: Any, env: Mapping[str, str]) -> tuple[Any, tuple[str, ...]]:
    if isinstance(obj, str):
        return expand_env(obj, env)
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        missing: list[str] = []
        for key, value in obj.items():
            expanded, found = _expand_tree(value, env)
            out[key] = expanded
            missing.extend(found)
        return out, tuple(dict.fromkeys(missing))
    if isinstance(obj, list):
        items: list[Any] = []
        missing = []
        for value in obj:
            expanded, found = _expand_tree(value, env)
            items.append(expanded)
            missing.extend(found)
        return items, tuple(dict.fromkeys(missing))
    return obj, ()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _HttpProvider(_StrictModel):
    """Shared URL handling for anything that speaks HTTP."""

    base_url: str
    api_key_env: str = Field(pattern=_API_KEY_PATTERN)
    timeout_s: float = Field(default=60.0, gt=0, le=600)
    unresolved_env: tuple[str, ...] = ()

    @field_validator("base_url")
    @classmethod
    def _normalise_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            raise ValueError("base_url must not be empty")
        return cleaned

    @model_validator(mode="after")
    def _require_https_when_resolved(self) -> _HttpProvider:
        if not self.unresolved_env and not self.base_url.startswith("https://"):
            raise ValueError(f"base_url must be an https URL, got {self.base_url!r}")
        return self

    @property
    def is_configured(self) -> bool:
        """True when every ``${VAR}`` resolved and the URL is usable."""
        return not self.unresolved_env and self.base_url.startswith("https://")


class LLMProviderConfig(_HttpProvider):
    """One entry in the ordered LLM fallback chain."""

    enabled: bool = True
    model: str = Field(min_length=1)
    rpm: int | None = Field(default=None, ge=1)
    rpd: int | None = Field(default=None, ge=1)
    rpd_timezone: str = "UTC"
    structured_output: StructuredOutput = "none"
    supports_reasoning: bool = False
    reasoning_effort: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("rpd_timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        # Imported lazily so the error message stays readable on Windows.
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown rpd_timezone {value!r}") from exc
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def _known_reasoning_effort(cls, value: str | None) -> str | None:
        if value is not None and value not in REASONING_EFFORTS:
            allowed = ", ".join(sorted(REASONING_EFFORTS))
            raise ValueError(f"reasoning_effort must be one of: {allowed}")
        return value

    @model_validator(mode="after")
    def _reasoning_consistency(self) -> LLMProviderConfig:
        if self.reasoning_effort is not None and not self.supports_reasoning:
            raise ValueError("reasoning_effort is set but supports_reasoning is false")
        return self


class LLMConfig(_StrictModel):
    order: list[str] = Field(min_length=1)
    providers: dict[str, LLMProviderConfig] = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0, le=5)
    backoff_base_s: float = Field(default=1.0, gt=0)
    backoff_cap_s: float = Field(default=30.0, gt=0)
    prompt_cache_enabled: bool = True

    @model_validator(mode="after")
    def _chain_is_coherent(self) -> LLMConfig:
        if len(set(self.order)) != len(self.order):
            raise ValueError("llm.order contains duplicate provider names")
        unknown = [name for name in self.order if name not in self.providers]
        if unknown:
            raise ValueError(f"llm.order references unknown providers: {unknown}")
        orphans = [
            name
            for name, provider in self.providers.items()
            if provider.enabled and name not in self.order
        ]
        if orphans:
            raise ValueError(f"enabled providers missing from llm.order: {orphans}")
        if not self.ordered_providers:
            raise ValueError("no enabled llm providers")
        return self

    @property
    def ordered_providers(self) -> list[str]:
        """Enabled providers in configured fallback order."""
        return [name for name in self.order if self.providers[name].enabled]


class EmbeddingProviderConfig(_HttpProvider):
    enabled: bool = True
    model: str = Field(min_length=1)
    dims: int = Field(default=768, ge=64, le=3072)
    task_type: str | None = None


class EmbeddingsConfig(_StrictModel):
    order: list[str] = Field(min_length=1)
    providers: dict[str, EmbeddingProviderConfig] = Field(min_length=1)
    fallback_enabled: bool = False

    @model_validator(mode="after")
    def _chain_is_coherent(self) -> EmbeddingsConfig:
        if len(set(self.order)) != len(self.order):
            raise ValueError("embeddings.order contains duplicate provider names")
        unknown = [name for name in self.order if name not in self.providers]
        if unknown:
            raise ValueError(f"embeddings.order references unknown providers: {unknown}")
        orphans = [
            name
            for name, provider in self.providers.items()
            if provider.enabled and name not in self.order
        ]
        if orphans:
            raise ValueError(f"enabled providers missing from embeddings.order: {orphans}")
        if not self.ordered_providers:
            raise ValueError("no enabled embedding providers")
        return self

    @property
    def ordered_providers(self) -> list[str]:
        """Providers to try. Fallback is opt-in so vector spaces never mix silently."""
        enabled = [name for name in self.order if self.providers[name].enabled]
        return enabled if self.fallback_enabled else enabled[:1]


class RerankConfig(_HttpProvider):
    enabled: bool = True
    model: str = Field(default="jina-reranker-v3.5", min_length=1)
    top_n: int = Field(default=10, ge=1, le=1000)


class ProvidersConfig(_StrictModel):
    schema_version: int = 1
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    rerank: RerankConfig


class Settings(BaseSettings):
    """Process settings. Secrets come from the environment or ``.env`` only."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    nvidia_api_key: str | None = None
    gemini_api_key: str | None = None
    cloudflare_api_token: str | None = None
    cloudflare_account_id: str | None = None
    openrouter_api_key: str | None = None
    jina_api_key: str | None = None

    rag_eval_cache_path: Path = DEFAULT_CACHE_PATH
    rag_eval_providers_config: Path = DEFAULT_CONFIG_PATH

    def env_map(self) -> dict[str, str]:
        """Expose settings as ``ENV_NAME -> value`` for ``${VAR}`` expansion."""
        out: dict[str, str] = {}
        for name in type(self).model_fields:
            if name.startswith("rag_eval_"):
                continue
            value = getattr(self, name, None)
            if isinstance(value, str) and value.strip():
                out[name.upper()] = value.strip()
        return out


def build_env(
    settings: Settings | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Merge ``.env``-derived settings with the real environment.

    Real environment variables win, so CI and shell overrides always take
    precedence over a stale local ``.env``.
    """
    base = dict(environ if environ is not None else os.environ)
    if settings is None:
        return base
    return {**settings.env_map(), **base}


def resolve_api_key(
    env_name: str,
    settings: Settings | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a provider key, returning ``None`` when it is absent or blank."""
    env = environ if environ is not None else os.environ
    value = env.get(env_name)
    if value is not None and value.strip():
        return value.strip()
    if settings is not None:
        fallback = getattr(settings, env_name.lower(), None)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return None


def _expand_provider_map(block: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    providers = block.get("providers")
    if not isinstance(providers, dict):
        return block
    expanded_providers: dict[str, Any] = {}
    for name, provider in providers.items():
        if not isinstance(provider, dict):
            expanded_providers[name] = provider
            continue
        expanded, missing = _expand_tree(provider, env)
        expanded["unresolved_env"] = list(missing)
        expanded_providers[name] = expanded
    return {**block, "providers": expanded_providers}


def _expand_rerank(block: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    expanded, missing = _expand_tree(block, env)
    expanded["unresolved_env"] = list(missing)
    return expanded


def load_providers_config(
    path: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ProvidersConfig:
    """Load and validate ``providers.yaml``.

    Providers whose ``${VAR}`` placeholders cannot be resolved are kept in the
    model but flagged via ``unresolved_env`` so callers can report them instead
    of crashing on an unrelated provider.
    """
    resolved_env = dict(os.environ) if env is None else dict(env)
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"provider config not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via a temp file
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    prepared = dict(raw)
    for section in ("llm", "embeddings"):
        block = prepared.get(section)
        if isinstance(block, dict):
            prepared[section] = _expand_provider_map(block, resolved_env)
    if isinstance(prepared.get("rerank"), dict):
        prepared["rerank"] = _expand_rerank(prepared["rerank"], resolved_env)

    try:
        return ProvidersConfig.model_validate(prepared)
    except ValidationError as exc:
        raise ConfigError(f"invalid provider config {config_path}:\n{exc}") from exc
