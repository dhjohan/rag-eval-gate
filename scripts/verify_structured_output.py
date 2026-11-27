"""Manual live check: which structured-output mode each LLM provider honours.

Run from the repo root in a terminal (never in CI or an agent loop)::

    python -m uv run python scripts/verify_structured_output.py --provider nvidia_super

It sends one request per mode (``json_schema`` then ``json_object``) using a
nested Pydantic model with ``Literal``, ``list[...]`` and ``extra="forbid"``,
passes no tools, and exits non-zero when no mode validates. Update
``configs/providers.yaml`` from the printed outcome.
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import LLMConfig, Settings, build_env, load_providers_config
from rag_eval.providers.llm import AllProvidersFailed, LLMRouter


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: str = Field(min_length=1)
    page: int = Field(ge=1)


class Verdict(BaseModel):
    """Nested schema exercising Literal, list[...] and extra="forbid"."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["pass", "fail"]
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(min_length=1)
    notes: str


PROMPT = (
    "Return a verdict for this claim: 'the document has 28 pages'. "
    "Use label='pass', confidence=0.9, one evidence item with quote='28 pages' "
    "and page=1, and notes='ok'."
)


def _single_provider_config(base: LLMConfig, provider: str, mode: str) -> LLMConfig:
    data = base.providers[provider].model_dump()
    data["structured_output"] = mode
    return LLMConfig.model_validate(
        {
            "order": [provider],
            "providers": {provider: data},
            "max_retries": 0,
            "backoff_base_s": 1.0,
            "backoff_cap_s": 5.0,
            "prompt_cache_enabled": False,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--config", default="configs/providers.yaml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--modes",
        default="json_schema,json_object",
        help="Comma-separated structured-output modes to try in order.",
    )
    args = parser.parse_args()

    settings = Settings(_env_file=args.env_file)
    base = load_providers_config(args.config, env=build_env(settings))
    if args.provider not in base.llm.providers:
        print(f"unknown provider {args.provider!r}; available: {sorted(base.llm.providers)}")
        return 2
    cfg = base.llm.providers[args.provider]
    if not cfg.is_configured:
        missing = ", ".join(cfg.unresolved_env) or "api key"
        print(f"FAIL {args.provider}: not configured ({missing})")
        return 2

    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        config = _single_provider_config(base, args.provider, mode)
        try:
            with SqliteCache(":memory:") as cache, LLMRouter.from_settings(
                config, cache, settings
            ) as router:
                result = router.generate(
                    [{"role": "user", "content": PROMPT}],
                    response_model=Verdict,
                    use_cache=False,
                )
        except AllProvidersFailed as exc:
            failure = exc.failures[0] if exc.failures else None
            status = failure.status_code if failure else None
            print(
                f"FAIL {args.provider} {mode}: "
                f"{failure.reason if failure else exc}"
                + (f" ({status})" if status else "")
            )
            # A 400/422 is a capability rejection: try the next mode. Anything
            # else (auth, quota, transport) means this run proved nothing.
            if status in (400, 422):
                continue
            return 1
        print(
            f"OK   {args.provider} {mode}: "
            f"provider={result.provider_used} model={result.model_used}"
        )
        print(f"     content={result.content[:300]!r}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
