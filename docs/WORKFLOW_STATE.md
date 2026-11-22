# Workflow State

## Phase 0 — Foundation: COMPLETE

Status date: 2026-11-22

Scope was limited to the foundation layer: project skeleton, typed provider
configuration, SQLite caches, free-tier provider clients, the `rag-eval doctor`
command, and an offline test suite. No corpus ingestion, chunking, retrieval,
metrics, or dashboards were implemented.

## Delivered

| Area | File | Notes |
| --- | --- | --- |
| Packaging | `pyproject.toml`, `uv.lock` | Python >= 3.11, hatchling, ruff + pytest |
| Config schema | `src/rag_eval/config.py` | Pydantic v2, frozen models, `extra="forbid"` |
| Provider catalog | `configs/providers.yaml` | Single source of truth for models and quotas |
| Cache | `src/rag_eval/cache/sqlite_cache.py` | Embedding cache, prompt cache, quota table |
| Throttling | `src/rag_eval/providers/rate_limit.py` | Token bucket, daily quota, jittered backoff |
| LLM router | `src/rag_eval/providers/llm.py` | Ordered fallback, structured output, cache |
| Embeddings | `src/rag_eval/providers/embeddings.py` | Gemini + Jina, cache-first, single vector space |
| Rerank | `src/rag_eval/providers/rerank.py` | Jina reranker client |
| CLI | `src/rag_eval/cli.py` | `doctor`, `cache-stats` |
| CI | `.github/workflows/ci.yml` | Offline gate + banned-dependency guard |
| Tests | `tests/` | 88 offline tests |

## Provider catalog verification (2026-11-22)

Every model ID below was checked against a live catalog, not documentation
prose:

- NVIDIA NIM: `GET https://integrate.api.nvidia.com/v1/models` (33 entries).
- Gemini: `aistudio.google.com/docs/models/*` plus the pricing free-tier table.
- Cloudflare: Workers AI model pages and the pricing paid-model list.
- OpenRouter: `GET https://openrouter.ai/api/v1/models`, filtered to zero price.
- Jina: `docs.jina.ai/llms.txt` (model roster and dimension ladder).

### Corrections applied to the earlier draft

| Earlier assumption | Verified reality | Resolution |
| --- | --- | --- |
| `meta/llama-3.3-70b-instruct` on NIM | Not in the live catalog | Removed |
| `moonshotai/kimi-k3.1-pro` as the NIM judge | Exists, but retired by request | Replaced with `nvidia/nemotron-3.5-super-vl-120b-a12b` |
| NIM also serves DeepSeek / GPT-OSS / Qwen | None of these appear in `/v1/models` | Not used |
| NIM limits are 30 RPM / 1000 RPD | Published as 40 RPM / 10,000 RPD | Config uses 35 / 9000 |
| `gemini-4-hydrogen` is paid-only | Free of charge | Kept in the chain |
| `gemini-4-carbon` / `gemini-4-neon` are free | Both paid-only | Excluded |
| Gemini accepts `reasoning_effort="none"` | Thinking cannot be disabled on Gemini 3/4 | Minimum `low` |
| `jina-embeddings-v5-text-small` truncates to 768 | Ladder is 32/64/128/256/512/1024 | Switched to `-nano` (native 768) |
| Embeddings use `gemini-embedding-001` | `gemini-embedding-2` is current, 128-3072 dims | Uses `gemini-embedding-2` at 768 |
| `jina-reranker-v3` | `jina-reranker-v3.5` is current | Upgraded |
| Cloudflare `glm-5.3-flash` / Kimi are free | Explicitly paid-only | Uses `@cf/openai/gpt-oss-120b` |

### Deliberately unverified

Gemini free-tier RPM/RPD values are **not published**. The values in
`configs/providers.yaml` (`4/20`, `4/20`, `12/500`) are marked `UNVERIFIED` in
comments and must be confirmed at
`https://aistudio.google.com/rate-limit?timeRange=last-28-days`.
Gemini limits are per **project** (not per key) and reset at midnight Pacific.

## Design decisions worth remembering

1. **Quota is enforced, not documented.** `quota_usage` is persisted per
   `(provider, window_date)` and checked *before* a request is spent. A provider
   that is out of budget is skipped, not failed.
2. **Structured output is capability-ranked.** `structured_output` is an enum
   (`json_schema` / `json_object` / `none`). For Pydantic requests the router
   tries `json_schema` providers first, then `json_object`, then prompt-embedded
   schema, so `nvidia_ultra` (no `response_format`) is never chosen over a
   schema-capable provider.
3. **The cache stores the model that actually answered.** `openrouter/free` may
   route to any free model, so `model_used` comes from the response body and is
   replayed on cache hits.
4. **Embedding fallback is opt-in.** Mixing providers would mix vector spaces,
   so `fallback_enabled: false` means only the first provider is used. When
   enabled, the *entire* batch is re-embedded by the fallback provider.
5. **Secrets never enter YAML.** Providers declare `api_key_env`; the CLI prints
   `yes`/`no`, never a key value.

## Verification evidence

```
$ python -m uv run ruff check src tests
All checks passed!

$ python -m uv run pytest -q
88 passed in 0.79s
```

The suite runs fully offline: every HTTP interaction goes through
`httpx.MockTransport`, and the Google SDK is replaced by an injected stub. No
provider environment variables are required.

Banned-dependency check (`uv pip list`): no `torch`, `onnxruntime`,
`transformers`, `sentence-transformers`, `spacy`, `docling`, `marker`,
`easyocr`, `ollama`, `ragas`, or `deepeval`. Vector math uses `numpy` only.

CLI smoke test (no credentials present):

```
$ python -m uv run rag-eval doctor
...
FAIL  no usable LLM provider        # exit 1, as intended
```

With `NVIDIA_API_KEY` and `CLOUDFLARE_ACCOUNT_ID` set, the same command exits 0
and reports `nvidia_super` / `nvidia_ultra` as `ready`.

## Tooling notes

- `uv` was absent from the machine; installed with
  `python -m pip install --user uv` (uv 0.13.9).
- `uv.lock` is committed so CI can run `uv sync --frozen`.

## Not done (deliberately)

- No corpus ingestion, PDF parsing, chunking, or index building.
- No retrieval metrics, golden QA generation, or LLM-as-judge evaluator.
- No `gemini-embedding-2` batch path: the client embeds one chunk per call
  because the model aggregates multi-input requests into a single vector.
- No live provider smoke test has been run, because no API keys are available in
  this environment.

## Next phase

Phase 1 should start with canonical document parsing and the citation-invariant
`(doc_id, page_num, evidence_quote, start_char, end_char)` contract, and must not
touch retrieval metrics until the golden QA schema exists.
