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
| CI | `.github/workflows/ci.yml` | Offline gate + banned-dependency guard; mypy type check on push/PR |
| Tests | `tests/` | 187 offline tests |

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
3. **The prompt cache is keyed by the requested model, not the routed one.**
   `(prompt_sha256, target, provider_used, requested_model)` identifies an entry,
   while `model_used` records which model actually answered and is replayed on a
   hit. This keeps `openrouter/free` cacheable without letting a configured
   model change reuse stale output. Databases created before `requested_model`
   existed are migrated in place on open.
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
96 passed in 0.86s
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

## Phase 1 — Sprint 2A: corpus manifest & PDF acquisition: COMPLETE

Status date: 2026-11-22

Scope was limited to *getting real PDFs onto disk and proving what they are*.
No PDF parsing, chunking, retrieval, metrics, or golden QA work was started.

FDA's Excel export lists documents but does not expose raw PDF URLs, so a
config-only downloader would have nothing to download. Sprint 2A therefore ships
a local-first scanner that turns hand-downloaded PDFs into manifest entries.

## Delivered

| Area | File | Notes |
| --- | --- | --- |
| Corpus schema | `configs/corpus.yaml` | `documents: []`, commented placeholder example, `example.invalid` URL only |
| Manifest schema + scanner | `src/rag_eval/ingest/manifest.py` | Pydantic models, YAML renderer, `scan_local_pdfs`, PDF verification |
| Downloader | `src/rag_eval/ingest/downloader.py` | Local-first, atomic `.tmp` publish, tenacity retries, always-written manifest |
| Package exports | `src/rag_eval/ingest/__init__.py` | Public surface only |
| CLI | `src/rag_eval/cli.py` | `corpus scan-local`, `corpus download`; Rich summary tables |
| Tests | `tests/test_downloader.py` | 29 offline cases, `httpx.MockTransport` + `tmp_path` |
| Dependencies | `pyproject.toml`, `uv.lock` | `rich` and `tenacity` promoted from transitive to direct; `mypy` and `types-PyYAML` added as dev deps |

## Code review fixes applied (post-Sprint 2A)

The full code review (10 correctness, 3 efficiency, 4 style/tooling fixes) was applied to both Phase 0 and Sprint 2A:

| # | Area | Fix |
|---|------|-----|
| C1 | `providers/embeddings.py` | Wrap `response.json()` / `item["embedding"]` in `EmbeddingError` so a non-JSON 200 triggers retry/fallback instead of crashing |
| C2 | `providers/rerank.py` | Wrap `response.json()` / malformed payload in `RerankError` |
| C3 | `ingest/manifest.py` | Case-insensitive PDF discovery (`iterdir()` + `suffix.lower() == ".pdf"`) so `.PDF` is found on Linux CI as well as Windows |
| C4 | `providers/llm.py` | Include effective `cfg.max_tokens` / `cfg.reasoning_effort` in `prompt_hash` so config changes don't replay stale cache |
| C5 | `providers/llm.py` | `_parse_structured` salvage via `json.JSONDecoder().raw_decode` (handles trailing prose braces) |
| C6 | `providers/llm.py` | Release quota on 429 before retrying (one throttled call no longer burns N daily slots) |
| C7 | `cache/sqlite_cache.py` | `try_consume_quota` is now a single atomic upsert guarded by `WHERE used < ?` |
| C8 | `providers/embeddings.py` | Reuse one backend across retries; per-item `on_vector` callback caches partial progress on mid-batch failure |
| C9 | `providers/embeddings.py` | Deduplicate duplicate texts within a batch before calling the backend |
| C10 | `providers/llm.py` | Replace `\x3c`/`\x3e` regex with literal `<think>` / `</think>` tags |
| E1 | `providers/embeddings.py` | Backend built once per provider call, reused across retries (no repeated TLS handshake) |
| E2 | `providers/embeddings.py` | Duplicate batch inputs embedded once |
| E3 | `ingest/downloader.py` | Lazy HTTP client (zero sockets on local-only runs); `%PDF-` checked on the first streamed chunk; hashed while streaming |
| S1 | `ingest/manifest.py` | `try/finally` `.tmp` cleanup in `write_corpus_config` / `write_manifest` |
| S2 | `providers/llm.py` | `google-genai` client created with configured `timeout_s` |
| S3 | `tests/` | Moved `RecordingBackend` + `test_embedding_retry_after_transport_error` into `tests/test_embeddings.py`; deduped the copy in `test_llm_router.py` |
| S4 | `ci.yml` | Added `mypy` type-check step; Python pinned to 3.12 |

## Follow-up `llm.py` review fixes

A second, focused review of `providers/llm.py` found edge cases and one broken regex. All are fixed:

| # | Area | Fix |
|---|------|-----|
| L1 | `_parse_structured` | **Critical:** the `<think>` regex was `r" thinking.*?</think>"` (space + "thinking"), so tags were never stripped. It is now `r"<think>.*?</think>"`, plus a stray-closing-tag rule for models that omit the opening tag |
| L2 | `generate` | A cached entry that fails `_parse_structured` no longer aborts the probe; remaining providers' caches are checked before the network |
| L3 | `_try_provider` | Latency now uses the injected `self._clock()` instead of `time.perf_counter()` |
| L4 | `_strict_json_schema` | No longer rewrites `default` / `examples` / `const` / `enum` data, and preserves a schema-valued `additionalProperties` (free-form map field) |
| L5 | `_extract_content` | Multipart chunks whose `text` is `None` are skipped instead of raising `TypeError` |
| L6 | `_try_provider` | Non-object `usage` (`[]`, `"n/a"`, …) no longer raises `AttributeError` |
| L7 | `_build_payload` | Schema instruction merges into an existing leading system message instead of creating two consecutive system turns |
| L8 | `_try_provider` | `base_url.rstrip("/")` before joining `/chat/completions`; explicit 3xx handling (redirects fail over instead of being parsed as JSON) |
| L9 | `generate` | Candidate order and prompt hashes are computed once per call, not twice |
| L10 | `_try_provider` | `Retry-After` longer than `backoff_cap_s` now fails over immediately instead of sleeping and retrying a throttled endpoint |
| L11 | `_prompt_hash` | Includes `cfg.structured_output`, so switching a provider from `none` to `json_schema` invalidates the cache |
| L12 | `_parse_structured` | Fenced-code extraction now handles any language tag (e.g. `json5`) and unclosed fences |
| L13 | `_error_snippet` | Slices the response body before splitting to avoid materialising a huge payload |

Deliberately unchanged: 5xx responses still keep their quota slot. The provider saw and counted the request, and the existing test pins that decision. 429 / 401 / 403 / transport errors continue to release the slot.

## Follow-up `ingest/` review fixes

A review of `ingest/manifest.py` and `ingest/downloader.py` found corpus-integrity and HTTP edge cases. Fixes 1-7 were applied; 8-9 were postponed.

| # | Area | Fix |
|---|------|-----|
| I1 | `CorpusConfig` | Validates that no two documents resolve to the same local PDF (case-folded), so a config cannot silently point two entries at one file |
| I2 | `CorpusConfig` | `schema_version` is `Literal[1]` plus a before-validator that also rejects `True`/`False` and any value other than `1` |
| I3 | `_unique_slug` | Suffix length is accounted for dynamically, so `-1000` cannot push a slug past the 80-character `doc_id` limit |
| I4 | `load_corpus_config` | Wraps `OSError` / `UnicodeDecodeError` in `ManifestError` instead of leaking raw exceptions |
| I5 | `PdfDownloader._attempt` | Explicit guard for `204` / non-2xx statuses, with a clear "unexpected HTTP N" message instead of a misleading WAF-page error |
| I6 | `verify_pdf_file` / `_verify_streamed` | Boundary is now `size < MIN_PDF_BYTES` (a 1024-byte file passes); error text says "at least"; `head[:8]` simplified to `head` |
| I7 | `download_corpus` | Exposes `timeout_s` and passes it through to `PdfDownloader` |

Postponed: honoring `Retry-After` in tenacity's downloader wait, retrying non-timeout `httpx.TransportError` mid-stream, and concurrent temp-file safety.

## Design decisions worth remembering

1. **Local-first is structural, not conditional.** `PdfDownloader.process`
   verifies an existing file and returns before any client is constructed, so a
   fully local run makes exactly zero HTTP requests (asserted in the tests).
2. **Verify before publish.** Downloads stream into `<name>.pdf.tmp`; magic
   bytes, the 1 KiB floor, and the optional SHA-256 are checked on the temp file
   *before* `Path.replace`. A 200-with-HTML WAF page or a truncated transfer
   therefore never leaves a `.pdf` (or a `.tmp`) behind.
3. **Only transient failures retry.** tenacity retries 429/502/503 and timeouts
   with exponential jitter. 403/404 fail immediately with the exact path the
   operator should save the file to, because FDA media endpoints reject scripted
   clients and retrying only wastes time.
4. **`scan-local` is the primary acquisition path.** It pins `expected_sha256`
   from the bytes it actually saw, so a later file replacement is detected.
   Duplicate content under a new name is skipped by hash, and colliding slugs get
   `-2` suffixes rather than overwriting an entry.
5. **The YAML header is canonical.** `scan-local` rewrites `corpus.yaml` from
   `CORPUS_HEADER` + `yaml.safe_dump`, which keeps the file readable but drops
   comments added below the header (documented in the file itself). A test pins
   the shipped file to `render_corpus_yaml([])` so header drift is caught.
6. **The manifest is always written**, including when every document fails, so a
   partial run is still auditable. Exit-code policy lives only in the CLI
   (`1` on failures unless `--allow-partial`).
7. **An empty corpus warns instead of reporting success.** `corpus download`
   prints `WARN no documents configured` when the manifest has zero entries. The
   exit code stays `0` because nothing failed; the warning prevents a
   zero-document manifest from looking like a green corpus gate.

## Verification evidence

```
$ python -m uv run ruff check src tests
All checks passed!

$ python -m uv run pytest -q
187 passed in 1.76s

$ python -m uv run mypy src tests
# clean modulo pre-existing import-untyped stub notes
```

Fully offline: every download path runs through `httpx.MockTransport`; the
local-first tests fail if the transport handler is ever invoked.

CLI smoke tests (no credentials, no network):

```
$ python -m uv run rag-eval doctor
FAIL  no usable LLM provider            # exit 1, unchanged from Phase 0

$ python -m uv run rag-eval corpus download
total 0  ok 0  failed 0
WARN  no documents configured in the corpus manifest
OK    corpus is ready                   # empty corpus, manifest written, exit 0

$ python -m uv run rag-eval corpus scan-local
no PDFs found in data\raw               # exit 0
```

Banned-dependency check unchanged: no `torch`, `onnxruntime`, `transformers`,
`sentence-transformers`, `spacy`, `docling`, `marker`, `easyocr`, `ollama`,
`ragas`, or `deepeval`.

## Not done (deliberately)

- No live download was executed inside the agent loop; the corpus is still empty.
- No PDF parsing, text extraction, chunking, index building, or metrics.
- `configs/corpus.yaml` ships with zero documents. The 20-25 real FDA documents
  must be added by a human with verified `source_url` values (or dropped into
  `data/raw/` and registered with `scan-local`); no URL was guessed.

## Next phase

Sprint 2B should start with canonical document parsing and the citation-invariant
`(doc_id, page_num, evidence_quote, start_char, end_char)` contract, and must not
touch retrieval metrics until the golden QA schema exists.
