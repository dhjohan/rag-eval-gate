# AGENTS.md — Evals-First RAG Engineering Rules

## Mission & Architecture

Build an evals-first, CI-gated RAG system over a real messy corpus using **100% cost-free cloud APIs** and **zero local ML model execution**.

## Hard Constraints (NEVER VIOLATE)

1. **NO LOCAL ML WEIGHTS OR HEAVY INFERENCE**:
   - NEVER add `torch`, `transformers`, `sentence-transformers`, `spacy`, `docling`, `marker`, `unstructured[all-docs]`, `easyocr`, `ollama`, or `qdrant-client[fastembed]` to dependencies.
   - Local machine has strict RAM/CPU limits. PDF parsing MUST use pure CPU libraries (`pymupdf`, `pdfplumber`). Vector similarity MUST use pure `numpy` + SQLite, `qdrant-client` (local disk/memory without fastembed), or Qdrant Cloud Free Tier. Lexical search MUST use SQLite FTS5.
2. **100% FREE-TIER CLOUD APIs ONLY**:
   - Embeddings: Google `gemini-embedding-2` (via `google-genai`) and/or Jina AI embeddings API (`httpx`).
   - Reranking: Jina AI Reranker API (`httpx`).
   - Generation & Judge: Multi-provider OpenAI-compatible/SDK router supporting NVIDIA NIM, Cloudflare Workers AI, and OpenRouter `:free`. Groq Free and Gemini AI Studio Free were dropped by explicit decision (Groq is geo-blocked; Gemini chat free-tier quotas are unpublished/too strict).
   - Every embedding call MUST check a local SQLite hash cache `(content_sha256, model_id, dims)` before hitting the network.
   - Every generative LLM call MUST go through a token-bucket rate limiter, exponential backoff retry handler, and optional SQLite prompt cache.
3. **NO MULTI-CALL EVAL FRAMEWORKS**:
   - Do NOT install `ragas` or `deepeval`. Implement transparent, unit-tested deterministic retrieval metrics (`Recall@k`, `MRR@k`, `nDCG@k`) and a single-call structured Pydantic LLM-as-judge evaluator.
4. **CITATION INVARIANCE**:
   - Golden QA items MUST NEVER reference `chunk_id`. They MUST reference `(doc_id, page_num, evidence_quote)` aligned to character offsets `[start_char, end_char]` in canonical parsed documents.
5. **PHASE DISCIPLINE**:
   - Only implement the currently requested Phase. Do not scaffold future phases prematurely.
   - Never fabricate or synthesize the 100-150 human-written golden QA pairs unless explicitly asked to create 3-5 schema test fixtures in `tests/fixtures/`.

## Tech & Code Standards

- Python 3.11+, managed with `uv`.
- Strict typing (`pydantic>=2`, type hints on all public functions), formatted with `ruff`.
- All CLI entrypoints live under `src/rag_eval/cli.py` (using `typer`).
- Secrets come strictly from environment variables / `.env` (never hardcode keys).

## Verification Commands

- Lint & types: `uv run ruff check src tests`
- Unit tests: `uv run pytest -q`
- Check banned heavy deps: verify `uv pip list` does not include `torch` or `onnxruntime` (unless explicitly approved).

## Git Workflow

- Never push directly to `main`. Create `feat/<scope>` or `fix/<scope>` (docs/tooling: `chore/<scope>`).
- Commit on the branch, push it, and open a PR to `main`.
- Merge only after CI is green (`ruff`, `mypy`, `pytest`, and the banned-dependency guard).
- `main` is branch-protected: a pull request and passing status checks are required before merge.

_Enable branch protection on `main` (Settings → Branches) to enforce this._
