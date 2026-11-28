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

- Lint: `uv run ruff check src tests scripts`
- Types: `uv run mypy src tests scripts`
- Unit tests: `uv run pytest -q`
- Check banned heavy deps: verify `uv pip list` does not include `torch` or `onnxruntime` (unless explicitly approved).

## Git Workflow & Hygiene

- Never push directly to `main`. Create `feat/<scope>` or `fix/<scope>` (docs/tooling: `chore/<scope>`).

### Workspace Safety & Boundaries

- **Inspect first:** Always start with `git status --short` before making edits or running git commands.
- **Preserve existing work:** Never discard, overwrite, reformat, or stage uncommitted changes you did not make. If unexpected dirty files conflict with your task, stop and ask.
- **Ask before destructive operations:** Never run `git reset --hard`, `git clean -fd`, `git checkout -- .`, `git restore .`, `git push --force` (or `--force-with-lease`), or `git branch -D` without explicit user approval.
- **Never alter git config or shared history:** Do not modify `git config` (`user.name`, `user.email`, `core.hooksPath`) and never amend, rebase, or squash commits already pushed to a shared branch.
- **Avoid interactive hangs & temp clutter:** Always pass `git --no-pager` on read commands (`log`, `diff`, `show`), and supply commit messages via repeated `-m` flags (`git commit -m "<subject>" -m "<body>"`) or stdin (`git commit -F -`). Never run `git rebase -i` or `git add -p`, and never leave temporary message or patch files in the working tree.

### Branching & Isolation

- **Never commit directly to protected branches** (`main`, `master`, `develop`, or `release/*`). Create a task branch from the up-to-date default branch (`origin/HEAD`, typically `main`) before making commits.
- **Branch naming:** Use `<type>/<kebab-case-slug>` where `<type>` is one of `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, or `ci` (e.g., `feat/rate-limiter`, `fix/null-webhook`).
- **Stay isolated:** Work strictly inside your assigned branch or git worktree. Do not switch branches in a shared working directory or modify/remove another worktree.

### Staging & Diff Hygiene

- **Stage by explicit path:** Use `git add <path/to/file>`. Never use `git add -A`, `git add .`, or `git add --all`.
- **Pre-commit inspection:** Before every commit, run:
  - `git diff --check` (catches trailing whitespace and leftover conflict markers)
  - `git --no-pager diff --cached --stat` and `git --no-pager diff --cached`
- Confirm the staged diff contains no secrets (`.env`, API keys, tokens, private keys), debug artifacts (`console.log`, `debugger`, `breakpoint()`, `pdb`, `dbg!`, scratch files), build outputs, or unintended lockfile/submodule pointer changes.

### Commits & Hook Integrity

- **Never bypass hooks:** Never pass `--no-verify` or `-n` to `git commit` or `git push`, and never set `SKIP`, `HUSKY=0`, or `LEFTHOOK=0`. If a pre-commit hook or check fails, fix the root cause and re-run.
- **Verify before committing:** Run the repository's configured lint, typecheck, and relevant tests before every commit (checking this file, CI workflows, or project manifests like `package.json`, `Makefile`, `pyproject.toml`, `Cargo.toml`, or `go.mod`). Explicitly report any check that could not be run and why.
- **Atomic Conventional Commits:** Make one logical change per commit using `<type>(<optional-scope>): <imperative summary>` with `<type>` in `{feat, fix, refactor, docs, test, chore, perf, ci, build, revert}`. Keep the subject <= 72 characters, lowercase imperative mood, and no trailing period (e.g., `fix(auth): handle expired refresh tokens`). Use `!` or a `BREAKING CHANGE:` footer for breaking changes, explain _why_ in the body when non-obvious, and omit tool boilerplate.

### Syncing, Conflicts & Pull Requests

- **Sync safely:** Run `git fetch origin` before branching or updating, and prefer `git pull --ff-only` when pulling upstream changes. Only push to remote when requested or when completing a PR workflow.
- **Conflict resolution:** Never blindly resolve merge or rebase conflicts with blanket `--ours` or `--theirs`; if resolution touches logic outside your scope or another author's intent, stop and ask.
- **Pull request discipline:** Keep PRs strictly scoped to the task. Summarize _what_ changed, _why_, any breaking changes or follow-ups, and the exact verification commands run with their observed outcomes.

_Enable branch protection on `main` (Settings → Branches) to enforce this._

## Agent Operating Contract (Nemotron 3 Ultra Profile)

### 1. Repository Grounding (Trust Filesystem Over Docs)

- **Verify reality first, cheaply:** Run a single shallow directory listing (including hidden/dotfiles like `.github/`, `.env.example`, config files) and check the actual manifest (`package.json`, `pyproject.toml`, `Cargo.toml`, `Makefile`) before trusting `README.md`. READMEs may be outdated.
- **Respect scope exclusions:** If the user restricts directories or says _"do not read X"_, obey strictly. Do NOT bulk-glob or read >5 files speculatively—target only files directly implicated by symbols, imports, or stack traces.
- **No modality bluffing:** You cannot view images, rendered UI screenshots, or binary artifacts. Rely exclusively on DOM/HTML, CSS, tests, logs, and CLI output.

### 2. Planning & Codex-Style Task Lock

- Before editing code on any non-trivial task, write a concise **<= 5-step execution plan** mapping:
  1. Target files/symbols to modify
  2. Exact verification command to prove the fix works
- Do **not** start editing until the acceptance criteria and test command are clear. If the user's request is ambiguous, ask **at most 3 crisp questions** (or state explicit assumptions if asked to proceed autonomously) rather than burning reasoning tokens deliberating.

### 3. Surgical File Editing & Tool-Calling Discipline

- **Read before patch:** Always read the exact target line range immediately before issuing an edit/patch call so indentation, whitespace, and context lines match verbatim.
- **Validate tool arguments:** Double-check JSON escaping, path accuracy (relative vs. absolute), and tool schema types _before_ emitting a tool call. Never dump raw tool-call JSON into plain text replies.
- **Batch independent reads:** Consolidate independent file reads or searches into parallel tool calls where supported, rather than chaining 10 sequential turns (prevents context bloat and API `429` rate-limit exhaustion).

### 4. Error Recovery & Anti-Loop Circuit Breakers

- **Never ignore a failed tool output:** If a command, linter, or patch fails, halt forward progress, quote the exact error line, and fix the root cause before moving to the next step.
- **FORBIDDEN — Destructive Reset Loops:** NEVER run `git checkout -- .`, `git reset --hard`, or wipe your own edits to "start over" unless the user explicitly commands it. Fix forward using targeted diffs.
- **2-Strike Escalation:** If the exact same edit, test, or build error occurs **twice** after an attempted fix:
  1. STOP executing tools immediately.
  2. Output a concise diagnosis: what failed, exact error output, and 2 alternative paths for the user to approve.

### 5. Verification & Completion Gate

- **Zero unverified claims:** Never report _"tests pass"_ or _"bug fixed"_ unless you executed the verification command _after your final edit_ and observed a clean exit code.
- **Verbatim reporting:** Cite exact file paths, symbol names, command run, and observed pass/fail counts.
- **Concision:** Keep final summaries to `<= 8 lines` unless a detailed report was requested. No filler or defensive commentary.

### 6. Project-Specific Ground Truth (Customize Per Repo)

- **Source Root:** `[e.g., src/]`
- **Test Command:** `[e.g., uv run pytest -q]`
- **Lint / Typecheck Command:** `[e.g., ruff check . && mypy src/]`
- **Protected / Do-Not-Touch Paths:** `[e.g., migrations/, .github/workflows/, lockfiles]`
