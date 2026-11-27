"""Command-line entrypoints for rag-eval-gate."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from rag_eval.cache.sqlite_cache import SqliteCache
from rag_eval.config import (
    ConfigError,
    ProvidersConfig,
    Settings,
    build_env,
    load_providers_config,
    resolve_api_key,
)
from rag_eval.evals.dataset import CATEGORY_VALUES
from rag_eval.evals.dataset_validator import (
    DEFAULT_SMOKE_SIZE,
    DatasetError,
    coverage_problems,
    dataset_statistics,
    load_golden_qa,
    validate_dataset,
    write_resolved,
    write_smoke,
)
from rag_eval.ingest.canonical_parser import (
    CanonicalParseError,
    parse_pdf,
    write_canonical_document,
)
from rag_eval.ingest.downloader import download_corpus
from rag_eval.ingest.manifest import (
    DEFAULT_CORPUS_CONFIG_PATH,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_RAW_DIR,
    ManifestEntry,
    ManifestError,
    load_corpus_config,
    scan_local_pdfs,
    sha256_file,
)
from rag_eval.ingest.models import CanonicalDocument
from rag_eval.pipeline.index import (
    DEFAULT_DOCS_DIR,
    DEFAULT_INDEXES_DIR,
    IndexBuildError,
    build_index,
    list_indexes,
    read_manifest,
)
from rag_eval.pipeline.models import STRATEGY_VALUES, ChunkConfig
from rag_eval.pipeline.retriever import (
    RETRIEVAL_MODES,
    HybridRetriever,
    RetrievalError,
    RetrievalMode,
)
from rag_eval.providers.embeddings import EmbeddingClient, EmbeddingError
from rag_eval.providers.llm import AllProvidersFailed, LLMRouter
from rag_eval.providers.rerank import Reranker, RerankError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Evals-first, CI-gated RAG tooling.",
)

_VALID_TARGETS = ("embedding", "llm", "rerank")


@app.callback()
def _root() -> None:
    """Keep Typer from collapsing the CLI into a single command."""


corpus_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Corpus manifest: register local PDFs and fetch the rest.",
)
app.add_typer(corpus_app, name="corpus")

dataset_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Golden QA dataset: validate citations and summarise coverage.",
)
app.add_typer(dataset_app, name="dataset")

index_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Hybrid index: chunk the parsed corpus and build FTS5 + vector artifacts.",
)
app.add_typer(index_app, name="index")


@app.command()
def doctor(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to providers.yaml."
    ),
    live: bool = typer.Option(False, "--live", help="Make one real API call."),
    target: str = typer.Option(
        "embedding", "--target", help="Live probe target: embedding, llm, or rerank."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Fail if any enabled provider is missing a key."
    ),
) -> None:
    """Check configuration, credentials, and (optionally) one live provider."""
    settings = Settings()
    env = build_env(settings)
    config_path = config if config is not None else settings.rag_eval_providers_config

    try:
        providers = load_providers_config(config_path, env=env)
    except ConfigError as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"config   {config_path}")
    typer.echo(f"cache    {settings.rag_eval_cache_path}")
    typer.echo("")

    missing_keys: list[str] = []
    usable = {
        "llm": _report_llm(providers, settings, env, missing_keys),
        "embeddings": _report_embeddings(providers, settings, env, missing_keys),
        "rerank": _report_rerank(providers, settings, env, missing_keys),
    }

    typer.echo("")
    if missing_keys:
        typer.secho(
            "missing credentials: " + ", ".join(sorted(set(missing_keys))),
            fg=typer.colors.YELLOW,
        )

    if usable["llm"] == 0:
        typer.secho("FAIL  no usable LLM provider", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    if strict and missing_keys:
        typer.secho("FAIL  --strict set and credentials are missing", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("OK    configuration is loadable", fg=typer.colors.GREEN)

    if live:
        if target not in _VALID_TARGETS:
            typer.secho(
                f"FAIL  --target must be one of: {', '.join(_VALID_TARGETS)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        _live_probe(target, providers, settings, env)


def _report_llm(
    providers: ProvidersConfig,
    settings: Settings,
    env: dict[str, str],
    missing_keys: list[str],
) -> int:
    rows: list[tuple[str, ...]] = []
    usable = 0
    for name in providers.llm.order:
        cfg = providers.llm.providers[name]
        has_key = bool(resolve_api_key(cfg.api_key_env, settings, environ=env))
        status = _status(cfg.enabled, cfg.is_configured, has_key, cfg.unresolved_env)
        if cfg.enabled and not has_key:
            missing_keys.append(cfg.api_key_env)
        if status == "ready":
            usable += 1
        rows.append(
            (
                name,
                cfg.model,
                f"{cfg.rpm or '-'}/{cfg.rpd or '-'}",
                cfg.structured_output,
                "yes" if has_key else "no",
                status,
            )
        )
    typer.echo("LLM chain (in fallback order)")
    typer.echo(_table(("provider", "model", "rpm/rpd", "structured", "key", "status"), rows))
    return usable


def _report_embeddings(
    providers: ProvidersConfig,
    settings: Settings,
    env: dict[str, str],
    missing_keys: list[str],
) -> int:
    rows: list[tuple[str, ...]] = []
    usable = 0
    for name in providers.embeddings.order:
        cfg = providers.embeddings.providers[name]
        has_key = bool(resolve_api_key(cfg.api_key_env, settings, environ=env))
        status = _status(cfg.enabled, cfg.is_configured, has_key, cfg.unresolved_env)
        if cfg.enabled and not has_key:
            missing_keys.append(cfg.api_key_env)
        if status == "ready":
            usable += 1
        rows.append((name, cfg.model, str(cfg.dims), "yes" if has_key else "no", status))
    typer.echo("")
    typer.echo(
        f"Embeddings (fallback {'on' if providers.embeddings.fallback_enabled else 'off'})"
    )
    typer.echo(_table(("provider", "model", "dims", "key", "status"), rows))
    return usable


def _report_rerank(
    providers: ProvidersConfig,
    settings: Settings,
    env: dict[str, str],
    missing_keys: list[str],
) -> int:
    cfg = providers.rerank
    has_key = bool(resolve_api_key(cfg.api_key_env, settings, environ=env))
    status = _status(cfg.enabled, cfg.is_configured, has_key, cfg.unresolved_env)
    if cfg.enabled and not has_key:
        missing_keys.append(cfg.api_key_env)
    typer.echo("")
    typer.echo("Rerank")
    typer.echo(
        _table(
            ("provider", "model", "top_n", "key", "status"),
            [("jina", cfg.model, str(cfg.top_n), "yes" if has_key else "no", status)],
        )
    )
    return 1 if status == "ready" else 0


def _status(enabled: bool, configured: bool, has_key: bool, unresolved: tuple[str, ...]) -> str:
    if not enabled:
        return "disabled"
    if unresolved:
        return f"missing env: {','.join(unresolved)}"
    if not configured:
        return "bad base_url"
    if not has_key:
        return "no key"
    return "ready"


def _live_probe(
    target: str,
    providers: ProvidersConfig,
    settings: Settings,
    env: dict[str, str],
) -> None:
    resolver = lambda name: resolve_api_key(name, settings, environ=env)  # noqa: E731

    with SqliteCache(settings.rag_eval_cache_path) as cache:
        try:
            if target == "embedding":
                client = EmbeddingClient(providers.embeddings, cache, api_key_resolver=resolver)
                embedding_result = client.embed_texts(["rag-eval doctor probe"])
                norm = float((embedding_result.vectors[0] ** 2).sum() ** 0.5)
                typer.secho(
                    f"OK    embedding via {embedding_result.provider_used} "
                    f"({embedding_result.model_used}) "
                    f"shape={embedding_result.vectors.shape} L2={norm:.4f}",
                    fg=typer.colors.GREEN,
                )
            elif target == "llm":
                router = LLMRouter(providers.llm, cache, api_key_resolver=resolver)
                try:
                    llm_result = router.generate(
                        [{"role": "user", "content": "Reply with the single word OK."}],
                        use_cache=False,
                    )
                finally:
                    router.close()
                typer.secho(
                    f"OK    llm via {llm_result.provider_used} ({llm_result.model_used}) "
                    f"tokens={llm_result.usage_tokens} latency={llm_result.latency_ms}ms",
                    fg=typer.colors.GREEN,
                )
            else:
                reranker = Reranker(providers.rerank, api_key_resolver=resolver)
                try:
                    hits = reranker.rerank(
                        "what is retrieval augmented generation",
                        [
                            "RAG combines a retriever with a generator.",
                            "Bananas are an excellent source of potassium.",
                        ],
                    )
                finally:
                    reranker.close()
                typer.secho(
                    f"OK    rerank via {providers.rerank.model} hits={hits}",
                    fg=typer.colors.GREEN,
                )
        except (EmbeddingError, AllProvidersFailed, RerankError) as exc:
            typer.secho(f"FAIL  live {target} probe: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None


@app.command("cache-stats")
def cache_stats(
    cache_path: Path | None = typer.Option(None, "--cache", help="Override the cache path."),
) -> None:
    """Show cache and quota table sizes."""
    settings = Settings()
    path = cache_path if cache_path is not None else settings.rag_eval_cache_path
    if not path.exists():
        typer.secho(f"FAIL  cache not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    with SqliteCache(path) as cache:
        for key, value in cache.stats().items():
            typer.echo(f"{key:28} {value}")
        rows = [(provider, day, str(used)) for provider, day, used in cache.iter_quota()]
        if rows:
            typer.echo("")
            typer.echo(_table(("provider", "window_date", "used"), rows))


@corpus_app.command("scan-local")
def corpus_scan_local(
    raw_dir: Path = typer.Option(
        DEFAULT_RAW_DIR, "--raw-dir", help="Directory holding manually downloaded PDFs."
    ),
    config: Path = typer.Option(
        DEFAULT_CORPUS_CONFIG_PATH, "--config", "-c", help="Path to corpus.yaml."
    ),
) -> None:
    """Register PDFs already present in the raw directory."""
    try:
        result = scan_local_pdfs(raw_dir, config)
    except ManifestError as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    if not result.added and not result.skipped:
        typer.echo(f"no PDFs found in {raw_dir}")
        typer.echo(f"config  {result.config_path}")
        return

    table = Table(title=f"scan-local  {raw_dir}", header_style="bold")
    table.add_column("file / doc_id", no_wrap=True)
    table.add_column("result", no_wrap=True)
    table.add_column("detail", overflow="fold")
    for doc_id in result.added:
        table.add_row(doc_id, Text("added", style="green"), "")
    for name, reason in result.skipped:
        table.add_row(name, Text("skipped", style="yellow"), reason)
    Console().print(table)
    typer.echo("")
    typer.echo(f"config  {result.config_path}")
    typer.secho(
        f"OK    {len(result.added)} added, {len(result.skipped)} skipped",
        fg=typer.colors.GREEN,
    )


@corpus_app.command("download")
def corpus_download(
    config: Path = typer.Option(
        DEFAULT_CORPUS_CONFIG_PATH, "--config", "-c", help="Path to corpus.yaml."
    ),
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, "--raw-dir", help="PDF output directory."),
    allow_partial: bool = typer.Option(
        False, "--allow-partial", help="Exit 0 even when some documents fail."
    ),
) -> None:
    """Verify local PDFs and download anything missing."""
    try:
        summary = download_corpus(config, raw_dir)
    except ManifestError as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    _print_manifest(summary.entries)
    typer.echo("")
    typer.echo(f"manifest  {summary.manifest_path}")
    typer.echo(
        f"total {len(summary.entries)}  ok {summary.ok_count}  failed {summary.failed_count}"
    )

    if not summary.entries:
        typer.secho(
            "WARN  no documents configured in the corpus manifest",
            fg=typer.colors.YELLOW,
        )

    if summary.has_failures:
        if allow_partial:
            typer.secho(
                "WARN  some documents failed (--allow-partial)", fg=typer.colors.YELLOW
            )
            return
        typer.secho(
            "FAIL  some documents failed; see the table above",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.secho("OK    corpus is ready", fg=typer.colors.GREEN)


@corpus_app.command("parse")
def corpus_parse(
    config: Path = typer.Option(
        DEFAULT_CORPUS_CONFIG_PATH, "--config", "-c", help="Path to corpus.yaml."
    ),
    manifest: Path = typer.Option(
        DEFAULT_RAW_DIR / DEFAULT_MANIFEST_FILENAME,
        "--manifest",
        help="Path to data/raw/manifest.json.",
    ),
    raw_dir: Path = typer.Option(DEFAULT_RAW_DIR, "--raw-dir", help="Directory holding the PDFs."),
    out_dir: Path = typer.Option(
        Path("data/processed/docs"), "--out-dir", help="Canonical JSON output directory."
    ),
) -> None:
    """Parse verified PDFs into canonical JSON documents."""
    if not manifest.is_file():
        typer.secho(f"FAIL  manifest not found: {manifest}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        entries = [
            ManifestEntry.model_validate(item) for item in payload.get("documents", [])
        ]
    except (ValueError, KeyError, TypeError) as exc:
        typer.secho(f"FAIL  invalid manifest {manifest}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    metadata = {}
    if config.is_file():
        try:
            corpus = load_corpus_config(config)
        except ManifestError as exc:
            typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from None
        metadata = {document.doc_id: document for document in corpus.documents}

    verified = [
        entry for entry in entries if entry.status in {"downloaded", "local_verified"}
    ]
    if not verified:
        typer.secho("WARN  no verified documents in the manifest", fg=typer.colors.YELLOW)
        return

    rows: list[tuple[str, str, str, str, str]] = []
    failures = 0
    for entry in verified:
        pdf_path = Path(entry.local_path)
        if not pdf_path.is_file():
            fallback = raw_dir / pdf_path.name
            if fallback.is_file():
                pdf_path = fallback
        if not pdf_path.is_file():
            rows.append((entry.doc_id, "-", "-", "-", "missing file"))
            failures += 1
            continue

        actual_sha256 = sha256_file(pdf_path)
        if entry.sha256 is not None and actual_sha256 != entry.sha256:
            rows.append((entry.doc_id, "-", "-", "-", "sha256 mismatch"))
            failures += 1
            continue

        meta = metadata.get(entry.doc_id)
        title = meta.title if meta is not None else entry.title
        source_url = meta.source_url if meta is not None else entry.source_url
        if meta is not None and meta.effective_date is not None:
            effective_date = meta.effective_date.isoformat()
        elif entry.effective_date is not None:
            effective_date = entry.effective_date.isoformat()
        else:
            effective_date = None

        try:
            document = parse_pdf(
                pdf_path,
                doc_id=entry.doc_id,
                title=title,
                sha256=actual_sha256,
                source_url=source_url,
                effective_date=effective_date,
            )
        except CanonicalParseError as exc:
            rows.append((entry.doc_id, "-", "-", "-", f"parse failed: {exc}"))
            failures += 1
            continue

        target = write_canonical_document(document, out_dir)
        rows.append(
            (
                document.doc_id,
                str(document.page_count),
                str(document.table_count),
                f"{document.token_count:,}",
                f"parsed -> {target.name}",
            )
        )

    table = Table(title="corpus parse", header_style="bold")
    table.add_column("doc_id", no_wrap=True)
    table.add_column("pages", justify="right", no_wrap=True)
    table.add_column("tables", justify="right", no_wrap=True)
    table.add_column("tokens", justify="right", no_wrap=True)
    table.add_column("result", overflow="fold")
    for doc_id, pages, tables, tokens, result in rows:
        style = "red" if result.startswith(("missing", "sha256", "parse failed")) else "green"
        table.add_row(doc_id, pages, tables, tokens, Text(result, style=style))
    Console().print(table)
    typer.echo("")
    typer.echo(f"output  {out_dir}")
    typer.echo(f"total {len(verified)}  parsed {len(verified) - failures}  failed {failures}")

    if failures:
        typer.secho(
            "FAIL  some documents could not be parsed; see the table above",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    typer.secho("OK    all verified documents parsed", fg=typer.colors.GREEN)


@corpus_app.command("stats")
def corpus_stats(
    docs_dir: Path = typer.Option(
        Path("data/processed/docs"), "--docs-dir", help="Canonical JSON directory."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Exit 1 if total tokens exceed this budget."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit 1 if total tokens exceed the 750k alert threshold."
    ),
) -> None:
    """Summarise parsed documents from precomputed canonical JSON."""
    files = sorted(docs_dir.glob("*.json"))
    if not files:
        typer.secho(
            "WARN  no parsed documents; run `rag-eval corpus parse` first",
            fg=typer.colors.YELLOW,
        )
        return

    rows: list[tuple[str, str, str, str, str]] = []
    total_pages = total_chars = total_tables = total_tokens = 0
    skipped = 0
    for path in files:
        try:
            document = CanonicalDocument.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (ValueError, KeyError) as exc:
            typer.secho(f"WARN  skipping {path.name}: {exc}", fg=typer.colors.YELLOW)
            skipped += 1
            continue
        rows.append(
            (
                document.doc_id,
                str(document.page_count),
                f"{len(document.full_text):,}",
                str(document.table_count),
                f"{document.token_count:,}",
            )
        )
        total_pages += document.page_count
        total_chars += len(document.full_text)
        total_tables += document.table_count
        total_tokens += document.token_count

    table = Table(title="corpus stats", header_style="bold")
    table.add_column("doc_id", no_wrap=True)
    table.add_column("pages", justify="right", no_wrap=True)
    table.add_column("chars", justify="right", no_wrap=True)
    table.add_column("tables", justify="right", no_wrap=True)
    table.add_column("tokens", justify="right", no_wrap=True)
    for row in rows:
        table.add_row(*row)
    Console().print(table)
    typer.echo("")
    typer.echo(
        f"documents {len(rows)}  pages {total_pages}  chars {total_chars:,}  "
        f"tables {total_tables}  tokens {total_tokens:,}"
    )
    if skipped:
        typer.echo(f"skipped {skipped} unreadable document(s)")

    if total_tokens > 750_000:
        typer.secho(
            f"ALERT  total tokens {total_tokens:,} exceed 750,000; reduce the corpus "
            "before running paid or rate-limited calls",
            fg=typer.colors.RED,
            err=True,
        )
    elif total_tokens > 500_000:
        typer.secho(
            f"WARN  total tokens {total_tokens:,} exceed 500,000; approaching the "
            "free-tier budget",
            fg=typer.colors.YELLOW,
        )

    if strict and total_tokens > 750_000:
        raise typer.Exit(code=1)
    if max_tokens is not None and total_tokens > max_tokens:
        typer.secho(
            f"FAIL  total tokens {total_tokens:,} exceed --max-tokens {max_tokens:,}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@dataset_app.command("validate")
def dataset_validate(
    input_path: Path = typer.Option(
        Path("data/eval/golden_qa.jsonl"),
        "--input",
        "-i",
        help="Path to golden_qa.jsonl.",
    ),
    docs_dir: Path = typer.Option(
        Path("data/processed/docs"),
        "--docs-dir",
        help="Directory holding canonical parsed documents.",
    ),
    out: Path = typer.Option(
        Path("data/eval/golden_qa.resolved.json"),
        "--out",
        "-o",
        help="Resolved dataset output path (smoke split is written beside it).",
    ),
    smoke_size: int = typer.Option(
        DEFAULT_SMOKE_SIZE, "--smoke-size", help="Number of stratified smoke items."
    ),
    strict: bool = typer.Option(
        False,
        "--strict/--no-strict",
        help="Fail on any problem or coverage shortfall (CI mode).",
    ),
) -> None:
    """Resolve citation offsets and write the resolved dataset + smoke split."""
    if not input_path.is_file():
        typer.secho(f"FAIL  dataset not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        result = validate_dataset(input_path, docs_dir, smoke_size=smoke_size)
    except DatasetError as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    resolved_path = write_resolved(out, result.resolved)
    smoke_path = write_smoke(out.with_name("smoke_qa_ids.json"), result.smoke_ids)

    if result.problems:
        table = Table(title="dataset validate", header_style="bold")
        table.add_column("line", justify="right", no_wrap=True)
        table.add_column("qa_id", no_wrap=True)
        table.add_column("problem", overflow="fold")
        for problem in result.problems:
            table.add_row(
                str(problem.line_number),
                problem.qa_id or "-",
                Text(problem.message, style="red"),
            )
        Console().print(table)
        typer.echo("")

    stats = dataset_statistics(result.items)
    category_summary = "  ".join(
        f"{category}={stats.by_category.get(category, 0)}" for category in CATEGORY_VALUES
    )
    typer.echo(
        f"lines {result.total_lines}  valid {len(result.items)}  "
        f"resolved {len(result.resolved)}  problems {result.error_count}"
    )
    typer.echo(f"categories  {category_summary}")
    typer.echo(f"resolved    {resolved_path}")
    typer.echo(f"smoke       {smoke_path}  ({len(result.smoke_ids)} items)")

    strict_problems = coverage_problems(result.items) if strict else []
    for message in strict_problems:
        typer.secho(f"FAIL  {message}", fg=typer.colors.RED, err=True)

    if result.problems:
        typer.secho(
            f"WARN  {result.error_count} problem(s); resolved output contains only "
            "valid items",
            fg=typer.colors.YELLOW,
        )

    if strict and (result.problems or strict_problems):
        raise typer.Exit(code=1)
    if not result.items:
        typer.secho("FAIL  no valid items in the dataset", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.secho("OK    dataset validated", fg=typer.colors.GREEN)


@dataset_app.command("stats")
def dataset_stats(
    input_path: Path = typer.Option(
        Path("data/eval/golden_qa.jsonl"),
        "--input",
        "-i",
        help="Path to golden_qa.jsonl (read directly; resolved JSON not required).",
    ),
) -> None:
    """Summarise category, difficulty, doc coverage, and citation counts."""
    if not input_path.is_file():
        typer.secho(f"FAIL  dataset not found: {input_path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    loaded, problems = load_golden_qa(input_path)
    stats = dataset_statistics([entry.item for entry in loaded])

    category_table = Table(title="by category", header_style="bold")
    category_table.add_column("category", no_wrap=True)
    category_table.add_column("items", justify="right", no_wrap=True)
    for category in CATEGORY_VALUES:
        category_table.add_row(category, str(stats.by_category.get(category, 0)))
    Console().print(category_table)

    difficulty_table = Table(title="by difficulty", header_style="bold")
    difficulty_table.add_column("difficulty", no_wrap=True)
    difficulty_table.add_column("items", justify="right", no_wrap=True)
    for difficulty in ("easy", "medium", "hard"):
        difficulty_table.add_row(difficulty, str(stats.by_difficulty.get(difficulty, 0)))
    Console().print(difficulty_table)

    citations = "  ".join(
        f"{count}c={items}" for count, items in stats.citation_counts.items()
    )
    typer.echo("")
    typer.echo(
        f"total {stats.total}  unanswerable {stats.unanswerable}  "
        f"docs {len(stats.docs_covered)}  citations {citations or '-'}"
    )
    if problems:
        typer.secho(
            f"WARN  {len(problems)} line(s) could not be parsed (see `dataset validate`)",
            fg=typer.colors.YELLOW,
        )


@index_app.command("build")
def index_build(
    strategy: str = typer.Option(
        "structure_aware",
        "--strategy",
        help="Chunking strategy: fixed_token, structure_aware, or contextual_heading.",
    ),
    max_tokens: int | None = typer.Option(
        None,
        "--max-tokens",
        help="Token window size (fixed_token defaults to 512, others to 450).",
    ),
    overlap_tokens: int | None = typer.Option(
        None, "--overlap-tokens", help="Token overlap between windows (default 64)."
    ),
    hard_max_table_tokens: int = typer.Option(
        900, "--hard-max-table-tokens", help="Cap before an oversized table is split."
    ),
    embedding_provider: str = typer.Option(
        "jina", "--embedding-provider", help="Embedding provider from providers.yaml."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to providers.yaml."
    ),
    docs_dir: Path = typer.Option(
        DEFAULT_DOCS_DIR, "--docs-dir", help="Directory holding canonical documents."
    ),
    out_dir: Path = typer.Option(
        DEFAULT_INDEXES_DIR, "--out-dir", help="Directory holding built indexes."
    ),
    force: bool = typer.Option(
        False, "--force", help="Rebuild even if a fresh matching index exists."
    ),
) -> None:
    """Chunk the parsed corpus and build a hybrid FTS5 + vector index."""
    if strategy not in STRATEGY_VALUES:
        typer.secho(
            f"FAIL  --strategy must be one of: {', '.join(STRATEGY_VALUES)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    resolved_max = max_tokens
    if resolved_max is None:
        resolved_max = 512 if strategy == "fixed_token" else 450
    resolved_overlap = overlap_tokens if overlap_tokens is not None else 64

    try:
        chunk_config = ChunkConfig.model_validate(
            {
                "strategy": strategy,
                "max_tokens": resolved_max,
                "overlap_tokens": resolved_overlap,
                "hard_max_table_tokens": hard_max_table_tokens,
            }
        )
    except ValueError as exc:
        typer.secho(f"FAIL  invalid chunk configuration: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from None

    settings = Settings()
    providers_config_path = config if config is not None else settings.rag_eval_providers_config

    try:
        result = build_index(
            chunk_config,
            docs_dir=docs_dir,
            out_dir=out_dir,
            embedding_provider=embedding_provider,
            providers_config_path=providers_config_path,
            force=force,
        )
    except (IndexBuildError, ConfigError, EmbeddingError) as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    manifest = result.manifest
    typer.echo(f"index_id    {manifest.index_id}")
    typer.echo(f"path        {result.index_dir}")
    typer.echo(f"strategy    {manifest.chunk_config.strategy}")
    typer.echo(
        f"chunking    max_tokens={manifest.chunk_config.max_tokens} "
        f"overlap={manifest.chunk_config.overlap_tokens} "
        f"hard_max_table={manifest.chunk_config.hard_max_table_tokens}"
    )
    typer.echo(
        f"embedding   {manifest.embedding_provider} / {manifest.embedding_model} "
        f"dims={manifest.dims}"
    )
    typer.echo(f"chunks      {manifest.chunk_count}")
    typer.echo(f"corpus      sha256={manifest.corpus_sha256[:16]}…")

    if result.skipped:
        typer.secho("SKIP  index is already fresh (use --force to rebuild)", fg=typer.colors.YELLOW)
        return
    typer.secho("OK    index built", fg=typer.colors.GREEN)


@index_app.command("info")
def index_info(
    index_id: str | None = typer.Argument(
        None, help="Specific index id; omit to list every built index."
    ),
    out_dir: Path = typer.Option(
        DEFAULT_INDEXES_DIR, "--out-dir", help="Directory holding built indexes."
    ),
) -> None:
    """List built indexes or show one manifest in detail."""
    if index_id is None:
        manifests = list_indexes(out_dir)
        if not manifests:
            typer.secho(f"WARN  no indexes found in {out_dir}", fg=typer.colors.YELLOW)
            return
        table = Table(title=f"indexes  {out_dir}", header_style="bold")
        table.add_column("index_id", no_wrap=True)
        table.add_column("strategy", no_wrap=True)
        table.add_column("max_tok", justify="right", no_wrap=True)
        table.add_column("provider", no_wrap=True)
        table.add_column("model", overflow="fold")
        table.add_column("dims", justify="right", no_wrap=True)
        table.add_column("chunks", justify="right", no_wrap=True)
        table.add_column("created_at", no_wrap=True)
        for manifest in sorted(manifests, key=lambda item: item.index_id):
            table.add_row(
                manifest.index_id,
                manifest.chunk_config.strategy,
                str(manifest.chunk_config.max_tokens),
                manifest.embedding_provider,
                manifest.embedding_model,
                str(manifest.dims),
                str(manifest.chunk_count),
                manifest.created_at,
            )
        Console().print(table)
        return

    try:
        manifest = read_manifest(Path(out_dir) / index_id)
    except IndexBuildError as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"index_id    {manifest.index_id}")
    typer.echo(f"path        {Path(out_dir) / index_id}")
    typer.echo(f"strategy    {manifest.chunk_config.strategy}")
    typer.echo(
        f"chunking    max_tokens={manifest.chunk_config.max_tokens} "
        f"overlap={manifest.chunk_config.overlap_tokens} "
        f"hard_max_table={manifest.chunk_config.hard_max_table_tokens}"
    )
    typer.echo(
        f"embedding   {manifest.embedding_provider} / {manifest.embedding_model} "
        f"dims={manifest.dims}"
    )
    typer.echo(f"chunks      {manifest.chunk_count}")
    typer.echo(f"corpus      sha256={manifest.corpus_sha256}")
    typer.echo(f"created_at  {manifest.created_at}")


@app.command("query")
def query(
    question: str = typer.Argument(..., help="Natural-language question."),
    mode: str = typer.Option(
        "hybrid_rrf", "--mode", help="Retrieval mode: bm25, dense, hybrid_rrf, or rerank."
    ),
    top_k: int = typer.Option(5, "--top-k", min=1, max=100, help="Number of chunks to return."),
    strategy: str = typer.Option(
        "contextual_heading",
        "--strategy",
        help="Index strategy to use when --index is omitted.",
    ),
    index_id: str | None = typer.Option(
        None, "--index", help="Explicit index id; overrides --strategy."
    ),
    out_dir: Path = typer.Option(
        DEFAULT_INDEXES_DIR, "--out-dir", help="Directory holding built indexes."
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to providers.yaml."
    ),
) -> None:
    """Retrieve the best chunks for a question from a built hybrid index."""
    if mode not in RETRIEVAL_MODES:
        typer.secho(
            f"FAIL  --mode must be one of: {', '.join(RETRIEVAL_MODES)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    if strategy not in STRATEGY_VALUES:
        typer.secho(
            f"FAIL  --strategy must be one of: {', '.join(STRATEGY_VALUES)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if index_id is None:
        matches = [
            manifest
            for manifest in list_indexes(out_dir)
            if manifest.chunk_config.strategy == strategy
        ]
        if not matches:
            typer.secho(
                f"FAIL  no {strategy} index in {out_dir}; run `rag-eval index build`",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        if len(matches) > 1:
            ids = ", ".join(sorted(manifest.index_id for manifest in matches))
            typer.secho(
                f"FAIL  multiple {strategy} indexes ({ids}); pass --index",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        index_id = matches[0].index_id

    settings = Settings()
    config_path = config if config is not None else settings.rag_eval_providers_config
    try:
        with HybridRetriever(
            Path(out_dir) / index_id,
            mode=cast(RetrievalMode, mode),
            top_k=top_k,
            providers_config_path=config_path,
            settings=settings,
        ) as retriever:
            results = retriever.search(question)
            manifest = retriever.manifest
    except (RetrievalError, EmbeddingError, RerankError, ConfigError) as exc:
        typer.secho(f"FAIL  {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from None

    typer.echo(f"index   {manifest.index_id}  {manifest.chunk_config.strategy}")
    typer.echo(f"mode    {mode}")
    typer.echo(f"query   {question}")
    typer.echo("")
    if not results:
        typer.secho("WARN  no chunks matched", fg=typer.colors.YELLOW)
        return

    for rank, chunk in enumerate(results, start=1):
        pages = (
            str(chunk.page_start)
            if chunk.page_start == chunk.page_end
            else f"{chunk.page_start}-{chunk.page_end}"
        )
        snippet = " ".join(chunk.raw_text.split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        typer.echo(f"{rank}. {chunk.doc_id} p.{pages}  score={chunk.score:.4f}")
        if chunk.section_path:
            typer.echo(f"   section: {chunk.section_path}")
        typer.echo(f"   {snippet}")
        typer.echo("")
    typer.secho(f"OK    {len(results)} chunk(s)", fg=typer.colors.GREEN)


def _print_manifest(entries: list[ManifestEntry]) -> None:
    table = Table(title="corpus download", header_style="bold")
    table.add_column("doc_id", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("bytes", justify="right", no_wrap=True)
    table.add_column("sha256", no_wrap=True)
    table.add_column("note", overflow="fold")
    for entry in entries:
        style = "red" if entry.status == "failed" else "green"
        table.add_row(
            entry.doc_id,
            Text(entry.status, style=style),
            "-" if entry.bytes is None else f"{entry.bytes:,}",
            entry.sha256[:12] if entry.sha256 else "-",
            entry.error or "",
        )
    Console().print(table)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Plain, deterministic table for machine-friendly doctor output."""
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    body = "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    )
    return "\n".join(part for part in (header_line, separator, body) if part)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
