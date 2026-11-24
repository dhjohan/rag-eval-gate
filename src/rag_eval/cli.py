"""Command-line entrypoints for rag-eval-gate."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

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
from rag_eval.ingest.downloader import download_corpus
from rag_eval.ingest.manifest import (
    DEFAULT_CORPUS_CONFIG_PATH,
    DEFAULT_RAW_DIR,
    ManifestEntry,
    ManifestError,
    scan_local_pdfs,
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
