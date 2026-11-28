"""Resilient, local-first PDF acquisition for the corpus manifest.

Design rules:

* A document already on disk is verified and reused; the network is never
  touched for it.
* Downloads stream into ``<name>.pdf.tmp`` and are validated (magic bytes,
  minimum size, optional SHA-256) *before* an atomic ``os.replace`` publishes
  them, so an interrupted transfer or an HTML WAF page can never masquerade as
  a PDF.
* Only 429/502/503 and timeouts are retried. 403/404 fail fast with a
  manual-download instruction, because FDA media endpoints reject scripted
  clients and retrying only burns time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from rag_eval.ingest.manifest import (
    DEFAULT_CORPUS_CONFIG_PATH,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_RAW_DIR,
    MIN_PDF_BYTES,
    PDF_MAGIC,
    CorpusDocument,
    ManifestEntry,
    PdfValidationError,
    load_corpus_config,
    verify_pdf_file,
    write_manifest,
)

RETRY_STATUSES = frozenset({429, 502, 503})
CHUNK_SIZE = 1 << 16

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class DownloadError(RuntimeError):
    """Raised when a document cannot be fetched or fails verification."""


class RetryableStatusError(DownloadError):
    """A transient HTTP status that is worth retrying."""

    def __init__(self, status_code: int, url: str) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code} from {url}")


@dataclass(frozen=True)
class DownloadSummary:
    """Outcome of one ``rag-eval corpus download`` run."""

    entries: list[ManifestEntry]
    manifest_path: Path

    @property
    def ok_count(self) -> int:
        return sum(1 for entry in self.entries if entry.status != "failed")

    @property
    def failed_count(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "failed")

    @property
    def has_failures(self) -> bool:
        return self.failed_count > 0


def build_client(
    *,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 60.0,
) -> httpx.Client:
    """Browser-like client. Injectable transport keeps tests fully offline."""
    return httpx.Client(
        follow_redirects=True,
        timeout=timeout_s,
        headers=BROWSER_HEADERS,
        transport=transport,
    )


def _manual_download_hint(status_code: int, url: str, target: Path) -> str:
    return (
        f"HTTP {status_code} for {url}; this status is not retried. "
        f"Download the PDF manually in a browser, save it as {target}, "
        "then re-run `rag-eval corpus download`."
    )


class PdfDownloader:
    """Verifies local PDFs and fetches the ones that are missing."""

    def __init__(
        self,
        raw_dir: str | Path,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 60.0,
        max_attempts: int = 3,
        wait: Callable[[Any], float] | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self._client = client
        self._owns_client = client is None
        self._transport = transport
        self._timeout_s = timeout_s
        self._max_attempts = max(1, max_attempts)
        self._wait = wait if wait is not None else wait_exponential_jitter(initial=0.5, max=8.0)
        self.requests_made = 0

    def _get_client(self) -> httpx.Client:
        """Build the HTTP client on first use so local-only runs stay offline."""
        if self._client is None:
            self._client = build_client(transport=self._transport, timeout_s=self._timeout_s)
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> PdfDownloader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def process(self, document: CorpusDocument) -> ManifestEntry:
        """Resolve one document into a manifest row. Never raises for per-doc errors."""
        target = document.resolved_path(self.raw_dir)
        doc_id = document.doc_id
        title = document.title
        category = document.category
        effective_date = document.effective_date
        local_path = str(target)
        source_url = document.source_url

        if target.is_file():
            try:
                digest, size = verify_pdf_file(target, expected_sha256=document.expected_sha256)
            except PdfValidationError as exc:
                return ManifestEntry(
                    doc_id=doc_id,
                    title=title,
                    category=category,
                    effective_date=effective_date,
                    local_path=local_path,
                    source_url=source_url,
                    status="failed",
                    error=f"local file failed verification: {exc}",
                )
            except OSError as exc:
                # A file that exists but cannot be stat'ed/read (permissions,
                # AV lock, dangling handle) must not abort the whole run.
                return ManifestEntry(
                    doc_id=doc_id,
                    title=title,
                    category=category,
                    effective_date=effective_date,
                    local_path=local_path,
                    source_url=source_url,
                    status="failed",
                    error=f"local file could not be read: {exc}",
                )
            return ManifestEntry(
                doc_id=doc_id,
                title=title,
                category=category,
                effective_date=effective_date,
                local_path=local_path,
                source_url=source_url,
                sha256=digest,
                bytes=size,
                status="local_verified",
            )

        if source_url is None:
            return ManifestEntry(
                doc_id=doc_id,
                title=title,
                category=category,
                effective_date=effective_date,
                local_path=local_path,
                source_url=source_url,
                status="failed",
                error=(
                    f"missing locally and no source_url configured; place the PDF at "
                    f"{target} or add source_url to the corpus config"
                ),
            )

        try:
            digest, size = self._download(
                source_url,
                target,
                expected_sha256=document.expected_sha256,
            )
        except DownloadError as exc:
            return ManifestEntry(
                doc_id=doc_id,
                title=title,
                category=category,
                effective_date=effective_date,
                local_path=local_path,
                source_url=source_url,
                status="failed",
                error=str(exc),
            )
        except httpx.HTTPError as exc:
            return ManifestEntry(
                doc_id=doc_id,
                title=title,
                category=category,
                effective_date=effective_date,
                local_path=local_path,
                source_url=source_url,
                status="failed",
                error=f"transport error: {exc}",
            )
        except OSError as exc:
            return ManifestEntry(
                doc_id=doc_id,
                title=title,
                category=category,
                effective_date=effective_date,
                local_path=local_path,
                source_url=source_url,
                status="failed",
                error=f"filesystem error: {exc}",
            )

        return ManifestEntry(
            doc_id=doc_id,
            title=title,
            category=category,
            effective_date=effective_date,
            local_path=local_path,
            source_url=source_url,
            sha256=digest,
            bytes=size,
            status="downloaded",
        )

    def _download(
        self,
        url: str,
        target: Path,
        *,
        expected_sha256: str | None,
    ) -> tuple[str, int]:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.tmp")
        retrying = Retrying(
            retry=retry_if_exception_type((RetryableStatusError, httpx.TimeoutException)),
            wait=self._wait,
            stop=stop_after_attempt(self._max_attempts),
            reraise=True,
        )
        try:
            return retrying(self._attempt, url, tmp, target, expected_sha256)
        finally:
            with suppress(OSError):  # pragma: no cover - Windows AV/file-lock edge case
                tmp.unlink(missing_ok=True)

    def _attempt(
        self,
        url: str,
        tmp: Path,
        target: Path,
        expected_sha256: str | None,
    ) -> tuple[str, int]:
        tmp.unlink(missing_ok=True)
        self.requests_made += 1
        client = self._get_client()

        digest = hashlib.sha256()
        size = 0
        head = b""
        try:
            with client.stream("GET", url) as response:
                if response.status_code in RETRY_STATUSES:
                    raise RetryableStatusError(response.status_code, url)
                if response.status_code >= 400:
                    raise DownloadError(_manual_download_hint(response.status_code, url, target))
                if response.status_code == 204 or not 200 <= response.status_code < 300:
                    # A successful download must be a 2xx with a body; a 204 or
                    # an unfollowed 3xx would otherwise produce a misleading
                    # "HTML/WAF" message after an empty stream.
                    raise DownloadError(
                        f"unexpected HTTP {response.status_code} for {url}; "
                        "expected a PDF response"
                    )
                with tmp.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        if len(head) < len(PDF_MAGIC):
                            head += chunk[: len(PDF_MAGIC) - len(head)]
                            if not PDF_MAGIC.startswith(head):
                                # Abort on the first chunk: a multi-megabyte WAF
                                # page should not be downloaded in full.
                                raise DownloadError(_not_a_pdf_message(target.name, head))
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        except httpx.TimeoutException:
            raise
        except httpx.HTTPError as exc:
            raise DownloadError(f"transport error for {url}: {exc}") from exc

        _verify_streamed(
            name=target.name,
            head=head,
            size=size,
            digest=digest.hexdigest(),
            expected_sha256=expected_sha256,
        )

        # Atomic publish: readers never observe a partially written .pdf.
        tmp.replace(target)
        return digest.hexdigest(), size


def _not_a_pdf_message(name: str, head: bytes) -> str:
    return (
        f"downloaded file failed verification: {name} is not a PDF: expected "
        f"{PDF_MAGIC.decode()!r} magic bytes, got {head!r}. An HTML error page "
        "or WAF challenge was probably saved."
    )


def _verify_streamed(
    *,
    name: str,
    head: bytes,
    size: int,
    digest: str,
    expected_sha256: str | None,
) -> None:
    """Validate a stream that was hashed while it was written to disk."""
    if head != PDF_MAGIC:
        raise DownloadError(_not_a_pdf_message(name, head))
    if size < MIN_PDF_BYTES:
        raise DownloadError(
            f"downloaded file failed verification: {name} is only {size} bytes; "
            f"a valid PDF must be at least {MIN_PDF_BYTES} bytes."
        )
    if expected_sha256 is not None and digest != expected_sha256.strip().lower():
        raise DownloadError(
            f"downloaded file failed verification: {name} sha256 mismatch: "
            f"expected {expected_sha256}, got {digest}."
        )


def download_corpus(
    config_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    *,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
    timeout_s: float = 60.0,
    max_attempts: int = 3,
    wait: Callable[[Any], float] | None = None,
    manifest_path: str | Path | None = None,
) -> DownloadSummary:
    """Verify local PDFs, fetch missing ones, and always write the manifest."""
    config_file = Path(config_path) if config_path is not None else DEFAULT_CORPUS_CONFIG_PATH
    raw = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR
    corpus = load_corpus_config(config_file)
    raw.mkdir(parents=True, exist_ok=True)

    downloader = PdfDownloader(
        raw,
        client=client,
        transport=transport,
        timeout_s=timeout_s,
        max_attempts=max_attempts,
        wait=wait,
    )
    try:
        entries = [downloader.process(document) for document in corpus.documents]
    finally:
        downloader.close()

    target_manifest = (
        Path(manifest_path) if manifest_path is not None else raw / DEFAULT_MANIFEST_FILENAME
    )
    write_manifest(target_manifest, entries, raw_dir=raw)
    return DownloadSummary(entries=entries, manifest_path=target_manifest)


__all__ = [
    "BROWSER_HEADERS",
    "RETRY_STATUSES",
    "DownloadError",
    "DownloadSummary",
    "PdfDownloader",
    "RetryableStatusError",
    "build_client",
    "download_corpus",
]
