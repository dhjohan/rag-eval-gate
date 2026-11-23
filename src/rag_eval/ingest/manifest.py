"""Corpus manifest schema, local PDF scanning, and PDF integrity checks.

``configs/corpus.yaml`` is the source of truth for *which* documents the corpus
contains. This module validates that file, renders it back without losing the
explanatory header, registers PDFs that were downloaded by hand, and performs
the integrity checks that both the scanner and the downloader rely on.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_CORPUS_CONFIG_PATH = Path("configs/corpus.yaml")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_CATEGORY = "fda_digital_health"

PDF_MAGIC = b"%PDF-"
MIN_PDF_BYTES = 1024

DOC_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

DocStatus = Literal["downloaded", "local_verified", "failed"]

CORPUS_HEADER = """\
# Corpus manifest for the rag-eval-gate FDA digital-health PDF set.
#
# Target: 20-25 PDFs from FDA's digital-health / software-as-a-medical-device
# material. FDA's own Excel export does not expose raw PDF URLs, so this file is
# the single source of truth for what the corpus contains and where it came from.
#
# Each document entry accepts:
#   doc_id          required  lowercase slug, unique, e.g. fda-ai-ml-action-plan
#   title           required  human-readable document title
#   category        optional  defaults to fda_digital_health
#   effective_date  optional  YYYY-MM-DD
#   source_url      optional  absolute http(s) URL to the PDF
#   local_filename  optional  bare *.pdf file name inside --raw-dir
#   expected_sha256 optional  64 hex chars; verified after download and on reuse
#
# Every entry needs source_url, local_filename, or both. Path resolution is
# `data/raw/{local_filename}` when local_filename is set, else
# `data/raw/{doc_id}.pdf`. A document that is already on disk is verified and
# reused; only missing documents are fetched.
#
# Acquisition workflow:
#   1. Drop PDFs into data/raw/ (manual browser download if the host blocks us).
#   2. Run `rag-eval corpus scan-local` to register them with their sha256.
#   3. Run `rag-eval corpus download` to verify local files and fetch the rest.
#
# NOTE: `rag-eval corpus scan-local` rewrites this file from a canonical header.
# Comments added below this header are not preserved.
#
# Example (replace the placeholder URL with the real one; never guess FDA URLs):
# documents:
#   - doc_id: fda-example-guidance
#     title: "Example guidance document"
#     category: fda_digital_health
#     effective_date: 2024-01-15
#     source_url: https://example.invalid/replace-with-real-url.pdf
#     local_filename: fda-example-guidance.pdf
#     expected_sha256: null
"""


class ManifestError(ValueError):
    """Raised when a corpus manifest or config file is invalid."""


class PdfValidationError(ManifestError):
    """Raised when a file is not an acceptable PDF."""


class CorpusDocument(BaseModel):
    """One tracked document in ``configs/corpus.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1)
    category: str = Field(default=DEFAULT_CATEGORY, min_length=1)
    effective_date: date | None = None
    source_url: str | None = None
    local_filename: str | None = None
    expected_sha256: str | None = None

    @field_validator("doc_id")
    @classmethod
    def _valid_slug(cls, value: str) -> str:
        cleaned = value.strip()
        if not DOC_ID_PATTERN.match(cleaned):
            raise ValueError(
                "doc_id must be a lowercase slug such as 'fda-ai-ml-action-plan'"
            )
        return cleaned

    @field_validator("title", "category")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("value must not be blank")
        return cleaned

    @field_validator("source_url")
    @classmethod
    def _absolute_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"source_url must be an absolute http(s) URL, got {cleaned!r}")
        return cleaned

    @field_validator("local_filename")
    @classmethod
    def _bare_pdf_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("local_filename must not be blank")
        if "/" in cleaned or "\\" in cleaned or Path(cleaned).is_absolute():
            raise ValueError("local_filename must be a bare file name, not a path")
        if Path(cleaned).name != cleaned:
            raise ValueError("local_filename must be a bare file name, not a path")
        if Path(cleaned).suffix.lower() != ".pdf":
            raise ValueError("local_filename must end with '.pdf'")
        return cleaned

    @field_validator("expected_sha256")
    @classmethod
    def _valid_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not SHA256_PATTERN.match(cleaned):
            raise ValueError("expected_sha256 must be 64 hexadecimal characters")
        return cleaned

    @model_validator(mode="after")
    def _has_a_source(self) -> CorpusDocument:
        if self.source_url is None and self.local_filename is None:
            raise ValueError("each document needs source_url, local_filename, or both")
        return self

    def resolved_path(self, raw_dir: Path | str) -> Path:
        """Where this document's PDF lives: ``local_filename`` else ``{doc_id}.pdf``."""
        name = self.local_filename or f"{self.doc_id}.pdf"
        return Path(raw_dir) / name


class CorpusConfig(BaseModel):
    """The whole ``corpus.yaml`` file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    documents: list[CorpusDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_doc_ids(self) -> CorpusConfig:
        seen: set[str] = set()
        duplicates: list[str] = []
        for document in self.documents:
            if document.doc_id in seen:
                duplicates.append(document.doc_id)
            seen.add(document.doc_id)
        if duplicates:
            listed = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"duplicate doc_id values: {listed}")
        return self


class ManifestEntry(BaseModel):
    """One row of ``data/raw/manifest.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    title: str
    category: str
    effective_date: date | None = None
    local_path: str
    sha256: str | None = None
    bytes: int | None = None
    source_url: str | None = None
    status: DocStatus
    error: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Outcome of ``rag-eval corpus scan-local``."""

    config_path: Path
    added: list[str]
    skipped: list[tuple[str, str]]


def load_corpus_config(path: str | Path | None = None) -> CorpusConfig:
    """Load and validate ``corpus.yaml``, raising :class:`ManifestError` on failure."""
    config_path = Path(path) if path is not None else DEFAULT_CORPUS_CONFIG_PATH
    if not config_path.is_file():
        raise ManifestError(f"corpus config not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid YAML in {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ManifestError(f"{config_path} must contain a YAML mapping at the top level")

    try:
        return CorpusConfig.model_validate(raw)
    except ValidationError as exc:
        raise ManifestError(f"invalid corpus config {config_path}:\n{exc}") from exc


def render_corpus_yaml(documents: Sequence[CorpusDocument]) -> str:
    """Render the canonical file text, header included."""
    payload = {
        "schema_version": 1,
        "documents": [
            document.model_dump(mode="json", exclude_none=True) for document in documents
        ],
    }
    body = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=100,
    )
    return f"{CORPUS_HEADER}{body}"


def write_corpus_config(path: str | Path, documents: Sequence[CorpusDocument]) -> Path:
    """Atomically rewrite ``corpus.yaml`` with the canonical header."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(render_corpus_yaml(documents), encoding="utf-8", newline="\n")
    os.replace(tmp, target)
    return target


def slugify(text: str) -> str:
    """Turn a filename stem into a stable, lowercase ``doc_id``."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned[:80].strip("-") or "document"


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pdf_file(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[str, int]:
    """Validate ``%PDF-`` magic bytes, minimum size, and an optional hash.

    Returns ``(sha256, size_in_bytes)``. HTML error pages and WAF challenge
    bodies are rejected even when the HTTP status was 200.
    """
    target = Path(path)
    if not target.is_file():
        raise PdfValidationError(f"file not found: {target}")

    size = target.stat().st_size
    with target.open("rb") as handle:
        head = handle.read(len(PDF_MAGIC))
    if head != PDF_MAGIC:
        raise PdfValidationError(
            f"{target.name} is not a PDF: expected {PDF_MAGIC.decode()!r} magic bytes, "
            f"got {head[:8]!r}. An HTML error page or WAF challenge was probably saved."
        )
    if size <= MIN_PDF_BYTES:
        raise PdfValidationError(
            f"{target.name} is only {size} bytes; a valid PDF must exceed {MIN_PDF_BYTES}."
        )

    digest = sha256_file(target)
    if expected_sha256 is not None and digest != expected_sha256.strip().lower():
        raise PdfValidationError(
            f"{target.name} sha256 mismatch: expected {expected_sha256}, got {digest}."
        )
    return digest, size


def write_manifest(
    path: str | Path,
    entries: Sequence[ManifestEntry],
    *,
    raw_dir: str | Path | None = None,
) -> Path:
    """Atomically write the run manifest, including failed documents."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir if raw_dir is not None else target.parent),
        "documents": [entry.model_dump(mode="json") for entry in entries],
    }
    tmp = target.with_name(f"{target.name}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(tmp, target)
    return target


def scan_local_pdfs(raw_dir: str | Path, config_path: str | Path) -> ScanResult:
    """Register valid PDFs already present in ``raw_dir`` into ``corpus.yaml``.

    Files that are already tracked, share a tracked filename, or duplicate a
    tracked SHA-256 are skipped, making the command idempotent.
    """
    raw = Path(raw_dir)
    config = Path(config_path)
    if not raw.is_dir():
        raise ManifestError(f"raw directory not found: {raw}")

    documents = list(load_corpus_config(config).documents) if config.is_file() else []

    tracked_paths = {document.resolved_path(raw).resolve() for document in documents}
    tracked_hashes = {
        document.expected_sha256 for document in documents if document.expected_sha256
    }
    used_ids = {document.doc_id for document in documents}

    added: list[str] = []
    skipped: list[tuple[str, str]] = []

    for path in sorted(raw.glob("*.pdf")):
        if path.resolve() in tracked_paths:
            skipped.append((path.name, "already tracked"))
            continue
        try:
            digest, _size = verify_pdf_file(path)
        except PdfValidationError as exc:
            skipped.append((path.name, str(exc)))
            continue
        if digest in tracked_hashes:
            skipped.append((path.name, "duplicate content (sha256 already tracked)"))
            continue

        doc_id = _unique_slug(slugify(path.stem), used_ids)
        documents.append(
            CorpusDocument(
                doc_id=doc_id,
                title=_humanise_title(path.stem),
                local_filename=path.name,
                expected_sha256=digest,
            )
        )
        used_ids.add(doc_id)
        tracked_hashes.add(digest)
        tracked_paths.add(path.resolve())
        added.append(doc_id)

    if added or not config.is_file():
        write_corpus_config(config, documents)

    return ScanResult(config_path=config, added=added, skipped=skipped)


def _unique_slug(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    stem = base[:76].strip("-") or "document"
    index = 2
    while f"{stem}-{index}" in used:
        index += 1
    return f"{stem}-{index}"


def _humanise_title(stem: str) -> str:
    words = re.sub(r"[_\-.]+", " ", stem)
    return re.sub(r"\s+", " ", words).strip() or stem
