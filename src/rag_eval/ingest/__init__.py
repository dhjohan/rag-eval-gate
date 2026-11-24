"""Corpus ingestion: manifest, acquisition, and canonical PDF parsing."""

from __future__ import annotations

from rag_eval.ingest.canonical_parser import (
    CanonicalParseError,
    parse_pdf,
    write_canonical_document,
)
from rag_eval.ingest.downloader import (
    DownloadError,
    DownloadSummary,
    PdfDownloader,
    build_client,
    download_corpus,
)
from rag_eval.ingest.manifest import (
    DEFAULT_CATEGORY,
    DEFAULT_CORPUS_CONFIG_PATH,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_RAW_DIR,
    MIN_PDF_BYTES,
    PDF_MAGIC,
    CorpusConfig,
    CorpusDocument,
    ManifestEntry,
    ManifestError,
    PdfValidationError,
    ScanResult,
    load_corpus_config,
    render_corpus_yaml,
    scan_local_pdfs,
    sha256_file,
    slugify,
    verify_pdf_file,
    write_corpus_config,
    write_manifest,
)
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage

__all__ = [
    "DEFAULT_CATEGORY",
    "DEFAULT_CORPUS_CONFIG_PATH",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_RAW_DIR",
    "MIN_PDF_BYTES",
    "PDF_MAGIC",
    "CanonicalDocument",
    "CanonicalPage",
    "CanonicalParseError",
    "CorpusConfig",
    "CorpusDocument",
    "DownloadError",
    "DownloadSummary",
    "ManifestEntry",
    "ManifestError",
    "PdfDownloader",
    "PdfValidationError",
    "ScanResult",
    "build_client",
    "download_corpus",
    "load_corpus_config",
    "parse_pdf",
    "render_corpus_yaml",
    "scan_local_pdfs",
    "sha256_file",
    "slugify",
    "verify_pdf_file",
    "write_canonical_document",
    "write_corpus_config",
    "write_manifest",
]
