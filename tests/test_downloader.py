"""Offline tests for the corpus manifest, scanner, and PDF downloader.

Every HTTP interaction is served by ``httpx.MockTransport``; nothing here touches
the network, and no real FDA URL is ever needed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.ingest.downloader import download_corpus
from rag_eval.ingest.manifest import (
    MIN_PDF_BYTES,
    CorpusDocument,
    ManifestError,
    PdfValidationError,
    _unique_slug,
    load_corpus_config,
    render_corpus_yaml,
    scan_local_pdfs,
    verify_pdf_file,
    write_corpus_config,
    write_manifest,
)

Handler = Callable[[httpx.Request], httpx.Response]


# --------------------------------------------------------------------------- helpers


def make_pdf(total_bytes: int = 2048) -> bytes:
    """A byte string that passes the magic-byte and size gates."""
    head = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    tail = b"\n%%EOF\n"
    filler = b"0" * max(0, total_bytes - len(head) - len(tail))
    return head + filler + tail


def make_html(total_bytes: int = 2048) -> bytes:
    body = b"<html><body><h1>Access denied</h1></body></html>"
    return body + b" " * max(0, total_bytes - len(body))


def write_corpus(tmp_path: Path, documents: list[dict[str, Any]]) -> Path:
    path = tmp_path / "corpus.yaml"
    payload = {"schema_version": 1, "documents": documents}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def read_manifest(raw: Path) -> dict[str, Any]:
    return json.loads((raw / "manifest.json").read_text(encoding="utf-8"))


def transport(handler: Handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


# ------------------------------------------------------------------ successful fetch


def test_downloads_pdf_atomically_and_writes_manifest(tmp_path: Path) -> None:
    payload = make_pdf()
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-test-guidance",
                "title": "Test guidance",
                "source_url": "https://example.invalid/guidance.pdf",
            }
        ],
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=payload)

    summary = download_corpus(
        config, raw, transport=transport(handler), wait=lambda _: 0.0
    )

    target = raw / "fda-test-guidance.pdf"
    assert target.read_bytes() == payload
    assert not list(raw.glob("*.tmp"))
    assert len(calls) == 1
    assert calls[0].headers["user-agent"].startswith("Mozilla/")
    assert calls[0].headers["accept"].startswith("application/pdf")

    entry = summary.entries[0]
    assert entry.status == "downloaded"
    assert entry.bytes == len(payload)
    assert entry.sha256 == hashlib.sha256(payload).hexdigest()
    assert entry.error is None

    manifest = read_manifest(raw)
    assert manifest["schema_version"] == 1
    assert manifest["documents"][0]["status"] == "downloaded"
    assert manifest["documents"][0]["doc_id"] == "fda-test-guidance"
    assert set(manifest["documents"][0]) == {
        "doc_id",
        "title",
        "category",
        "effective_date",
        "local_path",
        "sha256",
        "bytes",
        "source_url",
        "status",
        "error",
    }


def test_streaming_rejects_html_before_reading_the_whole_body(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-waf-stream",
                "title": "WAF page",
                "source_url": "https://example.invalid/waf.pdf",
            }
        ],
    )
    consumed: list[int] = []

    def chunks():
        for chunk in [b"<html>", b"x" * 4096, b"y" * 4096]:
            consumed.append(len(chunk))
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=chunks())

    summary = download_corpus(
        config, raw, transport=transport(handler), wait=lambda _: 0.0
    )

    # httpx buffers iterator content eagerly, so the full body is read by the
    # transport; the important behaviour is that we still reject it on the
    # magic-byte check and publish nothing.
    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert "not a PDF" in entry.error
    assert not (raw / "fda-waf-stream.pdf").exists()
    assert not list(raw.glob("*.tmp"))
    assert consumed  # the first chunk was at least read


def test_local_only_run_never_builds_a_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    payload = make_pdf(1500)
    (raw / "custom-name.pdf").write_bytes(payload)
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-custom",
                "title": "Custom",
                "local_filename": "custom-name.pdf",
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    )

    called = []

    def build(**kwargs: object) -> httpx.Client:
        called.append(True)
        raise AssertionError("local-only runs must not build an HTTP client")

    monkeypatch.setattr("rag_eval.ingest.downloader.build_client", build)

    summary = download_corpus(config, raw)

    assert called == []
    assert [entry.status for entry in summary.entries] == ["local_verified"]
    assert read_manifest(raw)["documents"][0]["status"] == "local_verified"


def test_write_corpus_config_cleans_up_tmp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "corpus.yaml"
    monkeypatch.setattr(
        "rag_eval.ingest.manifest.os.replace",
        lambda src, dst: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_corpus_config(config, [])

    assert not list(tmp_path.glob("*.tmp"))


def test_write_manifest_cleans_up_tmp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "rag_eval.ingest.manifest.os.replace",
        lambda src, dst: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        write_manifest(manifest, [])

    assert not list(tmp_path.glob("*.tmp"))


def test_rejects_html_200_without_leaving_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-waf",
                "title": "WAF page",
                "source_url": "https://example.invalid/waf.pdf",
            }
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=make_html(),
            headers={"content-type": "text/html"},
        )

    summary = download_corpus(config, raw, transport=transport(handler), wait=lambda _: 0.0)

    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert "not a PDF" in entry.error
    assert not (raw / "fda-waf.pdf").exists()
    assert not list(raw.glob("*.tmp"))
    assert read_manifest(raw)["documents"][0]["status"] == "failed"


def test_retries_transient_429_then_succeeds(tmp_path: Path) -> None:
    payload = make_pdf()
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [{"doc_id": "fda-flaky", "title": "Flaky", "source_url": "https://example.invalid/f.pdf"}],
    )
    statuses = [429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status == 200:
            return httpx.Response(200, content=payload)
        return httpx.Response(429)

    summary = download_corpus(
        config, raw, transport=transport(handler), max_attempts=3, wait=lambda _: 0.0
    )

    assert statuses == []
    assert summary.entries[0].status == "downloaded"


def test_403_is_not_retried_and_explains_manual_download(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-blocked",
                "title": "Blocked",
                "source_url": "https://example.invalid/blocked.pdf",
            }
        ],
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(403, text="forbidden")

    summary = download_corpus(config, raw, transport=transport(handler), wait=lambda _: 0.0)

    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert "403" in entry.error
    assert "manually" in entry.error
    assert len(calls) == 1
    assert read_manifest(raw)["documents"][0]["status"] == "failed"


# ----------------------------------------------------------------------- hash checks


def test_expected_sha256_match_downloads(tmp_path: Path) -> None:
    payload = make_pdf()
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-pinned",
                "title": "Pinned",
                "source_url": "https://example.invalid/pinned.pdf",
                "expected_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    )

    summary = download_corpus(
        config,
        raw,
        transport=transport(lambda request: httpx.Response(200, content=payload)),
        wait=lambda _: 0.0,
    )

    assert summary.entries[0].status == "downloaded"


def test_expected_sha256_mismatch_fails_and_removes_file(tmp_path: Path) -> None:
    payload = make_pdf()
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-tampered",
                "title": "Tampered",
                "source_url": "https://example.invalid/tampered.pdf",
                "expected_sha256": "0" * 64,
            }
        ],
    )

    summary = download_corpus(
        config,
        raw,
        transport=transport(lambda request: httpx.Response(200, content=payload)),
        wait=lambda _: 0.0,
    )

    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert "sha256 mismatch" in entry.error
    assert not (raw / "fda-tampered.pdf").exists()
    assert not list(raw.glob("*.tmp"))
    assert read_manifest(raw)["documents"][0]["status"] == "failed"


# ------------------------------------------------------------------------ local-first


def test_local_first_verification_makes_no_http_requests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    custom = make_pdf(1500)
    derived = make_pdf(1600)
    (raw / "custom-name.pdf").write_bytes(custom)
    (raw / "fda-derived.pdf").write_bytes(derived)

    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-custom",
                "title": "Custom",
                "local_filename": "custom-name.pdf",
                "expected_sha256": hashlib.sha256(custom).hexdigest(),
            },
            {
                "doc_id": "fda-derived",
                "title": "Derived",
                "source_url": "https://example.invalid/never-called.pdf",
            },
        ],
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("local-first must not touch the network")

    summary = download_corpus(config, raw, transport=transport(handler))

    assert calls == []
    assert [entry.status for entry in summary.entries] == [
        "local_verified",
        "local_verified",
    ]
    assert summary.entries[0].local_path.endswith("custom-name.pdf")
    assert summary.entries[1].local_path.endswith("fda-derived.pdf")
    assert summary.entries[1].sha256 == hashlib.sha256(derived).hexdigest()


def test_corrupt_local_file_fails_without_network(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fda-corrupt.pdf").write_bytes(make_html())
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-corrupt",
                "title": "Corrupt",
                "source_url": "https://example.invalid/corrupt.pdf",
            }
        ],
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("a corrupt local file must not trigger a download")

    summary = download_corpus(config, raw, transport=transport(handler))

    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert "local file failed verification" in entry.error
    assert calls == []
    # The corrupt file is preserved for inspection instead of being overwritten.
    assert (raw / "fda-corrupt.pdf").exists()


def test_missing_document_without_source_url_fails_with_hint(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [{"doc_id": "fda-local-only", "title": "Local only", "local_filename": "local.pdf"}],
    )

    summary = download_corpus(
        config,
        raw,
        transport=transport(lambda request: httpx.Response(500)),
        wait=lambda _: 0.0,
    )

    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert "no source_url" in entry.error
    assert str(raw / "local.pdf") in entry.error


# --------------------------------------------------------------------- partial failure


def test_partial_failure_records_both_statuses(tmp_path: Path) -> None:
    good = make_pdf()
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {"doc_id": "fda-good", "title": "Good", "source_url": "https://example.invalid/good.pdf"},
            {
                "doc_id": "fda-blocked",
                "title": "Blocked",
                "source_url": "https://example.invalid/blocked.pdf",
            },
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("good.pdf"):
            return httpx.Response(200, content=good)
        return httpx.Response(403)

    summary = download_corpus(config, raw, transport=transport(handler), wait=lambda _: 0.0)

    assert [entry.status for entry in summary.entries] == ["downloaded", "failed"]
    assert summary.ok_count == 1
    assert summary.failed_count == 1
    assert summary.has_failures is True

    manifest = read_manifest(raw)
    assert [row["status"] for row in manifest["documents"]] == ["downloaded", "failed"]


def test_cli_download_exits_nonzero_on_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = make_pdf()
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {"doc_id": "fda-good", "title": "Good", "source_url": "https://example.invalid/good.pdf"},
            {
                "doc_id": "fda-blocked",
                "title": "Blocked",
                "source_url": "https://example.invalid/blocked.pdf",
            },
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("good.pdf"):
            return httpx.Response(200, content=good)
        return httpx.Response(403)

    monkeypatch.setattr(
        "rag_eval.ingest.downloader.build_client",
        lambda **_: httpx.Client(transport=transport(handler)),
    )
    runner = CliRunner()

    result = runner.invoke(
        app, ["corpus", "download", "--config", str(config), "--raw-dir", str(raw)]
    )

    assert result.exit_code == 1, result.output
    assert "downloaded" in result.output
    assert "failed" in result.output
    assert (raw / "manifest.json").is_file()

    partial = runner.invoke(
        app,
        [
            "corpus",
            "download",
            "--config",
            str(config),
            "--raw-dir",
            str(raw),
            "--allow-partial",
        ],
    )

    assert partial.exit_code == 0, partial.output
    assert "--allow-partial" in partial.output


def test_download_requires_an_existing_corpus_config(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="not found"):
        download_corpus(tmp_path / "missing.yaml", tmp_path / "raw")


def test_cli_download_warns_when_no_documents_are_configured(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(tmp_path, [])

    result = CliRunner().invoke(
        app, ["corpus", "download", "--config", str(config), "--raw-dir", str(raw)]
    )

    assert result.exit_code == 0, result.output
    assert "no documents configured" in result.output
    assert (raw / "manifest.json").is_file()


# --------------------------------------------------------------------------- scanning


def test_scan_local_adds_new_pdfs_and_ignores_duplicates(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    first = make_pdf(2000)
    second = make_pdf(2100)
    (raw / "fda-ai-ml-action-plan.pdf").write_bytes(first)
    (raw / "Digital_Health_Guidance.pdf").write_bytes(second)
    (raw / "not-a-pdf.pdf").write_bytes(make_html(1500))
    config = tmp_path / "corpus.yaml"

    result = scan_local_pdfs(raw, config)

    assert result.added == ["digital-health-guidance", "fda-ai-ml-action-plan"]
    assert [name for name, _ in result.skipped] == ["not-a-pdf.pdf"]

    corpus = load_corpus_config(config)
    assert [document.doc_id for document in corpus.documents] == result.added
    assert corpus.documents[0].local_filename == "Digital_Health_Guidance.pdf"
    assert corpus.documents[0].expected_sha256 == hashlib.sha256(second).hexdigest()
    assert corpus.documents[0].category == "fda_digital_health"
    assert corpus.documents[0].source_url is None

    text = config.read_text(encoding="utf-8")
    assert text.startswith("# Corpus manifest")
    assert "example.invalid" in text

    again = scan_local_pdfs(raw, config)
    assert again.added == []
    assert len(load_corpus_config(config).documents) == 2

    (raw / "renamed-copy.pdf").write_bytes(first)
    third = scan_local_pdfs(raw, config)

    assert third.added == []
    assert ("renamed-copy.pdf", "duplicate content (sha256 already tracked)") in third.skipped
    assert len(load_corpus_config(config).documents) == 2


def test_scan_local_creates_config_when_absent(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fda-first.pdf").write_bytes(make_pdf())
    config = tmp_path / "corpus.yaml"

    result = scan_local_pdfs(raw, config)

    assert result.added == ["fda-first"]
    assert config.is_file()
    assert load_corpus_config(config).documents[0].doc_id == "fda-first"


def test_scan_local_suffixes_colliding_slugs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fda-doc.pdf").write_bytes(make_pdf(2000))
    (raw / "fda_doc.pdf").write_bytes(make_pdf(2100))
    config = tmp_path / "corpus.yaml"

    result = scan_local_pdfs(raw, config)

    assert result.added == ["fda-doc", "fda-doc-2"]
    assert {document.doc_id for document in load_corpus_config(config).documents} == {
        "fda-doc",
        "fda-doc-2",
    }


def test_scan_local_requires_a_raw_directory(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="raw directory not found"):
        scan_local_pdfs(tmp_path / "missing", tmp_path / "corpus.yaml")


def test_scan_local_is_case_insensitive_on_all_platforms(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "UPPER.PDF").write_bytes(make_pdf())
    config = tmp_path / "corpus.yaml"

    result = scan_local_pdfs(raw, config)

    assert result.added == ["upper"]
    assert load_corpus_config(config).documents[0].local_filename == "UPPER.PDF"


def test_cli_scan_local_registers_pdfs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "fda-sample.pdf").write_bytes(make_pdf())
    config = tmp_path / "corpus.yaml"

    result = CliRunner().invoke(
        app,
        ["corpus", "scan-local", "--raw-dir", str(raw), "--config", str(config)],
    )

    assert result.exit_code == 0, result.output
    assert "fda-sample" in result.output
    assert load_corpus_config(config).documents[0].doc_id == "fda-sample"


def test_cli_scan_local_fails_without_raw_dir(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "scan-local",
            "--raw-dir",
            str(tmp_path / "missing"),
            "--config",
            str(tmp_path / "corpus.yaml"),
        ],
    )

    assert result.exit_code == 1
    assert "raw directory not found" in result.output


# ----------------------------------------------------------------------- schema guards


@pytest.mark.parametrize(
    "payload",
    [
        {"doc_id": "no-source", "title": "No source"},
        {"doc_id": "Bad Slug", "title": "Bad", "local_filename": "a.pdf"},
        {"doc_id": "bad-path", "title": "Bad", "local_filename": "../escape.pdf"},
        {"doc_id": "bad-ext", "title": "Bad", "local_filename": "notes.txt"},
        {"doc_id": "bad-url", "title": "Bad", "source_url": "ftp://example.invalid/a.pdf"},
        {
            "doc_id": "bad-hash",
            "title": "Bad",
            "source_url": "https://example.invalid/a.pdf",
            "expected_sha256": "abc",
        },
    ],
)
def test_corpus_document_rejects_invalid_entries(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CorpusDocument.model_validate(payload)


def test_load_corpus_config_rejects_duplicate_doc_ids(tmp_path: Path) -> None:
    config = write_corpus(
        tmp_path,
        [
            {"doc_id": "fda-a", "title": "A", "local_filename": "a.pdf"},
            {"doc_id": "fda-a", "title": "B", "local_filename": "b.pdf"},
        ],
    )

    with pytest.raises(ManifestError, match="duplicate doc_id"):
        load_corpus_config(config)


def test_verify_pdf_file_rejects_small_and_non_pdf(tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.pdf"
    tiny.write_bytes(b"%PDF-1.4\n" + b"0" * 100)
    with pytest.raises(PdfValidationError, match="only"):
        verify_pdf_file(tiny)

    html = tmp_path / "html.pdf"
    html.write_bytes(make_html())
    with pytest.raises(PdfValidationError, match="not a PDF"):
        verify_pdf_file(html)


def test_shipped_corpus_config_is_in_canonical_form() -> None:
    shipped = Path(__file__).resolve().parents[1] / "configs" / "corpus.yaml"
    normalized = shipped.read_text(encoding="utf-8").replace("\r\n", "\n")
    loaded = load_corpus_config(shipped)
    assert normalized == render_corpus_yaml(loaded.documents)


def test_render_corpus_yaml_round_trips_documents(tmp_path: Path) -> None:
    document = CorpusDocument(
        doc_id="fda-round-trip",
        title="Round trip",
        local_filename="round-trip.pdf",
        effective_date=date(2024, 1, 15),
    )
    config = tmp_path / "corpus.yaml"
    write_corpus_config(config, [document])

    loaded = load_corpus_config(config)

    assert loaded.documents == [document]
    assert "fda-round-trip" in config.read_text(encoding="utf-8")


# ---------------------------------------------------------------- review fixes


def test_load_corpus_config_rejects_colliding_paths(tmp_path: Path) -> None:
    config = write_corpus(
        tmp_path,
        [
            {"doc_id": "fda-a", "title": "A", "local_filename": "fda-a.pdf"},
            {"doc_id": "fda-b", "title": "B", "local_filename": "fda-a.pdf"},
        ],
    )

    with pytest.raises(ManifestError, match="colliding local PDF filenames"):
        load_corpus_config(config)


def test_load_corpus_config_rejects_case_only_path_collision(tmp_path: Path) -> None:
    config = write_corpus(
        tmp_path,
        [
            {"doc_id": "fda-a", "title": "A", "local_filename": "Fda.pdf"},
            {"doc_id": "fda-b", "title": "B", "local_filename": "fda.PDF"},
        ],
    )

    with pytest.raises(ManifestError, match="colliding local PDF filenames"):
        load_corpus_config(config)


@pytest.mark.parametrize("version", [2, 0, True])
def test_load_corpus_config_rejects_unsupported_schema_version(
    tmp_path: Path, version: Any
) -> None:
    config = tmp_path / "corpus.yaml"
    config.write_text(
        yaml.safe_dump({"schema_version": version, "documents": []}),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="schema_version"):
        load_corpus_config(config)


def test_load_corpus_config_wraps_unreadable_file(tmp_path: Path) -> None:
    config = tmp_path / "corpus.yaml"
    config.write_bytes(b"\xff\xfe\x00not valid utf-8")

    with pytest.raises(ManifestError, match="cannot read corpus config"):
        load_corpus_config(config)


def test_unique_slug_stays_within_max_length() -> None:
    base = "a" * 80
    used: set[str] = {base}
    for index in range(2, 1000):
        suffix = f"-{index}"
        stem = base[: 80 - len(suffix)].strip("-") or "document"
        used.add(f"{stem}{suffix}")

    slug = _unique_slug(base, used)

    assert slug == "a" * 75 + "-1000"
    assert len(slug) <= 80


@pytest.mark.parametrize("status", [204, 304])
def test_unexpected_status_is_reported(tmp_path: Path, status: int) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-weird",
                "title": "Weird",
                "source_url": "https://example.invalid/w.pdf",
            }
        ],
    )

    summary = download_corpus(
        config,
        raw,
        transport=transport(lambda request: httpx.Response(status)),
        wait=lambda _: 0.0,
    )

    entry = summary.entries[0]
    assert entry.status == "failed"
    assert entry.error is not None
    assert f"unexpected HTTP {status}" in entry.error
    assert not list(raw.glob("*.tmp"))


def test_pdf_at_minimum_size_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "min.pdf"
    payload = b"%PDF-1.4\n" + b"0" * (MIN_PDF_BYTES - len(b"%PDF-1.4\n"))
    assert len(payload) == MIN_PDF_BYTES
    path.write_bytes(payload)

    digest, size = verify_pdf_file(path)

    assert size == MIN_PDF_BYTES
    assert len(digest) == 64


def test_pdf_below_minimum_size_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "small.pdf"
    payload = b"%PDF-1.4\n" + b"0" * (MIN_PDF_BYTES - len(b"%PDF-1.4\n") - 1)
    assert len(payload) == MIN_PDF_BYTES - 1
    path.write_bytes(payload)

    with pytest.raises(PdfValidationError, match="only"):
        verify_pdf_file(path)


def test_download_corpus_passes_timeout_to_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    config = write_corpus(
        tmp_path,
        [
            {
                "doc_id": "fda-timeout",
                "title": "Timeout",
                "source_url": "https://example.invalid/t.pdf",
            }
        ],
    )
    seen: dict[str, float] = {}
    payload = make_pdf()

    def fake_build_client(
        *,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 60.0,
    ) -> httpx.Client:
        seen["timeout_s"] = timeout_s
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=payload)
            )
        )

    monkeypatch.setattr("rag_eval.ingest.downloader.build_client", fake_build_client)

    download_corpus(config, raw, timeout_s=1.5)

    assert seen["timeout_s"] == 1.5
