"""Offline tests for the canonical PDF parser and its CLI commands.

Every PDF is generated in ``tmp_path`` with pymupdf, and token counting is an
injected offline function, so nothing here touches the network or tiktoken's
blob cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pymupdf as fitz
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.ingest.canonical_parser import parse_pdf, write_canonical_document
from rag_eval.ingest.manifest import CorpusDocument, sha256_file, write_corpus_config
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage

SHA = "0" * 64


def offline_counter(text: str) -> int:
    return len(text.split())


def _parse(
    path: Path,
    *,
    doc_id: str = "test-doc",
    title: str = "Test doc",
    sha256: str = SHA,
) -> CanonicalDocument:
    return parse_pdf(
        path,
        doc_id=doc_id,
        title=title,
        sha256=sha256,
        token_counter=offline_counter,
    )


def _grid(
    page: fitz.Page,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rows: int,
    cols: int,
) -> None:
    page.draw_rect(fitz.Rect(x0, y0, x1, y1), color=(0, 0, 0), width=1)
    for col in range(1, cols):
        x = x0 + (x1 - x0) * col / cols
        page.draw_line(fitz.Point(x, y0), fitz.Point(x, y1), color=(0, 0, 0), width=1)
    for row in range(1, rows):
        y = y0 + (y1 - y0) * row / rows
        page.draw_line(fitz.Point(x0, y), fitz.Point(x1, y), color=(0, 0, 0), width=1)


def _two_by_two_table(page: fitz.Page, x0: float, y0: float) -> None:
    x1, y1 = x0 + 200, y0 + 100
    _grid(page, x0, y0, x1, y1, rows=2, cols=2)
    page.insert_text(fitz.Point(x0 + 10, y0 + 30), "A|B", fontsize=11)
    page.insert_text(fitz.Point(x0 + 110, y0 + 30), "C", fontsize=11)
    page.insert_text(fitz.Point(x0 + 110, y0 + 45), "D", fontsize=11)
    page.insert_text(fitz.Point(x0 + 10, y0 + 80), "E", fontsize=11)
    page.insert_text(fitz.Point(x0 + 110, y0 + 80), "F", fontsize=11)


# -- header / footer --------------------------------------------------------


def test_repeating_header_and_footer_are_removed(tmp_path: Path) -> None:
    path = tmp_path / "header.pdf"
    doc = fitz.open()
    for index in range(1, 5):
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(72, 30), f"Page {index} of 4", fontsize=11)
        page.insert_text(fitz.Point(72, 120), f"Body page {index}", fontsize=11)
        page.insert_text(fitz.Point(72, 760), "Confidential", fontsize=11)
    doc.save(path)
    doc.close()

    document = _parse(path)

    assert document.page_count == 4
    for canonical_page in document.pages:
        assert "Page" not in canonical_page.text
        assert "Confidential" not in canonical_page.text
        assert "Body page" in canonical_page.text


# -- reading order ----------------------------------------------------------


def test_full_width_heading_then_two_columns(tmp_path: Path) -> None:
    path = tmp_path / "two-column.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(fitz.Point(72, 60), "Full width heading spanning the page", fontsize=16)
    for index, text in enumerate(["L1", "L2", "L3"]):
        page.insert_text(fitz.Point(72, 130 + index * 22), text, fontsize=11)
    for index, text in enumerate(["R1", "R2", "R3"]):
        page.insert_text(fitz.Point(330, 130 + index * 22), text, fontsize=11)
    doc.save(path)
    doc.close()

    text = _parse(path).pages[0].text

    assert text.index("Full width heading") < text.index("L1")
    assert text.index("L3") < text.index("R1")
    assert text.index("L1") < text.index("L2") < text.index("L3")
    assert text.index("R1") < text.index("R2") < text.index("R3")


def test_indented_bullets_are_not_split_into_columns(tmp_path: Path) -> None:
    path = tmp_path / "bullets.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for index, text in enumerate(
        ["- First bullet", "- Second bullet", "- Third bullet", "- Fourth bullet"]
    ):
        page.insert_text(fitz.Point(100, 130 + index * 24), text, fontsize=11)
    doc.save(path)
    doc.close()

    text = _parse(path).pages[0].text

    positions = [text.index(word) for word in ["First", "Second", "Third", "Fourth"]]
    assert positions == sorted(positions)


# -- tables -----------------------------------------------------------------


def test_table_is_markdown_and_not_duplicated(tmp_path: Path) -> None:
    path = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _two_by_two_table(page, 100, 100)
    page.insert_text(fitz.Point(72, 300), "Caption below the table", fontsize=11)
    doc.save(path)
    doc.close()

    document = _parse(path)
    text = document.pages[0].text

    assert document.table_count == 1
    assert "| --- | --- |" in text
    assert text.count("A\\|B") == 1
    # The two-line cell is collapsed to a single space.
    assert "C D" in text
    assert "Caption below the table" in text


def test_1x1_callout_stays_prose(tmp_path: Path) -> None:
    path = tmp_path / "callout.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(100, 100, 300, 160), color=(0, 0, 0), width=1)
    page.insert_text(fitz.Point(110, 130), "Important note", fontsize=11)
    doc.save(path)
    doc.close()

    document = _parse(path)

    assert document.table_count == 0
    assert "Important note" in document.pages[0].text


# -- coordinate alignment ---------------------------------------------------


def test_cropbox_offset_alignment(tmp_path: Path) -> None:
    path = tmp_path / "crop.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    _two_by_two_table(page, 100, 100)
    page.set_cropbox(fitz.Rect(50, 50, 350, 550))
    doc.save(path)
    doc.close()

    document = _parse(path)
    text = document.pages[0].text

    assert document.table_count == 1
    assert "| A\\|B | C D |" in text
    assert "| E | F |" in text


def test_rotated_page_table_alignment(tmp_path: Path) -> None:
    path = tmp_path / "rotated.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    _two_by_two_table(page, 100, 100)
    page.set_rotation(90)
    doc.save(path)
    doc.close()

    document = _parse(path)
    text = document.pages[0].text

    assert document.table_count == 1
    for word in ["A\\|B", "C", "D", "E", "F"]:
        assert text.count(word) == 1


# -- offsets ----------------------------------------------------------------


def test_empty_page_and_offsets(tmp_path: Path) -> None:
    path = tmp_path / "empty.pdf"
    doc = fitz.open()
    first = doc.new_page(width=400, height=600)
    first.insert_text(fitz.Point(72, 100), "First page", fontsize=11)
    doc.new_page(width=400, height=600)
    third = doc.new_page(width=400, height=600)
    third.insert_text(fitz.Point(72, 100), "Third page", fontsize=11)
    doc.save(path)
    doc.close()

    document = _parse(path)

    assert document.page_count == 3
    assert document.pages[1].text == ""
    for page in document.pages:
        assert document.full_text[page.start_char : page.end_char] == page.text


def test_write_canonical_document_is_atomic(tmp_path: Path) -> None:
    document = CanonicalDocument(
        doc_id="atomic-doc",
        title="Atomic",
        sha256=SHA,
        page_count=1,
        table_count=0,
        token_count=2,
        full_text="hello",
        pages=[
            CanonicalPage(
                page_num=1, start_char=0, end_char=5, text="hello", table_count=0
            )
        ],
    )

    target = write_canonical_document(document, tmp_path / "docs")

    assert target.is_file()
    assert not list((tmp_path / "docs").glob("*.tmp"))
    assert json.loads(target.read_text(encoding="utf-8"))["doc_id"] == "atomic-doc"


# -- model invariants -------------------------------------------------------


def _document(**overrides: Any) -> CanonicalDocument:
    payload: dict[str, Any] = {
        "doc_id": "doc",
        "title": "Doc",
        "sha256": SHA,
        "page_count": 1,
        "table_count": 0,
        "token_count": 1,
        "full_text": "hello",
        "pages": [
            CanonicalPage(
                page_num=1, start_char=0, end_char=5, text="hello", table_count=0
            )
        ],
    }
    payload.update(overrides)
    return CanonicalDocument.model_validate(payload)


def test_model_rejects_broken_offsets() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        _document(
            pages=[
                CanonicalPage(
                    page_num=1, start_char=0, end_char=5, text="wrong", table_count=0
                )
            ]
        )


def test_model_rejects_non_contiguous_pages() -> None:
    with pytest.raises(ValidationError, match="contiguously"):
        _document(
            page_count=1,
            pages=[
                CanonicalPage(
                    page_num=2, start_char=0, end_char=5, text="hello", table_count=0
                )
            ],
        )


def test_model_rejects_table_count_mismatch() -> None:
    with pytest.raises(ValidationError, match="table_count"):
        _document(table_count=3)


# -- CLI --------------------------------------------------------------------


def _write_manifest(manifest: Path, pdf: Path, digest: str) -> None:
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-11-24T00:00:00+00:00",
                "raw_dir": str(pdf.parent),
                "documents": [
                    {
                        "doc_id": "fda-test",
                        "title": "Test doc",
                        "category": "fda_digital_health",
                        "effective_date": None,
                        "local_path": str(pdf),
                        "sha256": digest,
                        "bytes": pdf.stat().st_size,
                        "source_url": None,
                        "status": "local_verified",
                        "error": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _combined_output(result: Any) -> str:
    stderr = getattr(result, "stderr", "")
    return result.output + (stderr or "")


def test_cli_parse_writes_canonical_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "fda-test.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=600)
    page.insert_text(fitz.Point(72, 100), "Hello corpus", fontsize=11)
    doc.save(pdf)
    doc.close()
    digest = sha256_file(pdf)

    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, pdf, digest)
    config = tmp_path / "corpus.yaml"
    write_corpus_config(
        config,
        [
            CorpusDocument(
                doc_id="fda-test",
                title="Test doc",
                local_filename="fda-test.pdf",
                expected_sha256=digest,
            )
        ],
    )

    monkeypatch.setattr(
        "rag_eval.ingest.canonical_parser._default_token_counter",
        lambda: offline_counter,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "removed-doc.json").write_text("stale", encoding="utf-8")
    (out_dir / "fda-test.json").write_text("stale", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "parse",
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, _combined_output(result)
    target = out_dir / "fda-test.json"
    assert target.is_file()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["doc_id"] == "fda-test"
    assert data["page_count"] == 1
    assert data["full_text"].strip() == "Hello corpus"
    assert not (out_dir / "removed-doc.json").exists()


def test_cli_parse_removes_stale_output_when_current_document_fails(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    pdf = raw / "fda-test.pdf"
    pdf.write_bytes(b"temporary")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, pdf, "0" * 64)
    pdf.unlink()

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stale = out_dir / "fda-test.json"
    stale.write_text("stale", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "parse",
            "--manifest",
            str(manifest),
            "--config",
            str(tmp_path / "missing-corpus.yaml"),
            "--raw-dir",
            str(raw),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 1
    assert "missing file" in _combined_output(result)
    assert not stale.exists()


def test_cli_parse_fails_without_manifest(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "corpus",
            "parse",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert "manifest not found" in _combined_output(result)


def _write_canonical(directory: Path, doc_id: str, token_count: int) -> None:
    text = "hello"
    document = CanonicalDocument(
        doc_id=doc_id,
        title=doc_id,
        sha256=SHA,
        page_count=1,
        table_count=0,
        token_count=token_count,
        full_text=text,
        pages=[
            CanonicalPage(
                page_num=1, start_char=0, end_char=len(text), text=text, table_count=0
            )
        ],
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{doc_id}.json").write_text(
        json.dumps(document.model_dump(mode="json")), encoding="utf-8"
    )


def test_cli_stats_warning_and_alert(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    _write_canonical(docs, "small", 100)

    result = CliRunner().invoke(app, ["corpus", "stats", "--docs-dir", str(docs)])
    assert result.exit_code == 0
    assert "documents 1" in result.output

    _write_canonical(docs, "big", 600_000)
    result = CliRunner().invoke(app, ["corpus", "stats", "--docs-dir", str(docs)])
    assert result.exit_code == 0
    assert "WARN" in _combined_output(result)

    _write_canonical(docs, "huge", 800_000)
    result = CliRunner().invoke(app, ["corpus", "stats", "--docs-dir", str(docs)])
    assert result.exit_code == 0
    assert "ALERT" in _combined_output(result)

    result = CliRunner().invoke(
        app, ["corpus", "stats", "--docs-dir", str(docs), "--strict"]
    )
    assert result.exit_code == 1

    result = CliRunner().invoke(
        app,
        ["corpus", "stats", "--docs-dir", str(docs), "--max-tokens", "700000"],
    )
    assert result.exit_code == 1


def test_cli_stats_without_documents_warns(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["corpus", "stats", "--docs-dir", str(tmp_path / "empty")]
    )

    assert result.exit_code == 0
    assert "no parsed documents" in result.output
