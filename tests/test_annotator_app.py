"""Offline tests for the annotator workbench.

The pure helpers never import streamlit, so they run in the default (offline)
CI job. The ``streamlit.testing.v1.AppTest`` smoke tests additionally need the
optional ``ui`` extra and are skipped when streamlit is not installed (the CI
``ui`` job runs this file with ``uv sync --frozen --extra ui``).
"""

from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from rag_eval.ingest.models import CanonicalDocument, CanonicalPage

SHA = "0" * 64
PAGE_TEXT = "The device must include a cybersecurity plan before marketing."


def _module() -> Any:
    return importlib.import_module("rag_eval.tools.annotator_app")


def _document() -> CanonicalDocument:
    page = CanonicalPage(
        page_num=1, start_char=0, end_char=len(PAGE_TEXT), text=PAGE_TEXT, table_count=0
    )
    return CanonicalDocument(
        doc_id="synthetic-doc",
        title="Synthetic document",
        sha256=SHA,
        page_count=1,
        table_count=0,
        token_count=len(PAGE_TEXT.split()),
        full_text=PAGE_TEXT,
        pages=[page],
    )


def _write_doc(docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "synthetic-doc.json").write_text(
        json.dumps(_document().model_dump(mode="json")), encoding="utf-8"
    )


def _valid_item(qa_id: str = "qa_001") -> dict[str, Any]:
    return {
        "qa_id": qa_id,
        "question": f"question for {qa_id}",
        "reference_answer": "answer",
        "category": "unanswerable",
        "unanswerable": True,
        "difficulty": "easy",
        "citations": [],
        "notes": "",
    }


# -- pure helpers (no streamlit required) --------------------------------------


def test_atomic_write_jsonl_uses_unique_same_dir_tmp_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    calls: list[tuple[Path, str]] = []
    real_mkstemp = tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any) -> tuple[int, str]:
        fd, name = real_mkstemp(*args, **kwargs)
        calls.append((Path(kwargs["dir"]), Path(name).name))
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    target = tmp_path / "nested" / "golden.jsonl"
    module._atomic_write_jsonl(target, [{"qa_id": "qa_001"}])
    module._atomic_write_jsonl(target, [{"qa_id": "qa_002"}, {"qa_id": "qa_003"}])

    assert len(calls) == 2
    for directory, name in calls:
        assert directory == tmp_path / "nested"
        assert name.startswith("golden.jsonl.")
        assert name.endswith(".tmp")
    # Unique temp names instead of a fixed '<name>.tmp'.
    assert calls[0][1] != calls[1][1]
    # No temp file is left behind next to the target.
    assert sorted(path.name for path in (tmp_path / "nested").iterdir()) == [
        "golden.jsonl"
    ]
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"qa_id": "qa_002"}, {"qa_id": "qa_003"}]


def test_next_qa_id_skips_ids_already_used() -> None:
    module = _module()
    items = [{"qa_id": "qa_001"}, {"qa_id": "qa_002"}, {"qa_id": "qa_004"}]

    assert module._next_qa_id(items) == "qa_003"
    assert module._next_qa_id([]) == "qa_001"


def test_next_qa_id_ignores_non_string_and_unhashable_qa_ids() -> None:
    module = _module()
    items: list[dict[str, Any]] = [
        {"qa_id": ["not", "hashable"]},
        {"qa_id": {"nested": "dict"}},
        {"qa_id": 42},
        {"qa_id": None},
        {},
        {"qa_id": "qa_001"},
    ]

    assert module._next_qa_id(items) == "qa_002"
    assert module._next_qa_id([{"qa_id": 7}]) == "qa_001"


def test_read_raw_entries_keeps_line_numbers_and_skips_blank_lines(
    tmp_path: Path,
) -> None:
    module = _module()
    path = tmp_path / "golden.jsonl"
    path.write_text('\n{"qa_id": "a"}\n   \n{"qa_id": "b"}\n', encoding="utf-8")

    assert module._read_raw_entries(path) == [(2, {"qa_id": "a"}), (4, {"qa_id": "b"})]
    assert module._read_raw_items(path) == [{"qa_id": "a"}, {"qa_id": "b"}]
    assert module._read_raw_entries(tmp_path / "missing.jsonl") == []


def test_build_payload_copies_citations_for_answerable_items() -> None:
    module = _module()
    draft = {
        "qa_id": "qa_001",
        "question": "q",
        "reference_answer": "a",
        "category": "single_fact",
        "difficulty": "easy",
        "notes": "n",
    }
    citations = [{"doc_id": "synthetic-doc", "page_num": 1, "evidence_quote": "quote"}]

    payload = module._build_payload(draft, citations)

    assert payload == {
        "qa_id": "qa_001",
        "question": "q",
        "reference_answer": "a",
        "category": "single_fact",
        "unanswerable": False,
        "difficulty": "easy",
        "citations": [{"doc_id": "synthetic-doc", "page_num": 1, "evidence_quote": "quote"}],
        "notes": "n",
    }
    assert payload["citations"] is not citations
    assert payload["citations"][0] is not citations[0]


def test_build_payload_drops_citations_for_unanswerable_items() -> None:
    module = _module()
    draft = {
        "qa_id": "qa_001",
        "question": "q",
        "reference_answer": "a",
        "category": "unanswerable",
        "difficulty": "hard",
        "notes": "",
    }
    citations = [{"doc_id": "synthetic-doc", "page_num": 1, "evidence_quote": "quote"}]

    payload = module._build_payload(draft, citations)

    assert payload["unanswerable"] is True
    assert payload["citations"] == []


# -- AppTest smoke tests (require the ui extra) --------------------------------


def _run_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dataset_source: str
) -> tuple[Any, Path]:
    """Run the real app script from ``tmp_path`` (its default paths are relative)."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    module = _module()
    monkeypatch.chdir(tmp_path)
    _write_doc(tmp_path / "data" / "processed" / "docs")
    dataset = tmp_path / "data" / "eval" / "golden_qa.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(dataset_source, encoding="utf-8")
    at = AppTest.from_file(str(Path(module.__file__)), default_timeout=10)
    at.run()
    return at, dataset


def _fill_new_draft(at: Any, qa_id: str) -> None:
    qa_input = next(ti for ti in at.text_input if ti.label == "qa_id")
    qa_input.set_value(qa_id)
    question = next(ta for ta in at.text_area if ta.label == "Question")
    question.set_value("a brand new question")
    answer = next(ta for ta in at.text_area if ta.label == "Reference answer")
    answer.set_value("a brand new answer")
    category = next(sb for sb in at.selectbox if sb.label == "Category")
    category.set_value("unanswerable")
    at.run()


def test_apptest_unreadable_dataset_shows_error_and_leaves_file_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = '{"qa_id": "kept"}\nnull\n'
    at, dataset = _run_app(monkeypatch, tmp_path, source)

    assert not at.exception
    assert len(at.error) == 1
    assert "line 2" in at.error[0].value
    assert "golden_qa.jsonl" in at.error[0].value
    assert dataset.read_text(encoding="utf-8") == source


def test_apptest_schema_invalid_item_warns_and_is_excluded_from_dropdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = json.dumps(_valid_item("qa_001")) + "\n" + '{"qa_id": "qa_999", "q": 1}\n'
    at, _ = _run_app(monkeypatch, tmp_path, source)

    assert not at.exception
    warnings = [warning.value for warning in at.warning]
    assert any("qa_999" in message and "line 2" in message for message in warnings)
    load = next(sb for sb in at.selectbox if sb.label == "Load existing item")
    assert "qa_001" in load.options
    assert "qa_999" not in load.options


def test_apptest_save_preserves_schema_invalid_lines_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_line = '{"qa_id": "qa_777", "question": "broken"}\n'
    at, dataset = _run_app(monkeypatch, tmp_path, invalid_line)
    assert not at.exception
    assert any("qa_777" in warning.value for warning in at.warning)

    _fill_new_draft(at, "qa_001")
    save = next(button for button in at.button if button.label == "Save item")
    save.click().run()

    assert not at.error
    lines = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    # The schema-invalid line is still there, byte-for-byte semantics preserved.
    assert lines[0] == {"qa_id": "qa_777", "question": "broken"}
    assert lines[1]["qa_id"] == "qa_001"
    assert lines[1]["unanswerable"] is True


def test_apptest_new_draft_reusing_an_existing_qa_id_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = json.dumps(_valid_item("qa_001")) + "\n"
    at, dataset = _run_app(monkeypatch, tmp_path, source)
    assert not at.exception

    _fill_new_draft(at, "qa_001")
    save = next(button for button in at.button if button.label == "Save item")
    save.click().run()

    assert any(
        "qa_001" in error.value and "load the existing item" in error.value
        for error in at.error
    )
    assert dataset.read_text(encoding="utf-8") == source
