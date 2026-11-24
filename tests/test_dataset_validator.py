"""Offline tests for the span-anchored golden QA schema, validator, and CLI.

Every canonical document is built in memory (and, for CLI tests, written to
``tmp_path``), so nothing here touches the network or the real corpus.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.evals.dataset import CATEGORY_VALUES, CitationAnchor, GoldenQAItem
from rag_eval.evals.dataset_validator import (
    CitationError,
    build_smoke_split,
    coverage_problems,
    dataset_statistics,
    load_golden_qa,
    resolve_citation,
    validate_dataset,
)
from rag_eval.ingest.models import CanonicalDocument, CanonicalPage

FIXTURE = Path(__file__).parent / "fixtures" / "sample_golden_qa.jsonl"
SHA = "0" * 64

PAGE_1 = (
    "FDA Digital Health Guidance\n"
    "The device must include a cybersecurity  plan before marketing.\n"
    "Effective January 15, 2024, submissions require a premarket review."
)
PAGE_2 = (
    "Section 2. Labeling\n"
    "The label must state the intended use clearly.\n"
    "| Metric | Value |\n| --- | --- |\n| Recall | 0.95 |"
)
PAGE_3 = "Appendix A\nContact information is not provided in this document."


def _build_document(
    page_texts: list[str], doc_id: str = "synthetic-doc"
) -> CanonicalDocument:
    pages: list[CanonicalPage] = []
    cursor = 0
    for index, text in enumerate(page_texts):
        if index:
            cursor += 2
        start = cursor
        end = start + len(text)
        pages.append(
            CanonicalPage(
                page_num=index + 1,
                start_char=start,
                end_char=end,
                text=text,
                table_count=0,
            )
        )
        cursor = end
    full_text = "\n\n".join(page_texts)
    return CanonicalDocument(
        doc_id=doc_id,
        title="Synthetic document",
        sha256=SHA,
        page_count=len(pages),
        table_count=0,
        token_count=len(full_text.split()),
        full_text=full_text,
        pages=pages,
    )


def _write_document(docs_dir: Path, document: CanonicalDocument) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / f"{document.doc_id}.json"
    target.write_text(
        json.dumps(document.model_dump(mode="json")), encoding="utf-8"
    )
    return target


def _anchor(page_num: int, quote: str, doc_id: str = "synthetic-doc") -> CitationAnchor:
    return CitationAnchor(doc_id=doc_id, page_num=page_num, evidence_quote=quote)


def _item(qa_id: str, category: str) -> GoldenQAItem:
    if category == "unanswerable":
        return GoldenQAItem(
            qa_id=qa_id,
            question="question",
            reference_answer="answer",
            category=category,  # type: ignore[arg-type]
            unanswerable=True,
            difficulty="easy",
            citations=[],
        )
    return GoldenQAItem(
        qa_id=qa_id,
        question="question",
        reference_answer="answer",
        category=category,  # type: ignore[arg-type]
        unanswerable=False,
        difficulty="easy",
        citations=[_anchor(1, "quote")],
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# -- offset correctness ------------------------------------------------------


def test_exact_quote_offsets_are_true_indices() -> None:
    document = _build_document([PAGE_1, PAGE_2, PAGE_3])
    anchor = _anchor(1, "must include a cybersecurity")

    resolved = resolve_citation(anchor, {"synthetic-doc": document})

    assert resolved.start_char is not None and resolved.end_char is not None
    assert (
        document.full_text[resolved.start_char : resolved.end_char]
        == "must include a cybersecurity"
    )
    assert resolved.page_end == 1
    assert resolved.doc_text_sha256 is not None


def test_whitespace_flexible_quote_uses_true_indices() -> None:
    document = _build_document([PAGE_1, PAGE_2, PAGE_3])
    # Doc has a double space between "cybersecurity" and "plan".
    anchor = _anchor(1, "cybersecurity plan before marketing")

    resolved = resolve_citation(anchor, {"synthetic-doc": document})

    assert resolved.start_char is not None and resolved.end_char is not None
    slice_text = document.full_text[resolved.start_char : resolved.end_char]
    assert slice_text == "cybersecurity  plan before marketing"
    assert _normalise(slice_text) == "cybersecurity plan before marketing"


def test_ambiguous_quote_on_same_page_is_rejected() -> None:
    document = _build_document(["The rule applies. The rule applies."])
    anchor = _anchor(1, "The rule applies.")

    with pytest.raises(CitationError, match="appears 2 times"):
        resolve_citation(anchor, {"synthetic-doc": document})


def test_page_out_of_range_is_reported() -> None:
    document = _build_document(["only one page"])
    with pytest.raises(CitationError, match="out of range"):
        resolve_citation(_anchor(9, "one page"), {"synthetic-doc": document})


def test_missing_document_is_reported() -> None:
    with pytest.raises(CitationError, match="not parsed"):
        resolve_citation(_anchor(1, "x", doc_id="nope"), {})


# -- wrong page & cross page -------------------------------------------------


def test_wrong_page_suggests_the_real_page() -> None:
    document = _build_document(
        ["Unique alpha phrase here.", "Another beta phrase there."]
    )
    anchor = _anchor(1, "beta phrase")

    with pytest.raises(CitationError, match="found on page 2"):
        resolve_citation(anchor, {"synthetic-doc": document})


def test_cross_page_quote_resolves_and_sets_page_end() -> None:
    document = _build_document(
        ["This sentence continues across", "the page break clearly."]
    )
    anchor = _anchor(1, "continues across the page break")

    resolved = resolve_citation(anchor, {"synthetic-doc": document})

    assert resolved.page_end == 2
    assert resolved.start_char is not None and resolved.end_char is not None
    assert resolved.start_char < document.pages[0].end_char
    assert resolved.end_char > document.pages[0].end_char
    slice_text = document.full_text[resolved.start_char : resolved.end_char]
    assert _normalise(slice_text) == "continues across the page break"


# -- model invariants --------------------------------------------------------


def test_unanswerable_category_must_match_flag() -> None:
    with pytest.raises(ValidationError, match="must agree"):
        GoldenQAItem(
            qa_id="qa_x",
            question="q",
            reference_answer="a",
            category="unanswerable",
            unanswerable=False,
            difficulty="easy",
            citations=[_anchor(1, "quote")],
        )


def test_unanswerable_item_cannot_have_citations() -> None:
    with pytest.raises(ValidationError, match="must not have citations"):
        GoldenQAItem(
            qa_id="qa_x",
            question="q",
            reference_answer="a",
            category="unanswerable",
            unanswerable=True,
            difficulty="easy",
            citations=[_anchor(1, "quote")],
        )


def test_answerable_item_needs_a_citation() -> None:
    with pytest.raises(ValidationError, match="at least one citation"):
        GoldenQAItem(
            qa_id="qa_x",
            question="q",
            reference_answer="a",
            category="single_fact",
            unanswerable=False,
            difficulty="easy",
            citations=[],
        )


def test_whitespace_only_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldenQAItem(
            qa_id="qa_x",
            question="   ",
            reference_answer="a",
            category="single_fact",
            unanswerable=False,
            difficulty="easy",
            citations=[_anchor(1, "quote")],
        )
    with pytest.raises(ValidationError):
        CitationAnchor(doc_id="d", page_num=1, evidence_quote="   ")


# -- loading -----------------------------------------------------------------


def test_load_reports_one_based_line_numbers(tmp_path: Path) -> None:
    valid = json.dumps(
        {
            "qa_id": "qa_001",
            "question": "q",
            "reference_answer": "a",
            "category": "single_fact",
            "unanswerable": False,
            "difficulty": "easy",
            "citations": [
                {"doc_id": "d", "page_num": 1, "evidence_quote": "quote"}
            ],
        }
    )
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{valid}\n{{not json\n", encoding="utf-8")

    loaded, problems = load_golden_qa(path)

    assert len(loaded) == 1
    assert len(problems) == 1
    assert problems[0].line_number == 2
    assert "invalid JSON" in problems[0].message


def test_duplicate_qa_id_is_rejected(tmp_path: Path) -> None:
    line = json.dumps(
        {
            "qa_id": "qa_001",
            "question": "q",
            "reference_answer": "a",
            "category": "single_fact",
            "unanswerable": False,
            "difficulty": "easy",
            "citations": [
                {"doc_id": "d", "page_num": 1, "evidence_quote": "quote"}
            ],
        }
    )
    path = tmp_path / "golden.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    loaded, problems = load_golden_qa(path)

    assert len(loaded) == 1
    assert len(problems) == 1
    assert problems[0].line_number == 2
    assert "duplicate qa_id" in problems[0].message


# -- smoke split -------------------------------------------------------------


def test_smoke_split_is_deterministic_and_stratified() -> None:
    items = [
        _item(f"qa_{category[:3]}_{index:02d}", category)
        for category in CATEGORY_VALUES
        for index in range(4)
    ]

    first = build_smoke_split(items, smoke_size=15, seed=42)
    second = build_smoke_split(items, smoke_size=15, seed=42)

    assert first == second
    assert first == sorted(first)
    assert len(first) == 15
    per_category: Counter[str] = Counter(
        str(next(item.category for item in items if item.qa_id == qa_id))
        for qa_id in first
    )
    assert all(per_category[category] == 3 for category in CATEGORY_VALUES)


def test_smoke_split_degrades_gracefully() -> None:
    items = [_item(f"qa_{index:03d}", "single_fact") for index in range(3)]

    smoke = build_smoke_split(items, smoke_size=15, seed=42)

    assert sorted(smoke) == ["qa_000", "qa_001", "qa_002"]


def test_coverage_problems_flag_small_datasets() -> None:
    items = [_item("qa_001", "single_fact")]

    problems = coverage_problems(items, min_items=100, min_per_category=3)

    assert any("at least 100" in problem for problem in problems)
    assert any("'unanswerable'" in problem for problem in problems)


def test_dataset_statistics_counts() -> None:
    items = [
        _item("qa_001", "single_fact"),
        _item("qa_002", "single_fact"),
        _item("qa_003", "unanswerable"),
    ]

    stats = dataset_statistics(items)

    assert stats.total == 3
    assert stats.by_category["single_fact"] == 2
    assert stats.by_category["unanswerable"] == 1
    assert stats.unanswerable == 1
    assert stats.citation_counts[0] == 1
    assert stats.citation_counts[1] == 2


# -- end-to-end validator ----------------------------------------------------


def test_validate_fixture_resolves_all_citations(tmp_path: Path) -> None:
    document = _build_document([PAGE_1, PAGE_2, PAGE_3])
    docs_dir = tmp_path / "docs"
    _write_document(docs_dir, document)

    result = validate_dataset(FIXTURE, docs_dir, smoke_size=5)

    assert len(result.items) == 5
    assert result.problems == []
    assert len(result.resolved) == 5
    assert len(result.smoke_ids) == 5

    for item in result.resolved:
        for citation in item.citations:
            assert citation.start_char is not None
            assert citation.end_char is not None
            assert citation.doc_text_sha256 is not None
            slice_text = document.full_text[citation.start_char : citation.end_char]
            assert _normalise(slice_text) == _normalise(citation.evidence_quote)


def test_validate_reports_unresolvable_citation(tmp_path: Path) -> None:
    document = _build_document([PAGE_1])
    docs_dir = tmp_path / "docs"
    _write_document(docs_dir, document)

    dataset = tmp_path / "golden.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "qa_id": "qa_bad",
                "question": "q",
                "reference_answer": "a",
                "category": "single_fact",
                "unanswerable": False,
                "difficulty": "easy",
                "citations": [
                    {
                        "doc_id": "synthetic-doc",
                        "page_num": 1,
                        "evidence_quote": "text that is not present",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = validate_dataset(dataset, docs_dir)

    assert len(result.items) == 1
    assert len(result.problems) == 1
    assert result.problems[0].line_number == 1
    assert result.problems[0].qa_id == "qa_bad"
    assert result.resolved == []


# -- CLI ---------------------------------------------------------------------


def _write_cli_docs(tmp_path: Path) -> Path:
    document = _build_document([PAGE_1, PAGE_2, PAGE_3])
    docs_dir = tmp_path / "docs"
    _write_document(docs_dir, document)
    return docs_dir


def _combined(result: Any) -> str:
    return result.output + (getattr(result, "stderr", "") or "")


def test_cli_validate_writes_resolved_and_smoke(tmp_path: Path) -> None:
    docs_dir = _write_cli_docs(tmp_path)
    out = tmp_path / "eval" / "golden_qa.resolved.json"

    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "validate",
            "--input",
            str(FIXTURE),
            "--docs-dir",
            str(docs_dir),
            "--out",
            str(out),
        ],
    )

    assert result.exit_code == 0, _combined(result)
    assert out.is_file()
    assert (out.parent / "smoke_qa_ids.json").is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload) == 5
    assert all(entry["citations"] or entry["unanswerable"] for entry in payload)


def test_cli_validate_strict_fails_on_small_dataset(tmp_path: Path) -> None:
    docs_dir = _write_cli_docs(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "validate",
            "--input",
            str(FIXTURE),
            "--docs-dir",
            str(docs_dir),
            "--out",
            str(tmp_path / "resolved.json"),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    assert "at least 100" in _combined(result)


def test_cli_validate_missing_input_fails(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "dataset",
            "validate",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--docs-dir",
            str(tmp_path / "docs"),
        ],
    )

    assert result.exit_code == 1
    assert "dataset not found" in _combined(result)


def test_cli_stats_reads_jsonl_directly() -> None:
    result = CliRunner().invoke(
        app, ["dataset", "stats", "--input", str(FIXTURE)]
    )

    assert result.exit_code == 0, _combined(result)
    assert "single_fact" in result.output
    assert "unanswerable" in result.output
    assert "total 5" in result.output


def test_annotator_module_imports_without_streamlit() -> None:
    import importlib

    module = importlib.import_module("rag_eval.tools.annotator_app")

    assert callable(module.main)
