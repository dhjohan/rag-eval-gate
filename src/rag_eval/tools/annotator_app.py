"""Local Streamlit annotation workbench for hand-written golden QA items.

Run it with the ``ui`` extra installed::

    uv sync --extra ui
    uv run streamlit run src/rag_eval/tools/annotator_app.py

Streamlit is imported lazily inside :func:`main`, so importing ``rag_eval`` (or
this module) never requires the optional dependency. Nothing here calls a model
or the network: citations are resolved against the local canonical JSON using
exactly the same logic as ``rag-eval dataset validate``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rag_eval.evals.dataset import (
    CATEGORY_VALUES,
    DIFFICULTY_VALUES,
    CitationAnchor,
    GoldenQAItem,
)
from rag_eval.evals.dataset_validator import (
    CitationError,
    DatasetError,
    load_documents,
    resolve_citation,
)
from rag_eval.ingest.models import CanonicalDocument

DEFAULT_DOCS_DIR = Path("data/processed/docs")
DEFAULT_DATASET = Path("data/eval/golden_qa.jsonl")
TARGET_MIN = 100
TARGET_MAX = 120
UNANSWERABLE_TARGET = 12


class AnnotatorDataError(RuntimeError):
    """Raised when the source JSONL cannot be read without losing data."""


def _read_raw_entries(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read the JSONL source as ``(1-based line number, raw dict)`` pairs."""
    if not path.is_file():
        return []
    entries: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AnnotatorDataError(
                f"invalid JSON in {path} on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(data, dict):
            raise AnnotatorDataError(
                f"expected a JSON object in {path} on line {line_number}"
            )
        entries.append((line_number, data))
    return entries


def _read_raw_items(path: Path) -> list[dict[str, Any]]:
    return [item for _, item in _read_raw_entries(path)]


def _atomic_write_jsonl(path: Path, items: Sequence[dict[str, Any]]) -> None:
    """Write JSONL atomically via a unique same-directory temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def _next_qa_id(items: Sequence[dict[str, Any]]) -> str:
    """First free ``qa_NNN`` id; non-string/unhashable ``qa_id`` values are ignored."""
    used: set[str] = set()
    for item in items:
        qa_id = item.get("qa_id")
        if isinstance(qa_id, str):
            used.add(qa_id)
    index = 1
    while f"qa_{index:03d}" in used:
        index += 1
    return f"qa_{index:03d}"


def _blank_draft(qa_id: str, doc_id: str) -> dict[str, Any]:
    return {
        "qa_id": qa_id,
        "question": "",
        "reference_answer": "",
        "category": "single_fact",
        "difficulty": "medium",
        "notes": "",
        "doc_id": doc_id,
    }


def _build_payload(draft: dict[str, Any], citations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    category = draft.get("category", "single_fact")
    unanswerable = category == "unanswerable"
    payload_citations = (
        [] if unanswerable else [dict(citation) for citation in citations]
    )
    return {
        "qa_id": draft.get("qa_id", ""),
        "question": draft.get("question", ""),
        "reference_answer": draft.get("reference_answer", ""),
        "category": category,
        "unanswerable": unanswerable,
        "difficulty": draft.get("difficulty", "medium"),
        "citations": payload_citations,
        "notes": draft.get("notes", ""),
    }


def _format_validation_error(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts: list[str] = []
        for error in errors():
            location = ".".join(str(part) for part in error.get("loc", ()))
            message = error.get("msg", "invalid value")
            parts.append(f"{location}: {message}" if location else message)
        return "; ".join(parts)
    return str(exc)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Golden QA Annotator", layout="wide")

    docs_dir = DEFAULT_DOCS_DIR
    dataset_path = DEFAULT_DATASET

    @st.cache_resource(show_spinner=False)
    def _load_docs(dir_str: str) -> dict[str, CanonicalDocument]:
        return load_documents(dir_str)

    try:
        documents = _load_docs(str(docs_dir))
    except DatasetError as exc:
        st.error(str(exc))
        st.stop()
        return

    st.title("Golden QA Annotation Workbench")
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    if st.session_state.get("pending_load_warning"):
        st.warning(st.session_state.pop("pending_load_warning"))

    try:
        entries = _read_raw_entries(dataset_path)
    except AnnotatorDataError as exc:
        st.error(str(exc))
        st.stop()
        return
    existing = [item for _, item in entries]
    doc_ids = sorted(documents)
    if not doc_ids:
        st.error("No parsed documents found. Run `rag-eval corpus parse` first.")
        st.stop()
        return

    # Schema-invalid lines stay in the file untouched, but they cannot be
    # edited here: they are excluded from the load dropdown and reported once.
    loadable: dict[str, dict[str, Any]] = {}
    for line_number, raw in entries:
        try:
            item = GoldenQAItem.model_validate(raw)
        except ValidationError as exc:
            qa_id = raw.get("qa_id")
            label = qa_id if isinstance(qa_id, str) else repr(qa_id)
            st.warning(
                f"line {line_number} (qa_id={label!r}) is not a valid golden QA item: "
                f"{_format_validation_error(exc)}; kept as-is but excluded from the "
                "load list"
            )
            continue
        loadable[item.qa_id] = raw

    def _snapshot() -> str:
        return json.dumps(
            {
                "draft": st.session_state.get("draft"),
                "citations": st.session_state.get("citations"),
            },
            sort_keys=True,
            default=str,
        )

    if "token" not in st.session_state:
        st.session_state.token = 0
    if "draft" not in st.session_state:
        st.session_state.draft = _blank_draft(_next_qa_id(existing), doc_ids[0])
        st.session_state.draft_source = "<new>"
    if "citations" not in st.session_state:
        st.session_state.citations = [
            {"doc_id": doc_ids[0], "page_num": 1, "evidence_quote": ""}
        ]
    if "draft_baseline" not in st.session_state:
        st.session_state.draft_baseline = _snapshot()
    token = st.session_state.token
    draft: dict[str, Any] = st.session_state.draft

    # -- sidebar progress ---------------------------------------------------
    counts: Counter[str] = Counter(
        str(item["category"])
        for item in existing
        if isinstance(item.get("category"), str)
    )
    total = len(existing)
    with st.sidebar:
        st.header("Progress")
        st.progress(
            min(total / TARGET_MAX, 1.0),
            text=f"{total} / {TARGET_MAX} items (target {TARGET_MIN}-{TARGET_MAX})",
        )
        for category in CATEGORY_VALUES:
            st.write(f"{category}: {counts.get(category, 0)}")
        st.write(
            f"unanswerable: {counts.get('unanswerable', 0)} / ~{UNANSWERABLE_TARGET}"
        )
        st.caption(f"docs: {docs_dir}")
        st.caption(f"dataset: {dataset_path}")

    left, right = st.columns(2)

    # -- left: source document ---------------------------------------------
    with left:
        st.subheader("Source document")
        selected_doc = st.selectbox(
            "Document",
            doc_ids,
            index=doc_ids.index(draft.get("doc_id", doc_ids[0]))
            if draft.get("doc_id") in doc_ids
            else 0,
            key=f"selected_doc_{token}",
        )
        document = documents[selected_doc]
        page_numbers = [page.page_num for page in document.pages]
        first_citation = (
            st.session_state.citations[0] if st.session_state.citations else {}
        )
        wanted_page = (
            int(first_citation.get("page_num", page_numbers[0]))
            if first_citation.get("doc_id") == selected_doc
            else page_numbers[0]
        )
        page_index = page_numbers.index(wanted_page) if wanted_page in page_numbers else 0
        selected_page = st.selectbox(
            "Page", page_numbers, index=page_index, key=f"selected_page_{token}"
        )
        search = st.text_input("Search this document", key=f"search_{token}")
        if search:
            needle = search.lower()
            hits = [
                page.page_num
                for page in document.pages
                if needle in page.text.lower()
            ]
            if hits:
                st.caption("matches on pages: " + ", ".join(str(n) for n in hits[:20]))
            else:
                st.caption("no matches")
        page = document.pages[page_numbers.index(selected_page)]
        st.caption(f"page {page.page_num} of {document.page_count} — raw text")
        st.code(page.text or "(empty page)", language=None)

    # -- right: annotation form --------------------------------------------
    with right:
        st.subheader("Golden QA item")

        existing_ids: list[str] = sorted(loadable)
        load_choice = st.selectbox(
            "Load existing item",
            ["<new>", *existing_ids],
            key=f"load_choice_{token}",
        )

        def _apply_load(item: dict[str, Any] | None) -> None:
            if item is None:
                return
            if _snapshot() != st.session_state.get("draft_baseline"):
                st.session_state.pending_load_warning = (
                    "Discarded unsaved changes in the current draft while loading "
                    f"{item.get('qa_id')!r}."
                )
            st.session_state.draft = {
                "qa_id": item.get("qa_id", ""),
                "question": item.get("question", ""),
                "reference_answer": item.get("reference_answer", ""),
                "category": item.get("category", "single_fact"),
                "difficulty": item.get("difficulty", "medium"),
                "notes": item.get("notes", ""),
                "doc_id": (item.get("citations") or [{}])[0].get("doc_id", doc_ids[0]),
            }
            loaded_citations = [dict(c) for c in item.get("citations", [])]
            st.session_state.citations = loaded_citations or [
                {"doc_id": doc_ids[0], "page_num": 1, "evidence_quote": ""}
            ]
            st.session_state.draft_source = str(item.get("qa_id", ""))
            st.session_state.draft_baseline = _snapshot()
            st.session_state.token += 1

        if load_choice != "<new>":
            st.button(
                "Load selected item",
                key=f"load_btn_{token}",
                on_click=_apply_load,
                args=(loadable.get(load_choice),),
            )

        draft["qa_id"] = st.text_input(
            "qa_id", value=draft.get("qa_id", ""), key=f"qa_id_{token}"
        )
        draft["question"] = st.text_area(
            "Question", value=draft.get("question", ""), key=f"question_{token}", height=80
        )
        draft["reference_answer"] = st.text_area(
            "Reference answer",
            value=draft.get("reference_answer", ""),
            key=f"answer_{token}",
            height=80,
        )
        draft["category"] = st.selectbox(
            "Category",
            CATEGORY_VALUES,
            index=CATEGORY_VALUES.index(draft.get("category", "single_fact"))
            if draft.get("category") in CATEGORY_VALUES
            else 0,
            key=f"category_{token}",
        )
        draft["difficulty"] = st.selectbox(
            "Difficulty",
            DIFFICULTY_VALUES,
            index=DIFFICULTY_VALUES.index(draft.get("difficulty", "medium"))
            if draft.get("difficulty") in DIFFICULTY_VALUES
            else 1,
            key=f"difficulty_{token}",
        )
        draft["notes"] = st.text_input(
            "Notes", value=draft.get("notes", ""), key=f"notes_{token}"
        )

        unanswerable = draft["category"] == "unanswerable"
        st.markdown("**Citations**")
        if unanswerable:
            st.info("Unanswerable items carry no citations; the citation editor is disabled.")
        else:
            for index, citation in enumerate(st.session_state.citations):
                doc_col, page_col, quote_col, remove_col = st.columns([2, 1, 5, 1])
                citation["doc_id"] = doc_col.selectbox(
                    "doc_id",
                    doc_ids,
                    index=doc_ids.index(citation.get("doc_id", doc_ids[0]))
                    if citation.get("doc_id") in doc_ids
                    else 0,
                    key=f"cit_doc_{token}_{index}",
                )
                citation["page_num"] = int(
                    page_col.number_input(
                        "page",
                        min_value=1,
                        value=int(citation.get("page_num", 1)),
                        step=1,
                        key=f"cit_page_{token}_{index}",
                    )
                )
                citation["evidence_quote"] = quote_col.text_area(
                    "evidence_quote",
                    value=citation.get("evidence_quote", ""),
                    key=f"cit_quote_{token}_{index}",
                    height=68,
                )
                if remove_col.button("✕", key=f"cit_remove_{token}_{index}"):
                    st.session_state.citations.pop(index)
                    st.session_state.token += 1
                    st.rerun()

            if st.button("Add citation", key=f"cit_add_{token}"):
                st.session_state.citations.append(
                    {
                        "doc_id": selected_doc,
                        "page_num": selected_page,
                        "evidence_quote": "",
                    }
                )
                st.session_state.token += 1
                st.rerun()

        verify_col, save_col = st.columns(2)

        def _verify() -> None:
            if unanswerable:
                st.success("Unanswerable item: no citations required.")
                return
            citations = st.session_state.citations
            if not citations:
                st.error("An answerable item needs at least one citation.")
                return
            for number, citation in enumerate(citations, start=1):
                try:
                    anchor = CitationAnchor(
                        doc_id=citation.get("doc_id", ""),
                        page_num=int(citation.get("page_num", 1)),
                        evidence_quote=citation.get("evidence_quote", ""),
                    )
                    resolved = resolve_citation(anchor, documents)
                except (CitationError, ValueError) as exc:
                    st.error(f"citation {number}: {exc}")
                else:
                    st.success(
                        f"citation {number}: [{resolved.start_char}, {resolved.end_char}] "
                        f"page_end={resolved.page_end}"
                    )

        def _save() -> None:
            payload = _build_payload(st.session_state.draft, st.session_state.citations)
            try:
                item = GoldenQAItem.model_validate(payload)
            except Exception as exc:  # pydantic ValidationError
                st.error(_format_validation_error(exc))
                return
            if not item.unanswerable:
                # Validate against the canonical text, but keep writing the raw
                # anchor: `dataset validate` is the single producer of resolved
                # JSON, so the source JSONL diff stays free of offsets.
                for number, anchor in enumerate(item.citations, start=1):
                    try:
                        resolve_citation(anchor, documents)
                    except (CitationError, ValueError) as exc:
                        st.error(f"citation {number}: {exc}")
                        return
            try:
                current = _read_raw_items(dataset_path)
            except AnnotatorDataError as exc:
                st.error(str(exc))
                return
            # A draft that started as "<new>" must never silently replace an
            # existing item: the user has to load that item first.
            origin = st.session_state.get("draft_source", "<new>")
            allow_replace = origin != "<new>" and item.qa_id == origin
            if not allow_replace and any(
                raw.get("qa_id") == item.qa_id for raw in current
            ):
                st.error(
                    f"qa_id '{item.qa_id}' already exists in {dataset_path}; load the "
                    "existing item from the dropdown instead of saving a new draft "
                    "over it"
                )
                return
            replaced = False
            for index, raw in enumerate(current):
                if raw.get("qa_id") == item.qa_id:
                    current[index] = item.model_dump(mode="json")
                    replaced = True
                    break
            if not replaced:
                current.append(item.model_dump(mode="json"))
            _atomic_write_jsonl(dataset_path, current)
            st.session_state.flash = (
                f"{'updated' if replaced else 'added'} {item.qa_id}"
            )
            st.session_state.draft = _blank_draft(
                _next_qa_id(current), st.session_state.draft.get("doc_id", doc_ids[0])
            )
            st.session_state.citations = [
                {"doc_id": doc_ids[0], "page_num": 1, "evidence_quote": ""}
            ]
            st.session_state.draft_source = "<new>"
            st.session_state.draft_baseline = _snapshot()
            st.session_state.token += 1
            st.rerun()

        verify_col.button("Verify alignment", key=f"verify_{token}", on_click=_verify)
        save_col.button(
            "Save item", key=f"save_{token}", type="primary", on_click=_save
        )


if __name__ == "__main__":  # pragma: no cover - Streamlit entrypoint
    main()
