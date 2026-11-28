r"""Golden QA dataset validator and deterministic split generator.

The validator resolves each :class:`CitationAnchor` to exact
``[start_char, end_char]`` indices in a canonical document's ``full_text``.

Resolution rules:

1. Search the *page slice* ``full_text[page.start_char:page.end_char]`` first.
2. Fall back to a whitespace-flexible regex built token-by-token
   (``r"\s+".join(...)``) so PDF line wraps, tabs, and multi-spaces match while
   the returned indices remain true ``full_text`` offsets.
3. If the quote spans a page break, search the combined contiguous slice
   ``full_text[page.start_char:next_page.end_char]`` and set ``page_end``.
4. Ambiguity is rejected for both exact and regex matches.
5. Otherwise report where the quote actually lives (or the closest snippet).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import random
import re
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from rag_eval.evals.dataset import (
    CATEGORY_VALUES,
    CitationAnchor,
    GoldenQAItem,
)
from rag_eval.ingest.models import CanonicalDocument

DEFAULT_SMOKE_SIZE = 15
DEFAULT_SEED = 42
MIN_ITEMS = 100
MIN_PER_CATEGORY = 3


class DatasetError(RuntimeError):
    """Raised for unrecoverable dataset or canonical-document problems."""


class CitationError(RuntimeError):
    """Raised when a citation cannot be resolved to exact offsets."""


@dataclass(frozen=True)
class LoadedItem:
    line_number: int
    item: GoldenQAItem


@dataclass(frozen=True)
class DatasetProblem:
    line_number: int
    qa_id: str | None
    message: str


@dataclass
class ValidateResult:
    total_lines: int
    items: list[GoldenQAItem]
    resolved: list[GoldenQAItem]
    problems: list[DatasetProblem]
    smoke_ids: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.problems)


# -- loading ----------------------------------------------------------------


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {error['msg']}" if location else str(error["msg"]))
    return "; ".join(parts)


def load_golden_qa(path: str | Path) -> tuple[list[LoadedItem], list[DatasetProblem]]:
    """Parse ``golden_qa.jsonl``, reporting 1-based line numbers for errors."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"dataset not found: {dataset_path}")

    loaded: list[LoadedItem] = []
    problems: list[DatasetProblem] = []
    seen_ids: dict[str, int] = {}

    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(DatasetProblem(line_number, None, f"invalid JSON: {exc}"))
            continue
        if not isinstance(data, dict):
            problems.append(DatasetProblem(line_number, None, "line is not a JSON object"))
            continue
        candidate_id = data.get("qa_id")
        qa_id = candidate_id if isinstance(candidate_id, str) else None
        try:
            item = GoldenQAItem.model_validate(data)
        except ValidationError as exc:
            problems.append(DatasetProblem(line_number, qa_id, _format_validation_error(exc)))
            continue
        if item.qa_id in seen_ids:
            problems.append(
                DatasetProblem(
                    line_number,
                    item.qa_id,
                    f"duplicate qa_id (first seen on line {seen_ids[item.qa_id]})",
                )
            )
            continue
        seen_ids[item.qa_id] = line_number
        loaded.append(LoadedItem(line_number, item))
    return loaded, problems


def load_documents(docs_dir: str | Path) -> dict[str, CanonicalDocument]:
    """Load every canonical JSON in ``docs_dir`` keyed by ``doc_id``."""
    directory = Path(docs_dir)
    if not directory.is_dir():
        raise DatasetError(
            f"canonical docs directory not found: {directory}; run `rag-eval corpus parse`"
        )
    documents: dict[str, CanonicalDocument] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            document = CanonicalDocument.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (ValueError, TypeError) as exc:
            raise DatasetError(f"invalid canonical document {path.name}: {exc}") from exc
        documents[document.doc_id] = document
    return documents


# -- quote matching ---------------------------------------------------------


def _locate(slice_text: str, quote: str) -> list[tuple[int, int]]:
    """Return all ``(start, end)`` matches of ``quote`` inside ``slice_text``.

    Exact substring matches win; only when none exist is the whitespace-flexible
    regex tried. Offsets are slice-relative. Overlapping occurrences are counted
    so ambiguity is detected.
    """
    exact: list[tuple[int, int]] = []
    cursor = 0
    while True:
        index = slice_text.find(quote, cursor)
        if index == -1:
            break
        exact.append((index, index + len(quote)))
        cursor = index + 1
    if exact:
        return exact

    tokens = quote.split()
    if not tokens:
        return []
    pattern = r"\s+".join(re.escape(token) for token in tokens)
    return [
        (match.start(1), match.end(1))
        for match in re.finditer(f"(?=({pattern}))", slice_text)
    ]


def _doc_text_sha256(document: CanonicalDocument) -> str:
    return hashlib.sha256(document.full_text.encode("utf-8")).hexdigest()


def _ambiguous(anchor: CitationAnchor, count: int, suffix: str = "") -> CitationError:
    return CitationError(
        f"quote appears {count} times on page {anchor.page_num}{suffix}; "
        "make the evidence_quote slightly longer to disambiguate"
    )


def _pages_containing(document: CanonicalDocument, quote: str) -> list[int]:
    hits: list[int] = []
    for page in document.pages:
        if _locate(document.full_text[page.start_char : page.end_char], quote):
            hits.append(page.page_num)
    return hits


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _closest_match(
    document: CanonicalDocument, quote: str, *, min_ratio: float = 0.34
) -> tuple[int, int, str] | None:
    """Best-effort ``(page_num, offset, snippet)`` for a missing quote.

    A required long token narrows the candidate pages, so this stays cheap on
    large documents.
    """
    tokens = [token for token in re.split(r"\W+", quote) if len(token) >= 4]
    key = max(tokens, key=len).lower() if tokens else None
    best_ratio = 0.0
    best: tuple[int, int, str] | None = None
    for page in document.pages:
        text = document.full_text[page.start_char : page.end_char]
        if not text:
            continue
        if key is not None and key not in text.lower():
            continue
        matcher = difflib.SequenceMatcher(None, quote, text, autojunk=False)
        match = matcher.find_longest_match(0, len(quote), 0, len(text))
        ratio = match.size / max(1, len(quote))
        if ratio < min_ratio or ratio <= best_ratio:
            continue
        start = page.start_char + match.b
        snippet = document.full_text[start : start + len(quote)]
        best_ratio = ratio
        best = (page.page_num, start, snippet)
    return best


def resolve_citation(
    anchor: CitationAnchor, documents: dict[str, CanonicalDocument]
) -> CitationAnchor:
    """Return a copy of ``anchor`` with exact offsets and the text digest."""
    document = documents.get(anchor.doc_id)
    if document is None:
        raise CitationError(
            f"document '{anchor.doc_id}' is not parsed; run `rag-eval corpus parse`"
        )

    pages = {page.page_num: page for page in document.pages}
    page = pages.get(anchor.page_num)
    if page is None:
        raise CitationError(
            f"page {anchor.page_num} is out of range; '{anchor.doc_id}' has "
            f"{document.page_count} pages"
        )

    quote = anchor.evidence_quote
    digest = _doc_text_sha256(document)
    page_text = document.full_text[page.start_char : page.end_char]
    page_length = page.end_char - page.start_char

    matches = _locate(page_text, quote)
    if len(matches) == 1:
        start, end = matches[0]
        return anchor.model_copy(
            update={
                "start_char": page.start_char + start,
                "end_char": page.start_char + end,
                "page_end": anchor.page_num,
                "doc_text_sha256": digest,
            }
        )
    if len(matches) > 1:
        raise _ambiguous(anchor, len(matches))

    next_page = pages.get(anchor.page_num + 1)
    if next_page is not None:
        combined = document.full_text[page.start_char : next_page.end_char]
        spanning = [
            (start, end)
            for start, end in _locate(combined, quote)
            if start < page_length < end
        ]
        if len(spanning) == 1:
            start, end = spanning[0]
            return anchor.model_copy(
                update={
                    "start_char": page.start_char + start,
                    "end_char": page.start_char + end,
                    "page_end": anchor.page_num + 1,
                    "doc_text_sha256": digest,
                }
            )
        if len(spanning) > 1:
            raise _ambiguous(anchor, len(spanning), " across the page break")

    found_on = _pages_containing(document, quote)
    if found_on:
        page_list = ", ".join(str(number) for number in found_on[:5])
        first_page = pages[found_on[0]]
        first_matches = _locate(
            document.full_text[first_page.start_char : first_page.end_char], quote
        )
        offset = first_page.start_char + (first_matches[0][0] if first_matches else 0)
        raise CitationError(
            f"quote not found on page {anchor.page_num}; found on page {page_list} "
            f"instead (line {_line_number(document.full_text, offset)})"
        )

    closest = _closest_match(document, quote)
    if closest is not None:
        page_num, offset, snippet = closest
        raise CitationError(
            f"quote not found; closest match is on page {page_num} at "
            f"line {_line_number(document.full_text, offset)}: {snippet!r}"
        )
    raise CitationError(
        f"quote not found anywhere in '{anchor.doc_id}' (page {anchor.page_num})"
    )


# -- validation -------------------------------------------------------------


def validate_dataset(
    input_path: str | Path,
    docs_dir: str | Path,
    *,
    smoke_size: int = DEFAULT_SMOKE_SIZE,
    seed: int = DEFAULT_SEED,
) -> ValidateResult:
    """Validate the dataset and resolve every citation to exact offsets."""
    loaded, problems = load_golden_qa(input_path)
    documents = load_documents(docs_dir)

    resolved: list[GoldenQAItem] = []
    for entry in loaded:
        item = entry.item
        if item.unanswerable:
            resolved.append(item)
            continue
        resolved_citations: list[CitationAnchor] = []
        failed = False
        for index, citation in enumerate(item.citations, start=1):
            try:
                resolved_citations.append(resolve_citation(citation, documents))
            except CitationError as exc:
                problems.append(
                    DatasetProblem(
                        entry.line_number,
                        item.qa_id,
                        f"citation {index} ({citation.doc_id} p{citation.page_num}): {exc}",
                    )
                )
                failed = True
        if not failed:
            resolved.append(item.model_copy(update={"citations": resolved_citations}))

    items = [entry.item for entry in loaded]
    return ValidateResult(
        total_lines=len(loaded) + len(problems),
        items=items,
        resolved=resolved,
        problems=problems,
        smoke_ids=build_smoke_split(resolved, smoke_size=smoke_size, seed=seed),
    )


def coverage_problems(
    items: Sequence[GoldenQAItem],
    *,
    min_items: int = MIN_ITEMS,
    min_per_category: int = MIN_PER_CATEGORY,
) -> list[str]:
    """Strict-mode coverage checks (item count and per-category floors)."""
    problems: list[str] = []
    if len(items) < min_items:
        problems.append(
            f"dataset has {len(items)} items; at least {min_items} are required"
        )
    for category in CATEGORY_VALUES:
        count = sum(1 for item in items if item.category == category)
        if count < min_per_category:
            problems.append(
                f"category '{category}' has {count} items; at least "
                f"{min_per_category} are required"
            )
    return problems


# -- statistics -------------------------------------------------------------


@dataclass(frozen=True)
class DatasetStats:
    total: int
    by_category: dict[str, int]
    by_difficulty: dict[str, int]
    unanswerable: int
    docs_covered: list[str]
    citation_counts: dict[int, int]


def dataset_statistics(items: Sequence[GoldenQAItem]) -> DatasetStats:
    by_category = {category: 0 for category in CATEGORY_VALUES}
    by_difficulty: dict[str, int] = {}
    docs: set[str] = set()
    citation_counts: dict[int, int] = {}
    unanswerable = 0
    for item in items:
        by_category[item.category] = by_category.get(item.category, 0) + 1
        by_difficulty[item.difficulty] = by_difficulty.get(item.difficulty, 0) + 1
        if item.unanswerable:
            unanswerable += 1
        count = len(item.citations)
        citation_counts[count] = citation_counts.get(count, 0) + 1
        for citation in item.citations:
            docs.add(citation.doc_id)
    return DatasetStats(
        total=len(items),
        by_category=by_category,
        by_difficulty=by_difficulty,
        unanswerable=unanswerable,
        docs_covered=sorted(docs),
        citation_counts=dict(sorted(citation_counts.items())),
    )


# -- deterministic stratified split ----------------------------------------


def build_smoke_split(
    items: Sequence[GoldenQAItem],
    *,
    smoke_size: int = DEFAULT_SMOKE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    """Deterministic, category-stratified sample of ``qa_id`` values.

    Round-robin across categories (3 per category for the default 15 across 5
    categories). Degrades gracefully when a category or the whole dataset is
    smaller than ``smoke_size``: it simply returns every available id.
    """
    if smoke_size <= 0 or not items:
        return []

    rng = random.Random(seed)
    buckets: dict[str, list[str]] = {}
    for category in CATEGORY_VALUES:
        ids = sorted(item.qa_id for item in items if item.category == category)
        rng.shuffle(ids)
        buckets[category] = ids

    selected: list[str] = []
    round_index = 0
    while len(selected) < smoke_size:
        added = False
        for category in CATEGORY_VALUES:
            bucket = buckets[category]
            if round_index < len(bucket):
                selected.append(bucket[round_index])
                added = True
                if len(selected) >= smoke_size:
                    break
        if not added:
            break
        round_index += 1
    return sorted(selected)


# -- writing ----------------------------------------------------------------


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def write_resolved(path: str | Path, resolved: Sequence[GoldenQAItem]) -> Path:
    target = Path(path)
    payload = [item.model_dump(mode="json") for item in resolved]
    _atomic_write_text(target, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return target


def write_smoke(
    path: str | Path, smoke_ids: Sequence[str], *, seed: int = DEFAULT_SEED
) -> Path:
    target = Path(path)
    payload = {"seed": seed, "count": len(smoke_ids), "qa_ids": list(smoke_ids)}
    _atomic_write_text(target, json.dumps(payload, indent=2) + "\n")
    return target


__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_SMOKE_SIZE",
    "MIN_ITEMS",
    "MIN_PER_CATEGORY",
    "CitationError",
    "DatasetError",
    "DatasetProblem",
    "DatasetStats",
    "LoadedItem",
    "ValidateResult",
    "build_smoke_split",
    "coverage_problems",
    "dataset_statistics",
    "load_documents",
    "load_golden_qa",
    "resolve_citation",
    "validate_dataset",
    "write_resolved",
    "write_smoke",
]
