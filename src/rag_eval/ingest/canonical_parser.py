"""Deterministic, pure-CPU canonical PDF parser.

The parser turns a PDF into a :class:`~rag_eval.ingest.models.CanonicalDocument`
whose ``full_text`` and per-page character offsets are exact, so a citation can
be resolved by slicing ``full_text``.

Pipeline per page:

1. Extract tables with ``pdfplumber`` and align their boxes into pymupdf's
   (unrotated, CropBox-relative) coordinate space.
2. Reject degenerate table detections and render the rest as Markdown.
3. Extract text at line granularity with pymupdf, excluding lines that belong to
   a validated table.
4. Detect and drop repeating headers/footers across pages.
5. Order text lines and tables with a band-segmented two-column pass.
6. Join page texts with ``"\\n\\n"`` and compute exact offsets.

No network I/O and no ML inference: ``pymupdf`` + ``pdfplumber`` are pure CPU,
and the tokenizer is injectable.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import pymupdf as fitz

from rag_eval.ingest.models import CanonicalDocument, CanonicalPage

TokenCounter = Callable[[str], int]

MARGIN_ZONE_RATIO = 0.08
HEADER_FOOTER_MIN_PAGES = 3
HEADER_FOOTER_MIN_RATIO = 0.40
PAGE_BORDER_AREA_RATIO = 0.90
PAGE_BORDER_MAX_CELLS = 2
FULL_WIDTH_RATIO = 0.70
TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "intersection_tolerance": 3,
    "snap_tolerance": 3,
    "join_tolerance": 3,
}


class CanonicalParseError(RuntimeError):
    """Raised when a PDF cannot be parsed into a canonical document."""


@dataclass(frozen=True)
class _LayoutItem:
    """A positioned piece of page content (a text line or a Markdown table)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    kind: str  # "text" | "table"


@dataclass(frozen=True)
class _TableBlock:
    rect: fitz.Rect
    markdown: str


@dataclass(frozen=True)
class _Page:
    page_num: int
    width: float
    height: float
    tables: list[_TableBlock]
    lines: list[_LayoutItem]


def _default_token_counter() -> TokenCounter:
    """Lazily build a cl100k_base counter (no import unless actually used)."""
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")

    def count(text: str) -> int:
        return len(encoding.encode(text))

    return count


def parse_pdf(
    path: str | Path,
    *,
    doc_id: str,
    title: str,
    sha256: str,
    source_url: str | None = None,
    effective_date: str | None = None,
    token_counter: TokenCounter | None = None,
) -> CanonicalDocument:
    """Parse ``path`` into a canonical document with exact page offsets."""
    pdf_path = Path(path)
    counter = token_counter or _default_token_counter()

    try:
        with fitz.open(pdf_path) as pm_doc, pdfplumber.open(pdf_path) as pl_doc:
            pages: list[_Page] = []
            for index in range(pm_doc.page_count):
                pm_page = pm_doc.load_page(index)
                pl_page = pl_doc.pages[index]
                transform = _page_transform(pm_page, pl_page)
                tables = _extract_tables(pm_page, pl_page, transform)
                lines = _extract_lines(pm_page, [table.rect for table in tables])
                pages.append(
                    _Page(
                        page_num=index + 1,
                        width=pm_page.cropbox.width,
                        height=pm_page.cropbox.height,
                        tables=tables,
                        lines=lines,
                    )
                )

            repeating = _repeating_signatures(pages)
            page_texts: list[str] = []
            table_counts: list[int] = []
            for page in pages:
                items = _filter_repeating(page.lines, page, repeating)
                items.extend(
                    _LayoutItem(
                        table.rect.x0,
                        table.rect.y0,
                        table.rect.x1,
                        table.rect.y1,
                        table.markdown,
                        "table",
                    )
                    for table in page.tables
                )
                ordered = _order_items(items, page.width)
                page_texts.append("\n".join(item.text for item in ordered))
                table_counts.append(len(page.tables))
    except CanonicalParseError:
        raise
    except Exception as exc:  # pymupdf/pdfplumber raise library-specific errors
        raise CanonicalParseError(f"failed to parse {pdf_path}: {exc}") from exc

    full_text, canonical_pages = _assemble_pages(page_texts, table_counts)
    return CanonicalDocument(
        doc_id=doc_id,
        title=title,
        source_url=source_url,
        effective_date=effective_date,
        sha256=sha256,
        page_count=len(canonical_pages),
        table_count=sum(table_counts),
        token_count=counter(full_text),
        full_text=full_text,
        pages=canonical_pages,
    )


def write_canonical_document(document: CanonicalDocument, out_dir: str | Path) -> Path:
    """Write ``document`` to ``{out_dir}/{doc_id}.json`` atomically."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{document.doc_id}.json"
    tmp = target.with_name(f"{target.name}.tmp")
    try:
        tmp.write_text(
            json.dumps(document.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp, target)
    finally:
        with suppress(OSError):
            tmp.unlink(missing_ok=True)
    return target


# -- coordinate alignment ---------------------------------------------------


def _page_transform(pm_page: fitz.Page, pl_page: pdfplumber.page.Page) -> Callable[..., fitz.Rect]:
    """Map pdfplumber (MediaBox/rotated) boxes into pymupdf text space.

    pymupdf reports text coordinates relative to the unrotated CropBox, while
    pdfplumber reports table boxes in MediaBox space (rotated when the page is).
    The transform scales to MediaBox units, derotates, then subtracts the
    CropBox origin.
    """
    media = pm_page.mediabox
    crop = pm_page.cropbox
    rotation = pm_page.rotation % 360
    pl_width = float(pl_page.bbox[2] - pl_page.bbox[0])
    pl_height = float(pl_page.bbox[3] - pl_page.bbox[1])
    expected_width = media.width if rotation in (0, 180) else media.height
    expected_height = media.height if rotation in (0, 180) else media.width
    scale_x = pl_width / expected_width if expected_width else 1.0
    scale_y = pl_height / expected_height if expected_height else 1.0

    def transform(bbox: Sequence[float]) -> fitz.Rect:
        x0, y0, x1, y1 = (float(value) for value in bbox)
        x0, x1 = x0 * scale_x, x1 * scale_x
        y0, y1 = y0 * scale_y, y1 * scale_y
        if rotation == 90:
            x0, y0, x1, y1 = y0, media.height - x1, y1, media.height - x0
        elif rotation == 180:
            x0, y0, x1, y1 = (
                media.width - x1,
                media.height - y1,
                media.width - x0,
                media.height - y0,
            )
        elif rotation == 270:
            x0, y0, x1, y1 = media.width - y1, x0, media.width - y0, x1
        rect = fitz.Rect(
            x0 - crop.x0,
            y0 - crop.y0,
            x1 - crop.x0,
            y1 - crop.y0,
        )
        return rect.normalize()

    return transform


# -- tables -----------------------------------------------------------------


def _extract_tables(
    pm_page: fitz.Page,
    pl_page: pdfplumber.page.Page,
    transform: Callable[..., fitz.Rect],
) -> list[_TableBlock]:
    page_area = pm_page.cropbox.width * pm_page.cropbox.height
    try:
        found = pl_page.find_tables(table_settings=TABLE_SETTINGS)
    except Exception:  # pragma: no cover - defensive against pdfplumber bugs
        return []

    blocks: list[_TableBlock] = []
    for table in found:
        rows = table.extract()
        if not rows:
            continue
        nrows = len(rows)
        ncols = max((len(row) for row in rows), default=0)
        if nrows < 2 and ncols < 2:
            # A 1x1 callout/warning box is prose, not a table.
            continue
        non_empty = [cell for row in rows for cell in row if cell and str(cell).strip()]
        if not non_empty:
            continue
        rect = transform(table.bbox)
        if (
            page_area
            and (rect.width * rect.height) / page_area > PAGE_BORDER_AREA_RATIO
            and len(non_empty) <= PAGE_BORDER_MAX_CELLS
        ):
            # An outer page-border rectangle with no real content.
            continue
        blocks.append(_TableBlock(rect=rect, markdown=_to_markdown(rows)))
    return blocks


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("|", "\\|")
    return re.sub(r"\s+", " ", text).strip()


def _to_markdown(rows: Sequence[Sequence[str | None]]) -> str:
    cleaned = [[_clean_cell(cell) for cell in row] for row in rows]
    width = max((len(row) for row in cleaned), default=0)
    for row in cleaned:
        row.extend([""] * (width - len(row)))
    header = cleaned[0]
    body = cleaned[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


# -- text lines -------------------------------------------------------------


def _extract_lines(pm_page: fitz.Page, table_rects: Sequence[fitz.Rect]) -> list[_LayoutItem]:
    data = pm_page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)
    items: list[_LayoutItem] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = _normalise_text(
                "".join(span.get("text", "") for span in line.get("spans", []))
            )
            if not text:
                continue
            rect = fitz.Rect(line["bbox"])
            if _overlaps_table(rect, table_rects):
                continue
            items.append(_LayoutItem(rect.x0, rect.y0, rect.x1, rect.y1, text, "text"))
    return items


def _normalise_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _overlaps_table(rect: fitz.Rect, table_rects: Sequence[fitz.Rect]) -> bool:
    if rect.is_empty:
        return False
    centre = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    line_area = rect.width * rect.height
    for table in table_rects:
        if centre in table:
            return True
        intersection = rect & table
        if intersection.is_empty:
            continue
        if line_area and (intersection.width * intersection.height) / line_area > 0.5:
            return True
    return False


# -- repeating headers / footers -------------------------------------------


def _zone_of(item: _LayoutItem, page: _Page) -> str | None:
    if item.y1 <= page.height * MARGIN_ZONE_RATIO:
        return "top"
    if item.y0 >= page.height * (1.0 - MARGIN_ZONE_RATIO):
        return "bottom"
    return None


def _signature(text: str) -> str | None:
    normalised = re.sub(r"\b\d+(?:\.\d+)*\b", "#", text)
    normalised = re.sub(r"\s+", " ", normalised).strip()
    if not normalised or not re.search(r"\w", normalised):
        return None
    return normalised


def _repeating_signatures(pages: Sequence[_Page]) -> dict[str, set[str]]:
    seen: dict[str, dict[str, set[int]]] = {"top": {}, "bottom": {}}
    for page in pages:
        for item in page.lines:
            zone = _zone_of(item, page)
            if zone is None:
                continue
            signature = _signature(item.text)
            if signature is None:
                continue
            seen[zone].setdefault(signature, set()).add(page.page_num)

    page_count = len(pages)
    repeating: dict[str, set[str]] = {}
    for zone, signatures in seen.items():
        repeating[zone] = {
            signature
            for signature, seen_on in signatures.items()
            if len(seen_on) >= HEADER_FOOTER_MIN_PAGES
            and len(seen_on) / page_count >= HEADER_FOOTER_MIN_RATIO
        }
    return repeating


def _filter_repeating(
    items: Sequence[_LayoutItem],
    page: _Page,
    repeating: dict[str, set[str]],
) -> list[_LayoutItem]:
    kept: list[_LayoutItem] = []
    for item in items:
        zone = _zone_of(item, page)
        if zone is not None:
            signature = _signature(item.text)
            if signature is not None and signature in repeating.get(zone, set()):
                continue
        kept.append(item)
    return kept


# -- layout / reading order -------------------------------------------------


def _y_key(item: _LayoutItem) -> tuple[float, float]:
    return (item.y0, item.x0)


def _detect_gutter(items: Sequence[_LayoutItem], page_width: float) -> float | None:
    if len(items) < 4:
        return None
    best: float | None = None
    best_clearance = 0.0
    for percent in range(30, 71):
        gutter = page_width * percent / 100.0
        left = [item for item in items if item.x1 <= gutter]
        right = [item for item in items if item.x0 >= gutter]
        if len(left) < 2 or len(right) < 2:
            continue
        if not any(
            left_item.y0 < right_item.y1 and right_item.y0 < left_item.y1
            for left_item in left
            for right_item in right
        ):
            continue
        clearance = min(item.x0 for item in right) - max(item.x1 for item in left)
        if clearance <= 0:
            continue
        if clearance > best_clearance:
            best_clearance = clearance
            best = gutter
    return best


def _is_full_width(item: _LayoutItem, gutter: float, page_width: float) -> bool:
    if item.x0 < gutter < item.x1:
        return True
    return (item.x1 - item.x0) > page_width * FULL_WIDTH_RATIO


def _order_band(items: Sequence[_LayoutItem], gutter: float) -> list[_LayoutItem]:
    left = sorted((item for item in items if item.x1 <= gutter), key=_y_key)
    right = sorted((item for item in items if item.x0 >= gutter), key=_y_key)
    straddling = sorted(
        (item for item in items if item.x0 < gutter < item.x1), key=_y_key
    )
    return [*left, *straddling, *right]


def _order_items(items: Sequence[_LayoutItem], page_width: float) -> list[_LayoutItem]:
    if not items:
        return []
    gutter = _detect_gutter(items, page_width)
    if gutter is None:
        return sorted(items, key=_y_key)

    full_width = sorted(
        (item for item in items if _is_full_width(item, gutter, page_width)),
        key=_y_key,
    )
    if not full_width:
        return _order_band(items, gutter)

    ordered: list[_LayoutItem] = []
    previous = float("-inf")
    for boundary in full_width:
        midpoint = (boundary.y0 + boundary.y1) / 2
        band = [
            item
            for item in items
            if not _is_full_width(item, gutter, page_width)
            and previous <= (item.y0 + item.y1) / 2 < midpoint
        ]
        ordered.extend(_order_band(band, gutter))
        ordered.append(boundary)
        previous = midpoint

    tail = [
        item
        for item in items
        if not _is_full_width(item, gutter, page_width)
        and (item.y0 + item.y1) / 2 >= previous
    ]
    ordered.extend(_order_band(tail, gutter))
    return ordered


# -- assembly ---------------------------------------------------------------


def _assemble_pages(
    page_texts: Sequence[str], table_counts: Sequence[int]
) -> tuple[str, list[CanonicalPage]]:
    full_text = "\n\n".join(page_texts)
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
                table_count=table_counts[index],
            )
        )
        cursor = end
    return full_text, pages


__all__ = [
    "CanonicalParseError",
    "TokenCounter",
    "parse_pdf",
    "write_canonical_document",
]
