"""Offset-invariant chunkers: C1 fixed token, C2 structure aware, C3 contextual.

Every chunk satisfies the hard invariant::

    document.full_text[chunk.start_char : chunk.end_char] == chunk.raw_text

C1 walks the tokenizer's own character spans, so offsets come straight from the
token stream and are never reconstructed by decoding tokens. C2 and C3 share
one structural pass (so their spans and ``raw_text`` are identical); C3 only
changes ``index_text`` by prepending a document/section context prefix.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from rag_eval.ingest.models import CanonicalDocument
from rag_eval.pipeline.models import Chunk, ChunkConfig
from rag_eval.pipeline.tokenizer import Tokenizer, default_tokenizer

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_NUM_HEADING = re.compile(
    r"^((?:[IVXLCM]+|\d+)(?:\.(?:[A-Z0-9]+|\d+))*\.?)\s+(\S.*)$"
)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


@dataclass(frozen=True)
class _Block:
    kind: str  # "heading" | "paragraph" | "table"
    start: int
    end: int
    text: str
    section_path: str
    level: int = 0
    title: str = ""


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    section_path: str
    table_header: str = ""


def _iter_lines(text: str) -> Iterator[tuple[int, int, str]]:
    position = 0
    for raw in text.split("\n"):
        start = position
        end = position + len(raw)
        yield start, end, raw
        position = end + 1


def _detect_heading(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    markdown = _MD_HEADING.match(stripped)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()
    numbered = _NUM_HEADING.match(stripped)
    if numbered and "." in numbered.group(1) and len(stripped) <= 100:
        title = numbered.group(2)
        # Require a capitalised title word to avoid treating sentences such as
        # "3.5 million people were affected" as headings.
        if title[:1].isupper():
            depth = numbered.group(1).rstrip(".").count(".") + 1
            return depth, stripped
    return None


def _path(stack: Sequence[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack) or "Preamble"


def _structure_blocks(document: CanonicalDocument) -> list[_Block]:
    text = document.full_text
    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    lines = list(_iter_lines(text))
    index = 0
    while index < len(lines):
        start, end, line = lines[index]
        if not line.strip():
            index += 1
            continue
        heading = _detect_heading(line)
        if heading is not None:
            level, title = heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            blocks.append(
                _Block("heading", start, end, text[start:end], _path(stack), level, title)
            )
            index += 1
            continue
        if _TABLE_LINE.match(line):
            last = index
            while last + 1 < len(lines) and _TABLE_LINE.match(lines[last + 1][2]):
                last += 1
            table_start = lines[index][0]
            table_end = lines[last][1]
            blocks.append(
                _Block("table", table_start, table_end, text[table_start:table_end], _path(stack))
            )
            index = last + 1
            continue
        last = index
        while last + 1 < len(lines):
            _, _, next_line = lines[last + 1]
            if (
                not next_line.strip()
                or _detect_heading(next_line)
                or _TABLE_LINE.match(next_line)
            ):
                break
            last += 1
        paragraph_start = lines[index][0]
        paragraph_end = lines[last][1]
        blocks.append(
            _Block(
                "paragraph",
                paragraph_start,
                paragraph_end,
                text[paragraph_start:paragraph_end],
                _path(stack),
            )
        )
        index = last + 1
    return blocks


def _token_windows(
    spans: Sequence[tuple[int, int]], max_tokens: int, overlap_tokens: int
) -> Iterator[tuple[int, int]]:
    count = len(spans)
    if count == 0:
        return
    stride = max(1, max_tokens - overlap_tokens)
    index = 0
    while index < count:
        last = min(index + max_tokens, count)
        yield spans[index][0], spans[last - 1][1]
        if last >= count:
            break
        index += stride


def _page_span(document: CanonicalDocument, start: int, end: int) -> tuple[int, int]:
    first: int | None = None
    last: int | None = None
    for page in document.pages:
        if page.start_char < end and page.end_char > start:
            if first is None:
                first = page.page_num
            last = page.page_num
    if first is not None and last is not None:
        return first, last
    # Chunks never normally fall in the "\n\n" gap between pages, but stay
    # defensive for zero-length spans.
    for page in document.pages:
        if page.end_char > start:
            return page.page_num, page.page_num
    final = document.pages[-1].page_num
    return final, final


def _table_parts(
    block: _Block,
    document: CanonicalDocument,
    config: ChunkConfig,
    tokenizer: Tokenizer,
    *,
    cap: int | None = None,
) -> list[tuple[int, int, str]]:
    """Split an oversized table on row boundaries.

    Part 0 keeps the header in ``raw_text``. Continuation parts repeat the
    header only in ``index_text`` (returned as the third tuple field).
    ``cap`` defaults to ``config.hard_max_table_tokens`` and bounds the
    header-plus-rows text that ends up in ``index_text``.
    """
    cap = cap if cap is not None else config.hard_max_table_tokens
    relative = list(_iter_lines(block.text))
    if len(relative) < 2:
        return [(block.start, block.end, "")]

    header_count = 1
    if (
        len(relative) >= 2
        and _TABLE_SEPARATOR.match(relative[1][2])
        and "-" in relative[1][2]
    ):
        header_count = 2
    if len(relative) <= header_count:
        return [(block.start, block.end, "")]

    header_lines = relative[:header_count]
    data_lines = relative[header_count:]
    header_text = document.full_text[
        block.start + header_lines[0][0] : block.start + header_lines[-1][1]
    ]
    header_tokens = tokenizer.count(header_text)

    spans: list[tuple[int, int, str]] = []
    header_covered = False
    current: list[tuple[int, int, str]] = []  # (relative start, relative end, row)

    def flush() -> None:
        nonlocal current, header_covered
        if not current:
            return
        end = block.start + current[-1][1]
        if header_covered:
            spans.append((block.start + current[0][0], end, header_text))
        else:
            # Part 0 keeps the header in ``raw_text``.
            spans.append((block.start + header_lines[0][0], end, ""))
            header_covered = True
        current = []

    for rel_start, rel_end, row_text in data_lines:
        if current:
            trial_text = header_text + "\n" + "\n".join(
                [row[2] for row in current] + [row_text]
            )
            if tokenizer.count(trial_text) > cap:
                flush()
        if not current and tokenizer.count(header_text + "\n" + row_text) > cap:
            # The first row of a part is trial-checked against header + row,
            # and a row that cannot fit even then is split at token boundaries
            # as a last resort. The first piece also carries the header lines
            # in ``raw_text`` when possible.
            budget = max(1, cap - header_tokens - 1)
            windows = list(
                _token_windows(tokenizer.token_char_spans(row_text), budget, 0)
            )
            for window_index, (win_start, win_end) in enumerate(windows):
                start = block.start + rel_start + win_start
                end = block.start + rel_start + win_end
                if window_index == 0 and not header_covered:
                    spans.append((block.start + header_lines[0][0], end, ""))
                    header_covered = True
                else:
                    spans.append((start, end, header_text))
            continue
        current.append((rel_start, rel_end, row_text))
    flush()
    spans.sort(key=lambda span: span[0])
    return spans


def _structure_spans(
    document: CanonicalDocument,
    config: ChunkConfig,
    tokenizer: Tokenizer,
    *,
    token_budget: int | None = None,
    table_cap: int | None = None,
    blocks: list[_Block] | None = None,
) -> list[_Span]:
    """Pack structural blocks into spans within their respective caps.

    Prose is packed to ``token_budget`` (default ``max_tokens``); tables are
    split at ``table_cap`` (default ``hard_max_table_tokens``). Token counts
    are always taken from the exact candidate slice of ``full_text`` so
    intervening newlines are budgeted too; summing per-block counts let
    packed chunks exceed the cap by the separator characters.
    """
    budget = token_budget if token_budget is not None else config.max_tokens
    table_cap = table_cap if table_cap is not None else config.hard_max_table_tokens
    spans: list[_Span] = []
    accumulator: list[_Block] = []

    def flush() -> None:
        nonlocal accumulator
        if not accumulator:
            return
        spans.append(
            _Span(accumulator[0].start, accumulator[-1].end, accumulator[0].section_path)
        )
        accumulator = []

    for block in blocks if blocks is not None else _structure_blocks(document):
        if block.kind == "heading":
            flush()
            accumulator = [block]
        elif block.kind == "table":
            flush()
            if tokenizer.count(block.text) <= table_cap:
                spans.append(_Span(block.start, block.end, block.section_path))
            else:
                for start, end, header in _table_parts(
                    block, document, config, tokenizer, cap=table_cap
                ):
                    spans.append(_Span(start, end, block.section_path, header))
        else:
            paragraph_tokens = tokenizer.count(block.text)
            if paragraph_tokens > budget:
                flush()
                relative = tokenizer.token_char_spans(block.text)
                for start, end in _token_windows(relative, budget, config.overlap_tokens):
                    spans.append(
                        _Span(block.start + start, block.start + end, block.section_path)
                    )
            else:
                if accumulator:
                    trial = tokenizer.count(
                        document.full_text[accumulator[0].start : block.end]
                    )
                    if trial > budget:
                        flush()
                accumulator.append(block)
    flush()
    return spans


def _context_prefix(document: CanonicalDocument, section_path: str) -> str:
    title = document.title.strip() or document.doc_id
    section = section_path or "Preamble"
    prefix = f"Document: {title} | Section: {section}"
    if document.effective_date:
        prefix += f" | Effective: {document.effective_date}"
    return f"[{prefix}]"


def _emit(
    document: CanonicalDocument,
    config: ChunkConfig,
    tokenizer: Tokenizer,
    strategy: str,
    spans: Iterable[_Span],
    *,
    contextual: bool,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, span in enumerate(spans):
        raw_text = document.full_text[span.start : span.end]
        page_start, page_end = _page_span(document, span.start, span.end)
        header = f"{span.table_header}\n" if span.table_header else ""
        if contextual:
            index_text = (
                _context_prefix(document, span.section_path) + "\n" + header + raw_text
            )
        else:
            index_text = header + raw_text
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{index:04d}",
                doc_id=document.doc_id,
                chunk_strategy=strategy,  # type: ignore[arg-type]
                chunk_index=index,
                start_char=span.start,
                end_char=span.end,
                page_start=page_start,
                page_end=page_end,
                section_path=span.section_path,
                raw_text=raw_text,
                index_text=index_text,
                token_count=tokenizer.count(raw_text),
            )
        )
    return chunks


def chunk_fixed_token(
    document: CanonicalDocument, config: ChunkConfig, tokenizer: Tokenizer
) -> list[Chunk]:
    """C1: fixed-size token windows over the whole document."""
    spans = tokenizer.token_char_spans(document.full_text)
    windows = [
        _Span(start, end, "")
        for start, end in _token_windows(spans, config.max_tokens, config.overlap_tokens)
    ]
    return _emit(document, config, tokenizer, "fixed_token", windows, contextual=False)


def chunk_structure_aware(
    document: CanonicalDocument, config: ChunkConfig, tokenizer: Tokenizer
) -> list[Chunk]:
    """C2: pack headings, paragraphs, and tables by structural boundaries."""
    return _emit(
        document,
        config,
        tokenizer,
        "structure_aware",
        _structure_spans(document, config, tokenizer),
        contextual=False,
    )


def chunk_contextual_heading(
    document: CanonicalDocument, config: ChunkConfig, tokenizer: Tokenizer
) -> list[Chunk]:
    """C3: C2 spans plus a ``[Document | Section]`` context prefix.

    The prefix is part of ``index_text``, so its tokens are subtracted from
    both budgets: prose ``index_text`` stays within ``max_tokens`` and table
    ``index_text`` stays within ``hard_max_table_tokens``.
    """
    blocks = _structure_blocks(document)
    prefix_tokens = max(
        (tokenizer.count(_context_prefix(document, block.section_path)) for block in blocks),
        default=0,
    )
    budget = max(1, config.max_tokens - prefix_tokens - 1)
    table_cap = max(1, config.hard_max_table_tokens - prefix_tokens - 1)
    return _emit(
        document,
        config,
        tokenizer,
        "contextual_heading",
        _structure_spans(
            document,
            config,
            tokenizer,
            token_budget=budget,
            table_cap=table_cap,
            blocks=blocks,
        ),
        contextual=True,
    )


def chunk_document(
    document: CanonicalDocument,
    config: ChunkConfig,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Dispatch to the configured strategy."""
    resolved = tokenizer or default_tokenizer()
    if config.strategy == "fixed_token":
        return chunk_fixed_token(document, config, resolved)
    if config.strategy == "structure_aware":
        return chunk_structure_aware(document, config, resolved)
    return chunk_contextual_heading(document, config, resolved)


def chunk_documents(
    documents: Iterable[CanonicalDocument],
    config: ChunkConfig,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Chunk every document, sorted by ``doc_id`` for deterministic ids."""
    resolved = tokenizer or default_tokenizer()
    chunks: list[Chunk] = []
    for document in sorted(documents, key=lambda item: item.doc_id):
        chunks.extend(chunk_document(document, config, resolved))
    return chunks


__all__ = [
    "chunk_contextual_heading",
    "chunk_document",
    "chunk_documents",
    "chunk_fixed_token",
    "chunk_structure_aware",
]
