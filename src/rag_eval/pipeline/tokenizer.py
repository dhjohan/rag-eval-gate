"""Injectable tokenizers for offset-safe chunking.

Production uses tiktoken's ``cl100k_base`` encoding; tests use a deterministic
whitespace tokenizer so the suite stays offline and never touches tiktoken's
blob cache.

The tokenizer maps every token to a *character* span in the input string. A
token may begin or end in the middle of a multi-byte UTF-8 codepoint, so spans
are snapped outward to whole codepoints. That keeps
``text[start:end]`` from ever splitting a character.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from typing import Protocol

_WHITESPACE_TOKEN = re.compile(r"\S+")


class Tokenizer(Protocol):
    """Minimal tokenizer surface the chunkers depend on."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        ...

    def token_char_spans(self, text: str) -> list[tuple[int, int]]:
        """Return one ``(start_char, end_char)`` span per token."""
        ...


class TiktokenTokenizer:
    """Lazy ``cl100k_base`` tokenizer with byte-to-character offset mapping.

    ``encoding`` may be injected in tests with a fake object exposing
    ``encode_ordinary`` and ``decode_tokens_bytes``; production leaves it
    ``None`` so tiktoken is imported lazily on first use.
    """

    def __init__(
        self, encoding_name: str = "cl100k_base", *, encoding: object | None = None
    ) -> None:
        self._encoding_name = encoding_name
        self._encoding: object | None = encoding

    def _get_encoding(self) -> object:
        if self._encoding is None:
            import tiktoken

            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding

    def count(self, text: str) -> int:
        if not text:
            return 0
        encoding = self._get_encoding()
        return len(encoding.encode_ordinary(text))  # type: ignore[attr-defined]

    def token_char_spans(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        encoding = self._get_encoding()
        tokens = encoding.encode_ordinary(text)  # type: ignore[attr-defined]
        if not tokens:
            return []

        data = text.encode("utf-8")
        # Character boundaries are byte offsets that start a codepoint, i.e.
        # every byte that is not a UTF-8 continuation byte (10xxxxxx).
        boundaries = [0]
        boundaries.extend(
            index
            for index, byte in enumerate(data)
            if index and (byte & 0xC0) != 0x80
        )
        boundaries.append(len(data))

        token_bytes = encoding.decode_tokens_bytes(tokens)  # type: ignore[attr-defined]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for chunk in token_bytes:
            end = cursor + len(chunk)
            # Floor the start and ceil the end so a split codepoint is included
            # whole rather than cut in half.
            start_char = bisect_right(boundaries, cursor) - 1
            end_char = bisect_left(boundaries, end)
            spans.append((start_char, end_char))
            cursor = end
        if cursor != len(data):  # pragma: no cover - defensive
            raise ValueError("token bytes do not reconstruct the input text")
        return spans


class OfflineTokenizer:
    """Deterministic whitespace tokenizer for offline tests and fixtures."""

    def count(self, text: str) -> int:
        return len(_WHITESPACE_TOKEN.findall(text))

    def token_char_spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in _WHITESPACE_TOKEN.finditer(text)]


def default_tokenizer() -> Tokenizer:
    return TiktokenTokenizer()


__all__ = [
    "OfflineTokenizer",
    "TiktokenTokenizer",
    "Tokenizer",
    "default_tokenizer",
]
