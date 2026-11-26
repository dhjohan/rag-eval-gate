"""Span-anchored golden QA schema.

A golden item never references a chunk. It points at
``(doc_id, page_num, evidence_quote)``; the validator resolves that quote to
exact ``[start_char, end_char]`` offsets in the canonical ``full_text``. This is
the citation-invariance contract from ``AGENTS.md``.

Whitespace is stripped declaratively via ``str_strip_whitespace`` so the models
stay frozen and pure: empty or whitespace-only required fields fail
``min_length=1`` instead of being mutated after validation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal[
    "single_fact",
    "tabular_numeric",
    "multi_hop",
    "conditional_policy",
    "unanswerable",
]
Difficulty = Literal["easy", "medium", "hard"]

CATEGORY_VALUES: tuple[str, ...] = (
    "single_fact",
    "tabular_numeric",
    "multi_hop",
    "conditional_policy",
    "unanswerable",
)
DIFFICULTY_VALUES: tuple[str, ...] = ("easy", "medium", "hard")

_BASE_CONFIG = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class CitationAnchor(BaseModel):
    """A page-scoped evidence quote, optionally resolved to exact offsets."""

    model_config = _BASE_CONFIG

    doc_id: str = Field(min_length=1)
    page_num: int = Field(ge=1)
    evidence_quote: str = Field(min_length=1)
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=1)
    doc_text_sha256: str | None = None

    @model_validator(mode="after")
    def _offsets_are_ordered(self) -> CitationAnchor:
        if (
            self.start_char is not None
            and self.end_char is not None
            and self.end_char < self.start_char
        ):
            raise ValueError("end_char must be >= start_char")
        if (self.start_char is None) != (self.end_char is None):
            raise ValueError("start_char and end_char must be set together")
        return self


class GoldenQAItem(BaseModel):
    """One human-authored evaluation question with span-anchored citations."""

    model_config = _BASE_CONFIG

    qa_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    category: Category
    unanswerable: bool
    difficulty: Difficulty
    citations: list[CitationAnchor]
    notes: str = ""

    @model_validator(mode="after")
    def _category_matches_flags(self) -> GoldenQAItem:
        if (self.category == "unanswerable") != self.unanswerable:
            raise ValueError(
                "category 'unanswerable' and the unanswerable flag must agree"
            )
        if self.unanswerable:
            if self.citations:
                raise ValueError("an unanswerable item must not have citations")
        elif not self.citations:
            raise ValueError("an answerable item needs at least one citation")
        return self


__all__ = [
    "CATEGORY_VALUES",
    "DIFFICULTY_VALUES",
    "Category",
    "CitationAnchor",
    "Difficulty",
    "GoldenQAItem",
]
