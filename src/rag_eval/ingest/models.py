"""Canonical parsed-document models with strict offset invariants.

These models are the citation-invariance contract: a golden QA item points at
``(doc_id, page_num, evidence_quote)`` and resolves it through
``full_text[page.start_char:page.end_char]`` on the canonical document. The
validator therefore refuses any document whose page slices do not reproduce the
recorded page text exactly.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DOC_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class CanonicalPage(BaseModel):
    """One page of a canonical document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_num: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str
    table_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> CanonicalPage:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        return self


class CanonicalDocument(BaseModel):
    """A fully parsed PDF with exact page-to-character offsets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1)
    source_url: str | None = None
    effective_date: str | None = None
    sha256: str
    page_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    full_text: str
    pages: list[CanonicalPage]

    @field_validator("doc_id")
    @classmethod
    def _valid_doc_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not DOC_ID_PATTERN.match(cleaned):
            raise ValueError("doc_id must be a lowercase slug such as 'fda-ai-ml-action-plan'")
        return cleaned

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not SHA256_PATTERN.match(cleaned):
            raise ValueError("sha256 must be 64 hexadecimal characters")
        return cleaned

    @model_validator(mode="after")
    def _invariants(self) -> CanonicalDocument:
        if self.page_count != len(self.pages):
            raise ValueError(f"page_count {self.page_count} != len(pages) {len(self.pages)}")

        expected_numbers = list(range(1, len(self.pages) + 1))
        actual_numbers = [page.page_num for page in self.pages]
        if actual_numbers != expected_numbers:
            raise ValueError(f"pages must be numbered contiguously from 1, got {actual_numbers}")

        total_tables = sum(page.table_count for page in self.pages)
        if self.table_count != total_tables:
            raise ValueError(
                f"table_count {self.table_count} != sum(page.table_count) {total_tables}"
            )

        for page in self.pages:
            if not 0 <= page.start_char <= page.end_char <= len(self.full_text):
                raise ValueError(
                    f"page {page.page_num} offsets [{page.start_char}:{page.end_char}] "
                    f"are outside full_text of length {len(self.full_text)}"
                )
            if self.full_text[page.start_char : page.end_char] != page.text:
                raise ValueError(
                    f"page {page.page_num} text does not match "
                    f"full_text[{page.start_char}:{page.end_char}]"
                )
        return self


__all__ = ["CanonicalDocument", "CanonicalPage"]
