"""Pipeline data models for chunking and hybrid indexing."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Strategy = Literal["fixed_token", "structure_aware", "contextual_heading"]
STRATEGY_VALUES: tuple[str, ...] = (
    "fixed_token",
    "structure_aware",
    "contextual_heading",
)

# Bump when chunker/tokenizer/index-writing behavior changes so a changed
# implementation can never reuse an index built by the old one.
PIPELINE_VERSION = "2"


class ChunkConfig(BaseModel):
    """How a document should be split into chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Strategy
    max_tokens: int = Field(default=450, ge=1)
    overlap_tokens: int = Field(default=64, ge=0)
    hard_max_table_tokens: int = Field(default=900, ge=1)

    @model_validator(mode="after")
    def _overlap_is_smaller_than_window(self) -> ChunkConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class Chunk(BaseModel):
    """One offset-exact chunk of a canonical document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    chunk_strategy: Strategy
    chunk_index: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: str
    raw_text: str
    index_text: str
    token_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _spans_are_coherent(self) -> Chunk:
        if self.end_char < self.start_char:
            raise ValueError("end_char must be >= start_char")
        if self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        if len(self.raw_text) != self.end_char - self.start_char:
            raise ValueError("raw_text length must equal end_char - start_char")
        return self


class IndexManifest(BaseModel):
    """Metadata describing one built hybrid index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    index_id: str = Field(min_length=1)
    chunk_config: ChunkConfig
    embedding_provider: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    dims: int = Field(ge=1)
    corpus_sha256: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    created_at: str = Field(min_length=1)
    # Defaults keep pre-version manifests readable; they can never match a
    # freshly computed index_id, so they are rebuilt rather than trusted.
    pipeline_version: str = "1"
    embedding_task_type: str = ""


__all__ = [
    "PIPELINE_VERSION",
    "STRATEGY_VALUES",
    "Chunk",
    "ChunkConfig",
    "IndexManifest",
    "Strategy",
]
