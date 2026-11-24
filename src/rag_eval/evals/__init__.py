"""Golden QA dataset: span-anchored schema, validator, and split helpers."""

from __future__ import annotations

from rag_eval.evals.dataset import (
    CATEGORY_VALUES,
    DIFFICULTY_VALUES,
    CitationAnchor,
    GoldenQAItem,
)
from rag_eval.evals.dataset_validator import (
    CitationError,
    DatasetError,
    DatasetProblem,
    DatasetStats,
    ValidateResult,
    build_smoke_split,
    coverage_problems,
    dataset_statistics,
    load_documents,
    load_golden_qa,
    resolve_citation,
    validate_dataset,
    write_resolved,
    write_smoke,
)

__all__ = [
    "CATEGORY_VALUES",
    "DIFFICULTY_VALUES",
    "CitationAnchor",
    "CitationError",
    "DatasetError",
    "DatasetProblem",
    "DatasetStats",
    "GoldenQAItem",
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
