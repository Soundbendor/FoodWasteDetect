"""Typed dictionaries for persisted index summaries and evaluation reports."""

from typing import Any, TypedDict

from classification.retrieval.metrics import RetrievalMetrics
from classification.schemas import Prediction


class IndexSummary(TypedDict):
    """Complete reference-build summary used for both writing and reading."""

    table: str
    uri: str
    source_counts: dict[str, int]
    config: dict[str, Any]


class EvaluationReport(TypedDict):
    """Completed predictions and metrics written before tracker publication."""

    config: dict[str, Any]
    table: str
    reference_source_counts: dict[str, int] | None
    predictions: list[Prediction]
    metrics: RetrievalMetrics
