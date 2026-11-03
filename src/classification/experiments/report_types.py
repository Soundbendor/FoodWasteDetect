"""Typed dictionaries for persisted index summaries and evaluation reports."""

from typing import Any, TypedDict

from classification.retrieval.metrics import RetrievalMetrics
from classification.schemas import Prediction


class IndexSummary(TypedDict):
    """Summary published after a completed reference build."""

    table: str
    uri: str
    source_counts: dict[str, int]
    config: dict[str, Any]


class LoadedIndexSummary(TypedDict, total=False):
    """Optional fields read from an existing, unvalidated index summary."""

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
