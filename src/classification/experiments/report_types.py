"""Validated models for persisted index summaries and evaluation reports."""

from pydantic import BaseModel

from classification.experiments.config import ExperimentConfig
from classification.retrieval.metrics import RetrievalMetrics
from classification.schemas import Prediction


class IndexSummary(BaseModel):
    """Completed reference-build summary with effective typed settings."""

    table: str
    uri: str
    source_counts: dict[str, int]
    config: ExperimentConfig


class EvaluationReport(BaseModel):
    """Completed predictions and metrics written before tracker publication.

    Attributes:
        reference_source_counts: Stored reference-table rows per source kind, read
            from the table itself; never null.
    """

    config: ExperimentConfig
    table: str
    reference_source_counts: dict[str, int]
    predictions: list[Prediction]
    metrics: RetrievalMetrics
