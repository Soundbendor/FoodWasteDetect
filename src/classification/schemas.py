"""Records shared by experiment construction, retrieval, and reporting."""
from dataclasses import dataclass, field
from typing import TypedDict
from uuid import uuid4

type SourceIdentity = tuple[str, str]

ImageMetadata = TypedDict(
    "ImageMetadata",
    {"fname": str, "class": str, "source_id": str, "source_split": str},
)


class Neighbor(TypedDict):
    """Retrieved record and cosine similarity, which may be negative."""

    id: str
    label: str
    score: float


class QueryRetrievalMetrics(TypedDict):
    """Ranking scores and relevant-reference count for one validation image."""

    top5_hit: bool
    precision_at_k: float
    reciprocal_rank_at_k: float
    ndcg_at_k: float
    average_precision_at_r: float
    relevant_reference_count: int


class Prediction(ImageMetadata):
    """Image metadata, vote result, voting neighbors, and ranking metrics."""

    prediction: str | None
    vote_fraction: float
    nearest: str | None
    neighbors: list[Neighbor]
    retrieval_metrics: QueryRetrievalMetrics


@dataclass(frozen=True)
class PointRecord:
    """Reference or cache row with a stable source identity and generated record ID."""

    label: str
    source_kind: str
    source_identity: str
    value: str
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_payload(self) -> dict[str, str]:
        """Return storage fields, mapping ``label`` to the reserved key ``class``."""
        return {
            'id': self.id,
            'class': self.label,
            'source_kind': self.source_kind,
            'source_identity': self.source_identity,
            'value': self.value,
        }
