"""Records shared by experiment construction, retrieval, and reporting."""
from dataclasses import dataclass, field
from typing import TypedDict
from uuid import uuid4


ImageMetadata = TypedDict(
    "ImageMetadata",
    {"fname": str, "class": str, "source_id": str, "source_split": str},
)


class Neighbor(TypedDict):
    id: str
    label: str
    score: float


class Prediction(ImageMetadata):
    prediction: str | None
    vote_fraction: float
    nearest: str | None
    neighbors: list[Neighbor]


@dataclass(frozen=True)
class PointRecord:
    label: str
    source_kind: str
    source_identity: str
    value: str
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_payload(self) -> dict:
        """Keep the stored class key, which is a reserved word in Python."""
        return {
            'id': self.id,
            'class': self.label,
            'source_kind': self.source_kind,
            'source_identity': self.source_identity,
            'value': self.value,
        }
