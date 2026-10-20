"""Point records shared by experiment construction and vector storage."""
from dataclasses import dataclass, field
from uuid import uuid4


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
