"""Read saved vocabularies and construct ordered text references."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..schemas import PointRecord


def load_descriptions(path: Path) -> dict[str, list[str]]:
    """Read the vocabulary at the experiment's resolved path without modifying it."""
    return json.loads(path.read_text(encoding="utf-8"))["descriptions"]


def text_references(
    classes: Sequence[str],
    *,
    include_class_labels: bool,
    descriptions: Mapping[str, Sequence[str]] | None,
) -> list[PointRecord]:
    """Keep class names first, followed by descriptions in class and sentence order."""
    records = (
        [PointRecord(label, "class_name", label, label) for label in classes]
        if include_class_labels else []
    )
    if descriptions is not None:
        records.extend(
            PointRecord(label, "generated_description", f"{label}:{index}", text)
            for label in classes
            for index, text in enumerate(descriptions[label])
        )
    return records
