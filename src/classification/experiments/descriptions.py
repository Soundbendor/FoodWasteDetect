"""Read saved vocabularies and construct ordered text references."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from classification.experiments.config import ClassLabelPrompt
from classification.schemas import PointRecord


def load_descriptions(path: Path) -> dict[str, list[str]]:
    """Read the vocabulary at the experiment's resolved path without modifying it."""
    return json.loads(path.read_text(encoding="utf-8"))["descriptions"]


def text_references(
    classes: Sequence[str],
    *,
    include_class_labels: bool,
    descriptions: Mapping[str, Sequence[str]] | None,
    class_label_prompt: ClassLabelPrompt = "bare",
) -> list[PointRecord]:
    """Keep class names first, followed by descriptions in class and sentence order."""
    records = []
    if include_class_labels:
        for label in classes:
            text = label
            if class_label_prompt == "food_photo":
                text = f"a photo of {label.lower()}, a type of food."
            elif class_label_prompt == "siglip2_photo":
                text = f"This is a photo of {label}"
            records.append(PointRecord(label, "class_name", label, text))
    if descriptions is not None:
        records.extend(
            PointRecord(label, "generated_description", f"{label}:{index}", text)
            for label in classes
            for index, text in enumerate(descriptions[label])
        )
    return records
