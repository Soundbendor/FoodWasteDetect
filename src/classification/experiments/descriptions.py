"""Read saved vocabularies and construct ordered text references."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from classification.experiments.config import ClassLabelPrompt
from classification.schemas import PointRecord


def load_descriptions(path: Path) -> dict[str, list[str]]:
    """Read the vocabulary at the experiment's resolved path without modifying it."""
    descriptions = json.loads(path.read_text(encoding="utf-8"))["descriptions"]
    if not isinstance(descriptions, dict):
        raise ValueError(f"Expected a descriptions mapping in {path}")
    return descriptions


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
        for label in classes:
            captions = descriptions[label]
            if (
                not isinstance(captions, (list, tuple))
                or not captions
                or any(not isinstance(text, str) or not text.strip() for text in captions)
            ):
                raise ValueError(f"Expected a nonempty list of nonblank captions for {label}")
            records.append(PointRecord(label, "description_prototype", label, json.dumps(captions)))
    return records
