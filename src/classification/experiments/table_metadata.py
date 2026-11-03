"""Creation snapshots for classification tables, independent of storage and models."""

import json
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict, cast

BUILD_METADATA_KEY = "classification.build"


class TableMetadata(TypedDict):
    """Versioned configuration snapshot written only at table creation."""

    metadata_version: int
    table_kind: Literal["reference", "validation"]
    vector_dimension: int
    config: dict[str, Any]
    derived_from: NotRequired[dict[str, str | int]]


def build_metadata(
    config: Mapping[str, Any],
    table_kind: Literal["reference", "validation"],
    dimension: int,
) -> TableMetadata:
    """Select configuration fields without copying runtime or sensitive settings."""
    options = config["embedding"]
    embedding: dict[str, Any] = {
        "model_id": options.get("model_id"),
        "truncate_dim": options.get("truncate_dim"),
    }
    if options.get("model_id") == "google/ge-v2":
        embedding["vertex"] = {
            "classification_prompt": options.get("vertex", {}).get("classification_prompt", False)
        }
    settings: dict[str, Any] = {
        "dataset": {"name": config["dataset"].get("name")},
        "embedding": embedding,
    }
    if table_kind == "reference":
        settings.update(
            embed_images=config["embed_images"],
            embed_class_labels=config["embed_class_labels"],
            embed_descriptions=config["embed_descriptions"],
            n_imgs_per_class=config.get("n_imgs_per_class"),
        )
        if config["embed_descriptions"]:
            settings["description_count"] = config.get("description_count")
    return {
        "metadata_version": 1,
        "table_kind": table_kind,
        "vector_dimension": dimension,
        "config": settings,
    }


def serialize_metadata(snapshot: TableMetadata) -> dict[str, str]:
    """Encode a snapshot as UTF-8-compatible Arrow schema metadata."""
    return {BUILD_METADATA_KEY: json.dumps(snapshot, ensure_ascii=False)}


def derive_metadata(
    metadata: Mapping[bytes, bytes] | None,
    source_table: str,
    source_dimension: int,
    dimension: int,
) -> dict[bytes, bytes]:
    """Preserve schema metadata and record slicing lineage only for known snapshots."""
    result = dict(metadata or {})
    key = BUILD_METADATA_KEY.encode("utf-8")
    if key in result:
        snapshot = cast(TableMetadata, json.loads(result[key]))
        snapshot["vector_dimension"] = dimension
        snapshot["derived_from"] = {
            "table": source_table,
            "vector_dimension": source_dimension,
        }
        result[key] = serialize_metadata(snapshot)[BUILD_METADATA_KEY].encode("utf-8")
    return result
