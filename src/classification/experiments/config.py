"""Shared experiment configuration validation."""

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any


def validate_ollama_config(settings: Mapping[str, Any]) -> None:
    """Require finite, positive timeout seconds before opening an Ollama session."""
    for name in ("startup_timeout", "request_timeout"):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"ollama.{name} must be positive finite seconds")


def validate_config(config: dict) -> dict:
    """Check reference switches, retrieval limits, and output/cache destinations.

    Require at least one reference source, a positive integer k, and a null or
    positive integer image cap. Booleans are invalid for both integer settings.
    Output must stay outside the raw dataset; enabled validation caching needs a
    nonempty table name distinct from the references.

    Returns:
        The supplied configuration unchanged.

    Raises:
        ValueError: A checked setting violates these constraints.
        KeyError: A required setting is missing.
    """
    if type(config["k"]) is not int or config["k"] <= 0:
        raise ValueError("k must be a positive integer")
    sources = ("embed_images", "embed_class_labels", "embed_descriptions")
    for key in sources:
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not any(config[key] for key in sources):
        raise ValueError("Select at least one reference source")
    limit = config.get("n_imgs_per_class")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("n_imgs_per_class must be null or a positive integer")
    # Reject output paths equal to or nested inside the resolved raw dataset directory.
    raw = config.get("dataset", {}).get("root")
    if raw:
        output = Path(config["output_dir"]).expanduser().resolve()
        if output.is_relative_to(Path(raw).expanduser().resolve()):
            raise ValueError("Output directory must be separate from raw data")
    if config.get("cache_validation_embeddings", False):
        name = config["lancedb"].get("validation_table")
        if not isinstance(name, str) or not name.strip() or name == config["lancedb"]["table"]:
            raise ValueError("validation_table must be nonempty and distinct from reference table")
    return config
