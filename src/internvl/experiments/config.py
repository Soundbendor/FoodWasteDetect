"""Shared experiment configuration validation."""

from pathlib import Path


# TODO: Review overly defensive config checks
def validate_config(config: dict) -> dict:
    """Validate experiment config does not force exp pipeline into invalid state.

    - Require that at least one modality (class labels, images, descriptions)
        is used in training dataset
    - Ensure that results output directory is not nested inside data directory.
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


