"""Fingerprints of the configuration settings that change stored vectors.

Each key names the setting it records, so incompatible-table errors read as
configuration paths. Settings that never change vectors, such as device, batch size,
Vertex location, concurrency, attempts, timeout, dataset root, and output directory,
are deliberately excluded and never block resumption.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence

from classification.experiments.config import ExperimentConfig
from classification.retrieval.embedding_table import Fingerprint


def encoder_fingerprint(config: ExperimentConfig) -> Fingerprint:
    """Return the encoder settings that determine every stored vector.

    Args:
        config: Current experiment configuration.

    Returns:
        The model ID, truncation dimension, and Vertex classification-prompt flag.
    """
    embedding = config.embedding
    return {
        "embedding.model_id": embedding.model_id,
        "embedding.truncate_dim": embedding.truncate_dim,
        "embedding.vertex.classification_prompt": embedding.vertex.classification_prompt,
    }


def inputs_fingerprint(
    config: ExperimentConfig, vocabulary: Mapping[str, Sequence[str]] | None
) -> Fingerprint:
    """Return the settings and vocabulary that select a reference table's rows.

    Args:
        config: Current experiment configuration.
        vocabulary: The loaded class-description mapping, or None when descriptions
            are disabled.

    Returns:
        The dataset name, the three reference-source switches, the per-class image
        cap, and the vocabulary's SHA-256 (None without a vocabulary).
    """
    return {
        "dataset.name": config.dataset.name,
        "embed_images": config.embed_images,
        "embed_class_labels": config.embed_class_labels,
        "embed_descriptions": config.embed_descriptions,
        "n_imgs_per_class": config.n_imgs_per_class,
        "vocabulary_sha256": None if vocabulary is None else vocabulary_digest(vocabulary),
    }


def vocabulary_digest(vocabulary: Mapping[str, Sequence[str]]) -> str:
    """Return the SHA-256 of a vocabulary as canonical JSON.

    Canonical JSON sorts class keys, keeps each class's description order, uses no
    insignificant whitespace, and encodes non-ASCII text as UTF-8.

    Args:
        vocabulary: Class labels mapped to their ordered descriptions.

    Returns:
        The lowercase hexadecimal digest.
    """
    canonical = json.dumps(
        {label: list(descriptions) for label, descriptions in vocabulary.items()},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
