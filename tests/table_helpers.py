"""Fingerprint arguments for embedding tables created directly by tests."""

from classification.experiments.config import ExperimentConfig
from classification.experiments.fingerprints import encoder_fingerprint, inputs_fingerprint

SYNTHETIC_ENCODER = {"embedding.model_id": "synthetic", "embedding.truncate_dim": None}
SYNTHETIC_INPUTS = {"dataset.name": "synthetic", "n_imgs_per_class": None}


def fingerprints(role, config=None, vocabulary=None):
    """Return ``encoder``/``inputs`` keyword arguments for a table of ``role``.

    Without a configuration, return fixed synthetic fingerprints. With one, return the
    fingerprints the experiment computes, so builds and evaluations accept the table.
    """
    if config is None:
        encoder, inputs = SYNTHETIC_ENCODER, SYNTHETIC_INPUTS
    else:
        if not isinstance(config, ExperimentConfig):
            config = ExperimentConfig.model_validate(config)
        encoder, inputs = encoder_fingerprint(config), inputs_fingerprint(config, vocabulary)
    return {"encoder": encoder, "inputs": inputs if role == "reference" else None}
