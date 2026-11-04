"""Valid synthetic configuration inputs and validated replacements."""

from classification.experiments.config import ExperimentConfig


def config_data(**updates):
    values = {
        "dataset": {"name": "foodx251", "root": "/synthetic/raw"},
        "embedding": {"model_id": "jinaai/jina-clip-v2"},
        "lancedb": {"uri": "/synthetic/db", "table": "references"},
        "output_dir": "/synthetic/output",
        "k": 1,
        "embed_images": False,
        "embed_class_labels": True,
        "embed_descriptions": False,
    }
    values.update(updates)
    return values


def updated(config, path, value):
    data = config.model_dump()
    target = data
    *parents, key = path.split(".")
    for parent in parents:
        target = target[parent]
    target[key] = value
    return ExperimentConfig.model_validate(data)
