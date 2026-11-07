"""Strict configuration validation without datasets, encoders, or services."""

from itertools import product
from pathlib import Path

import pytest
import yaml
from config_helpers import config_data, updated
from pydantic import ValidationError

from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig, OllamaConfig
from classification.experiments.report_types import IndexSummary


def test_defaults_spelling_and_mutable_nested_attributes():
    config = ExperimentConfig.model_validate(
        config_data(dataset={"name": "FoodX251", "root": "/raw"})
    )
    assert config.dataset.name == "FoodX251"
    assert config.embedding.batch_size == 32
    assert config.embedding.device == "auto"
    assert config.embedding.truncate_dim is None
    assert config.embedding.vertex.model_dump() == {
        "project": None,
        "location": "global",
        "max_concurrency": 4,
        "max_attempts": 5,
        "request_timeout": 60.0,
        "classification_prompt": False,
    }
    assert config.n_imgs_per_class is config.descriptions_file is config.ollama is None
    assert config.description_count is None
    assert config.vector_dimension is None
    assert config.cache_validation_embeddings is False
    config.k = 3
    config.embedding.batch_size = 5
    config.vector_dimension = 2
    assert config.k == 3
    assert config.embedding.batch_size == 5
    assert config.vector_dimension == 2


@pytest.mark.parametrize(
    "key",
    [
        "dataset",
        "embedding",
        "lancedb",
        "output_dir",
        "k",
        "embed_images",
        "embed_class_labels",
        "embed_descriptions",
    ],
)
def test_required_fields(key):
    data = config_data()
    del data[key]
    with pytest.raises(ValidationError, match=key):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("section", [None, "dataset", "embedding", "lancedb", "vertex"])
def test_unknown_fields(section):
    data = config_data()
    target = (
        data
        if section is None
        else (
            data["embedding"].setdefault("vertex", {})
            if section == "vertex"
            else data[section]
        )
    )
    target["unknown"] = 1
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize(
    "path",
    [
        "k",
        "embedding.batch_size",
        "embedding.truncate_dim",
        "n_imgs_per_class",
        "description_count",
        "vector_dimension",
        "embedding.vertex.max_attempts",
        "embedding.vertex.max_concurrency",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, False, "2", 1.5])
def test_integer_constraints(path, value):
    with pytest.raises(ValidationError):
        updated(ExperimentConfig.model_validate(config_data()), path, value)


@pytest.mark.parametrize(
    "path",
    [
        "embed_images",
        "embed_class_labels",
        "embed_descriptions",
        "cache_validation_embeddings",
        "embedding.vertex.classification_prompt",
    ],
)
@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_strict_booleans(path, value):
    with pytest.raises(ValidationError):
        updated(ExperimentConfig.model_validate(config_data()), path, value)


@pytest.mark.parametrize("flags", list(product([False, True], repeat=3)))
def test_reference_selection(flags):
    data = config_data(
        **dict(
            zip(
                ("embed_images", "embed_class_labels", "embed_descriptions"),
                flags,
                strict=True,
            )
        )
    )
    if any(flags):
        ExperimentConfig.model_validate(data)
    else:
        with pytest.raises(ValidationError, match="at least one reference"):
            ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("value", [0, -1, True, "2", float("nan"), float("inf")])
def test_timeout_constraints(value):
    with pytest.raises(ValidationError):
        OllamaConfig(
            binary_path="ollama",
            host="localhost",
            startup_timeout=value,
            request_timeout=2,
        )
    with pytest.raises(ValidationError):
        updated(
            ExperimentConfig.model_validate(config_data()),
            "embedding.vertex.request_timeout",
            value,
        )


def test_output_directory_must_be_separate_from_raw_data(tmp_path):
    data = config_data(dataset={"name": "foodx251", "root": str(tmp_path)})
    data["output_dir"] = str(tmp_path / "output")
    with pytest.raises(ValidationError, match="separate from raw"):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("table", ["", " ", "references"])
def test_unusable_validation_table_fails_at_load(table, cache):
    data = config_data(cache_validation_embeddings=cache)
    data["lancedb"]["validation_table"] = table
    with pytest.raises(ValidationError, match=r"lancedb\.validation_table"):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize(
    "settings",
    [{"cache_validation_embeddings": True}, {"sweep_dimensions": [8, 16]}],
    ids=["cache", "sweep"],
)
def test_validation_table_required_for_cache_or_sweep(settings):
    data = config_data(**settings)
    with pytest.raises(ValidationError, match=r"lancedb\.validation_table is required"):
        ExperimentConfig.model_validate(data)
    data["lancedb"]["validation_table"] = "validation"
    assert ExperimentConfig.model_validate(data).lancedb.validation_table == "validation"


def test_validation_table_optional_without_cache_or_sweep():
    assert ExperimentConfig.model_validate(config_data()).lancedb.validation_table is None


@pytest.mark.parametrize(
    "name", ["classification.yml", "classification_embeddinggemma2_images.yml", "ge_v2.yml"]
)
def test_shipped_experiment_configurations_load(name):
    data = yaml.safe_load((Path("cfg") / name).read_text())
    assert ExperimentConfig.model_validate(data).model_dump(exclude_unset=True) == data


def test_optional_services_and_report_serialization():
    config = ExperimentConfig.model_validate(config_data())
    summary = IndexSummary(
        table="references", uri="/db", source_counts={"class_name": 2}, config=config
    )
    assert summary.config is config
    assert summary.model_dump(mode="json")["config"] == config.model_dump(mode="json")
    assert '"batch_size":32' in summary.model_dump_json()
    # Only the backend constructor requires the project.
    config = updated(config, "embedding.model_id", "google/ge-v2")
    experiment = Experiment.__new__(Experiment)
    experiment.config = config
    with pytest.raises(ValueError, match="vertex.project"):
        experiment._create_embedder()


def test_python_api_requires_model():
    with pytest.raises(TypeError, match="ExperimentConfig"):
        Experiment(config_data())
