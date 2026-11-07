"""Experiment configuration snapshots persist in real temporary LanceDB tables."""

import itertools
import json

import numpy as np
import pytest
from config_helpers import config_data, updated
from lancedb.db import LanceDBConnection

from classification.experiments.config import ExperimentConfig
from classification.retrieval.lancedb import LanceDBIndex
from classification.schemas import PointRecord


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=3))[1:])
@pytest.mark.parametrize("cap", [None, 3])
@pytest.mark.parametrize("truncate", [None, 2])
def test_full_config_round_trip(flags, cap, truncate):
    config = ExperimentConfig.model_validate(
        config_data(
            dataset={"name": "FoodX251", "root": "/synthetic/raw"},
            embedding={
                "model_id": "jinaai/jina-clip-v2",
                "truncate_dim": truncate,
                "device": "cpu",
                "vertex": {"project": "synthetic-project"},
            },
            **dict(
                zip(
                    ("embed_images", "embed_class_labels", "embed_descriptions"), flags, strict=True
                )
            ),
            n_imgs_per_class=cap,
            description_count=5,
            vector_dimension=2,
            k=99,
            ollama=dict(
                binary_path="/synthetic/ollama",
                host="http://localhost:11434",
                startup_timeout=10.0,
                request_timeout=30.0,
            ),
        )
    )
    encoded = config.model_dump_json()
    assert ExperimentConfig.model_validate_json(encoded) == config
    settings = json.loads(encoded)
    assert settings == config.model_dump(mode="json")
    assert settings["vector_dimension"] == 2
    assert settings["embedding"]["batch_size"] == 32
    assert settings["embedding"]["vertex"]["location"] == "global"
    assert settings["embedding"]["vertex"]["project"] == "synthetic-project"
    assert settings["ollama"]["llm"] is None
    assert settings["descriptions_file"] is None
    assert settings["lancedb"]["validation_table"] is None
    assert settings["cache_validation_embeddings"] is False


@pytest.mark.parametrize("with_snapshot", [False, True])
def test_config_persistence_and_derived_dimensions(tmp_path, with_snapshot):
    config = ExperimentConfig.model_validate(
        config_data(
            embedding={"model_id": "google/ge-v2", "truncate_dim": 3},
            vector_dimension=3,
        )
    )
    metadata = {"unrelated": "preserved café"}
    if with_snapshot:
        metadata["classification.build"] = config.model_dump_json()
    index = LanceDBIndex.create(tmp_path / "db", "source", 3, metadata=metadata)
    expected = {key.encode(): value.encode() for key, value in metadata.items()}
    assert index.table.schema.metadata == expected
    schema = index.table.schema.remove_metadata()
    for label, vector in [("A", [1.0, 0.0, 0.0]), ("B", [0.0, 1.0, 0.0])]:
        index.append([PointRecord(label, "validation_image", label, label)], np.array([vector]))
        assert index.table.schema.metadata == expected
    before = index.query(np.array([1.0, 0.0, 0.0]), 2)
    index.optimize()
    reopened = LanceDBIndex.open(index.uri, "source")
    assert reopened.table.schema.metadata == expected
    assert reopened.table.schema.remove_metadata() == schema
    assert reopened.query(np.array([1.0, 0.0, 0.0]), 2) == before
    connection = LanceDBConnection(index.uri)
    LanceDBIndex.open(index.uri, "source").derive_prefix("derived", 2, batch_size=1)
    destination = connection.open_table("derived")
    derived = destination.schema.metadata
    assert derived[b"unrelated"] == expected[b"unrelated"]
    if with_snapshot:
        snapshot = ExperimentConfig.model_validate_json(derived[b"classification.build"])
        assert snapshot == updated(config, "vector_dimension", 2)
        assert snapshot.embedding.truncate_dim == 3
    else:
        assert derived == expected
    assert destination.to_arrow().to_pylist() == [
        dict(row, vector=row["vector"][:2]) for row in reopened.table.to_arrow().to_pylist()
    ]
    version = destination.version
    LanceDBIndex.open(index.uri, "source").derive_prefix("derived", 1, batch_size=1)
    destination = connection.open_table("derived")
    assert destination.version == version
    assert destination.schema.metadata == derived
    assert connection.open_table("source").schema.metadata == expected

    LanceDBIndex.open(index.uri, "derived").derive_prefix("second_generation", 1, batch_size=1)
    second_metadata = connection.open_table("second_generation").schema.metadata
    assert second_metadata[b"unrelated"] == expected[b"unrelated"]
    if with_snapshot:
        second = ExperimentConfig.model_validate_json(second_metadata[b"classification.build"])
        assert second == updated(config, "vector_dimension", 1)
    else:
        assert second_metadata == expected


def test_missing_schema_metadata_is_supported(tmp_path):
    index = LanceDBIndex.create(tmp_path / "db", "source", 3)
    index.append([PointRecord("A", "class_name", "A", "A")], np.array([[1.0, 0.0, 0.0]]))
    connection = LanceDBConnection(index.uri)
    LanceDBIndex.open(index.uri, "source").derive_prefix("derived", 2, batch_size=1)
    assert b"classification.build" not in (connection.open_table("derived").schema.metadata or {})
