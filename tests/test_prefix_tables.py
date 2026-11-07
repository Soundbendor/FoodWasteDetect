"""Prefix tables derived through the storage interface on temporary LanceDB."""

from unittest.mock import Mock

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from config_helpers import config_data
from lancedb.db import LanceDBConnection

from classification.experiments.config import ExperimentConfig
from classification.retrieval import lancedb as lancedb_module
from classification.retrieval.lancedb import LanceDBIndex
from classification.schemas import PointRecord

ROWS = [
    PointRecord("A", "class_name", "apple", "apple"),
    PointRecord("B", "training_image", "train/b.jpg", "train/b.jpg"),
]
VECTORS = np.array([[0.1, 0.2, 0.3, 0.4], [-1.5, 2.25, 1e-8, 7.0]], dtype=np.float32)


@pytest.fixture
def source(tmp_path):
    index = LanceDBIndex.create(tmp_path / "db", "source", 4, metadata={"unrelated": "kept"})
    index.append(ROWS, VECTORS)
    return index


def rows(index):
    return index.table.to_arrow().to_pylist()


def test_prefix_keeps_rows_and_replaces_vectors_with_float32_prefixes(source):
    derived = source.derive_prefix("source_d2", 2, batch_size=1)

    assert derived.name == "source_d2"
    assert derived.dimension == 2
    assert LanceDBIndex.open(source.uri, "source_d2").dimension == 2
    expected = [
        dict(row, vector=prefix)
        for row, prefix in zip(
            rows(source),
            [[np.float32(0.1), np.float32(0.2)], [-1.5, 2.25]],
            strict=True,
        )
    ]
    assert rows(derived) == expected
    assert derived.table.schema.metadata[b"unrelated"] == b"kept"
    assert source.dimension == 4


def test_prefix_wider_than_source_fails_before_writing(source):
    with pytest.raises(ValueError, match="exceeds the stored width 4"):
        source.derive_prefix("too_wide", 5, batch_size=1)

    assert not LanceDBIndex.exists(source.uri, "too_wide")


def test_full_width_prefix_copies_vectors_unchanged(source):
    derived = source.derive_prefix("copy", 4, batch_size=1)

    assert rows(derived) == rows(source)


def test_existing_destination_is_reused_without_reading_source(source, monkeypatch):
    first = source.derive_prefix("source_d2", 2, batch_size=1)
    version = first.table.version
    monkeypatch.setattr(
        lancedb_module, "_prefix_batches", Mock(side_effect=AssertionError("source scan forbidden"))
    )
    broken = LanceDBIndex.create(
        source.uri, "broken_snapshot", 4, metadata={"classification.build": "{broken"}
    )

    reused = broken.derive_prefix("source_d2", 2, batch_size=1)

    assert reused.table.version == version
    assert rows(reused) == rows(first)


@pytest.mark.parametrize(
    "encoded",
    [
        ExperimentConfig.model_validate(
            config_data(embedding={"model_id": "google/ge-v2", "truncate_dim": 3})
        ).model_dump_json(),
        "{broken",
        "",
    ],
)
def test_creation_snapshot_is_copied_verbatim(tmp_path, encoded):
    metadata = {"classification.build": encoded, "unrelated": "kept"}
    source = LanceDBIndex.create(tmp_path / "db", "source", 4, metadata=metadata)

    derived = source.derive_prefix("source_d2", 2, batch_size=1)

    expected = {key.encode(): value.encode() for key, value in metadata.items()}
    assert derived.table.schema.metadata == expected
    assert LanceDBIndex.open(source.uri, "source_d2").table.schema.metadata == expected
    assert source.table.schema.metadata == expected


def test_absent_snapshot_stays_absent(source):
    derived = source.derive_prefix("source_d2", 2, batch_size=1)

    assert derived.table.schema.metadata == {b"unrelated": b"kept"}


def test_stream_copies_every_row_without_a_result_limit(source):
    many = [PointRecord("A", "training_image", f"extra-{i}", f"extra-{i}") for i in range(23)]
    source.append(many, np.ones((23, 4), dtype=np.float32))

    derived = source.derive_prefix("many_d3", 3, batch_size=3)

    assert derived.table.count_rows() == 25
    assert rows(derived) == [dict(row, vector=row["vector"][:3]) for row in rows(source)]


def test_stream_failure_propagates_and_leaves_source_unchanged(source, monkeypatch):
    snapshot = rows(source)
    original = lancedb_module._prefix_batches

    def failing_after_one_batch(*args):
        yield next(original(*args))
        raise RuntimeError("synthetic reader failure")

    monkeypatch.setattr(lancedb_module, "_prefix_batches", failing_after_one_batch)

    # Arrow may wrap the Python error while consuming the reader natively.
    with pytest.raises(Exception, match="synthetic reader failure"):
        source.derive_prefix("broken", 2, batch_size=1)

    assert rows(source) == snapshot


@pytest.mark.parametrize("vector_type", [None, pa.list_(pa.float32()), pa.int32()])
def test_dimension_requires_positive_fixed_size_vector(tmp_path, vector_type):
    fields = [pa.field("id", pa.string())]
    if vector_type is not None:
        fields.append(pa.field("vector", vector_type))
    table = lancedb.connect(tmp_path / "db").create_table("malformed", schema=pa.schema(fields))

    with pytest.raises(ValueError, match="fixed-size vector"):
        LanceDBIndex(tmp_path / "db", table).dimension


@pytest.mark.parametrize(
    ("operation", "existing"), [("open_table", True), ("create_table", False)]
)
def test_storage_errors_propagate_without_cleanup(source, monkeypatch, operation, existing):
    if existing:
        source.derive_prefix("destination", 2, batch_size=1)
    drop = Mock()
    monkeypatch.setattr(LanceDBConnection, "drop_table", drop)
    monkeypatch.setattr(LanceDBConnection, operation, Mock(side_effect=RuntimeError("storage failure")))

    with pytest.raises(RuntimeError, match="storage failure"):
        source.derive_prefix("destination", 2, batch_size=1)

    drop.assert_not_called()
