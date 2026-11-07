"""Embedding table contracts on temporary LanceDB, observed through the module interface."""

import multiprocessing
from unittest.mock import Mock, patch

import lancedb
import numpy as np
import pyarrow as pa
import pytest
from lancedb.db import LanceDBConnection

from classification.retrieval.embedding_table import (
    EmbeddingTable,
    EmbeddingTableError,
    IncompatibleTableError,
    IncompleteTableError,
    LegacyTableError,
    PrefixSource,
)
from classification.retrieval.voting import classify
from classification.schemas import PointRecord
from table_helpers import SYNTHETIC_ENCODER, SYNTHETIC_INPUTS, fingerprints


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_new_table_header_round_trips_on_reopen(tmp_path, role):
    created = EmbeddingTable.create(tmp_path / "db", "table", role, 3, snapshot='{"kept": 1}', **fingerprints(role))

    reopened = EmbeddingTable.open(tmp_path / "db", "table", role)

    for table in (created, reopened):
        assert (table.role, table.dimension, table.prefix_source) == (role, 3, None)
        assert table.creation_snapshot == '{"kept": 1}'


def legacy_table(uri, name, metadata):
    """Write a table the way earlier code did: the same columns, a foreign header."""
    fields = [pa.field(column, pa.string(), nullable=False)
              for column in ("id", "class", "source_kind", "source_identity", "value")]
    fields.append(pa.field("vector", pa.list_(pa.float32(), 2), nullable=False))
    lancedb.connect(uri).create_table(name, schema=pa.schema(fields, metadata=metadata))


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {"classification.build": "{}"},
        {"classification.embedding_table": "{broken", "classification.creation_snapshot": ""},
        {
            "classification.embedding_table": (
                '{"schema_version": 0, "role": "reference", "dimension": 2}'
            ),
            "classification.creation_snapshot": "",
        },
        {
            "classification.embedding_table": (
                '{"schema_version": 1, "role": "reference", "dimension": 2}'
            ),
            "classification.creation_snapshot": "",
        },
        {
            "classification.embedding_table": (
                '{"schema_version": 2, "role": "reference", "dimension": 3,'
                ' "encoder": {}, "inputs": {}}'
            ),
            "classification.creation_snapshot": "",
        },
        {
            "classification.embedding_table": (
                '{"schema_version": 2, "role": "validation_cache", "dimension": 2,'
                ' "encoder": {}, "inputs": {}}'
            ),
            "classification.creation_snapshot": "",
        },
    ],
    ids=[
        "no-metadata", "old-snapshot-only", "malformed", "other-version",
        "before-fingerprints", "wrong-width", "cache-with-inputs",
    ],
)
@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_tables_without_a_valid_header_are_refused_as_legacy(tmp_path, metadata, role):
    legacy_table(tmp_path / "db", "old", metadata)

    for open_table in (
        lambda: EmbeddingTable.open(tmp_path / "db", "old", role),
        lambda: EmbeddingTable.open_or_create(
            tmp_path / "db", "old", role, lambda: 2, snapshot="new", **fingerprints(role)),
    ):
        with pytest.raises(LegacyTableError, match="Rebuild it under a new table name") as error:
            open_table()
        assert "ADR-0001" in str(error.value)
        assert isinstance(error.value, EmbeddingTableError)
        assert isinstance(error.value, ValueError)


@pytest.mark.parametrize(("stored", "requested"), [
    ("reference", "validation_cache"), ("validation_cache", "reference"),
])
def test_opening_under_the_wrong_role_fails(tmp_path, stored, requested):
    EmbeddingTable.create(tmp_path / "db", "table", stored, 2, snapshot="", **fingerprints(stored))

    for open_table in (
        lambda: EmbeddingTable.open(tmp_path / "db", "table", requested),
        lambda: EmbeddingTable.open_or_create(
            tmp_path / "db", "table", requested, lambda: 2, snapshot="", **fingerprints(requested)),
    ):
        with pytest.raises(EmbeddingTableError, match=f"is a {stored} table") as error:
            open_table()
        assert not isinstance(error.value, LegacyTableError)


def test_validation_caches_refuse_queries_and_reference_tables_refuse_vector_fetches(tmp_path):
    cache = EmbeddingTable.create(tmp_path / "db", "cache", "validation_cache", 2, snapshot="", **fingerprints("validation_cache"))
    reference = EmbeddingTable.create(tmp_path / "db", "reference", "reference", 2, snapshot="", **fingerprints("reference"))

    with pytest.raises(EmbeddingTableError, match="Cannot query table 'cache'"):
        cache.query(np.ones(2), 1)
    with pytest.raises(EmbeddingTableError, match="Cannot fetch vectors from table 'reference'"):
        reference.fetch_vectors([])


def records_of(table):
    return [record for record, _ in table.rows()]


def vectors_of(table):
    return np.asarray([vector for _, vector in table.rows()], dtype=np.float32)


def populated(uri, role="reference", dtype=np.float32):
    table = EmbeddingTable.create(uri, "fixture", role, 2, snapshot="", **fingerprints(role))
    kind = "class_name" if role == "reference" else "validation_image"
    records = [PointRecord(label, kind, label, label) for label in ("A", "B", "C")]
    vectors = np.array([[3., 0.], [1., 1.], [-2., 0.]], dtype=dtype)
    table.append(records[:2], vectors[:2])
    table.append(records[2:], vectors[2:])
    return table, records, vectors


# Storage, search, and counts


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_records_and_vectors_round_trip_with_exact_cosine_search(tmp_path, dtype):
    _, records, vectors = populated(tmp_path / "db", dtype=dtype)

    reopened = EmbeddingTable.open(tmp_path / "db", "fixture", "reference")

    assert records_of(reopened) == records
    np.testing.assert_allclose(vectors_of(reopened), vectors)
    assert vectors_of(reopened).dtype == np.float32
    neighbors = reopened.query(np.array([2., 0.], dtype=dtype), 10)
    assert [n["label"] for n in neighbors] == ["A", "B", "C"]
    assert [n["id"] for n in neighbors] == [record.id for record in records]
    np.testing.assert_allclose([n["score"] for n in neighbors], [1., 2**-0.5, -1.], atol=1e-6)
    assert len(reopened.query(vectors[0], 1)) == 1


def test_existing_names_are_never_recreated(tmp_path):
    table, records, _ = populated(tmp_path / "db")

    with pytest.raises(ValueError):
        EmbeddingTable.create(tmp_path / "db", "fixture", "reference", 2, snapshot="other", **fingerprints("reference"))

    assert records_of(table) == records
    assert table.creation_snapshot == ""


def test_full_width_vectors_round_trip_with_cosine_search(tmp_path):
    table = EmbeddingTable.create(tmp_path / "db", "fixture", "reference", 2048, snapshot="", **fingerprints("reference"))
    records = [PointRecord(label, "class_name", label, label) for label in ("A", "B")]
    vectors = np.zeros((2, 2048), dtype=np.float32)
    vectors[0, -1] = 3
    vectors[1, 0] = 2
    table.append(records, vectors)

    reopened = EmbeddingTable.open(tmp_path / "db", "fixture", "reference")

    assert reopened.dimension == 2048
    neighbors = reopened.query(vectors[0] * 2, 10)
    assert [n["id"] for n in neighbors] == [r.id for r in records]
    np.testing.assert_allclose([n["score"] for n in neighbors], [1., 0.], atol=1e-6)
    assert classify(neighbors) == ("A", 0.5)


def test_count_mismatch_and_bad_vectors_append_nothing(tmp_path):
    table = EmbeddingTable.create(tmp_path / "db", "fixture", "reference", 2, snapshot="", **fingerprints("reference"))
    record = PointRecord("A", "class_name", "A", "A")
    assert table.query(np.array([1., 0.]), 10) == []

    for records, vectors in [([record], np.empty((0, 2))), ([], np.ones((1, 2)))]:
        with pytest.raises(ValueError, match="Record/vector count mismatch"):
            table.append(records, vectors)
    for vector in ([[1., 2., 3.]], [[float("nan"), 1.]]):
        with pytest.raises((ValueError, pa.ArrowInvalid)):
            table.append([record], np.array(vector))

    assert table.rows() == []


def test_missing_storage_is_never_created(tmp_path):
    uri = tmp_path / "db"

    assert not EmbeddingTable.exists(uri, "fixture")
    with pytest.raises(FileNotFoundError):
        EmbeddingTable.open(uri, "fixture", "reference")
    assert not uri.exists()
    uri.mkdir()
    with pytest.raises(ValueError):
        EmbeddingTable.open(uri, "fixture", "reference")
    assert list(uri.iterdir()) == []


def test_database_paths_resolve_home_and_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    home = EmbeddingTable.create("~/fixtures", "home", "reference", 2, snapshot="", **fingerprints("reference"))
    relative = EmbeddingTable.create("fixtures", "relative", "reference", 2, snapshot="", **fingerprints("reference"))

    assert home.uri == tmp_path / "home" / "fixtures"
    assert relative.uri == tmp_path / "fixtures"
    assert EmbeddingTable.exists(tmp_path / "home" / "fixtures", "home")


def concurrent_reader(uri, barrier, queue):
    table = EmbeddingTable.open(uri, "fixture", "reference")
    barrier.wait(timeout=30)
    queue.put([table.query(np.array([1., 0.]), 10) for _ in range(10)])


def test_concurrent_readers_preserve_the_table(tmp_path):
    table, records, vectors = populated(tmp_path / "db")
    ctx = multiprocessing.get_context("spawn")
    barrier, queue = ctx.Barrier(3), ctx.Queue()
    processes = [
        ctx.Process(target=concurrent_reader, args=(str(tmp_path / "db"), barrier, queue))
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            for result in queue.get(timeout=45):
                assert [n["label"] for n in result] == ["A", "B", "C"]
                np.testing.assert_allclose(
                    [n["score"] for n in result], [1., 2**-0.5, -1.], atol=1e-6
                )
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        queue.close()

    reopened = EmbeddingTable.open(tmp_path / "db", "fixture", "reference")
    assert records_of(reopened) == records
    np.testing.assert_array_equal(vectors_of(reopened), vectors)


def test_vector_fetch_preserves_requested_order_repeats_and_quotes(tmp_path):
    table = EmbeddingTable.create(tmp_path / "db", "vectors", "validation_cache", 3, snapshot="", **fingerprints("validation_cache"))
    records = [PointRecord("Apple", "validation_image", f"quoted'{i}", "value") for i in range(26)]
    vectors = np.asarray([[i + 1, i + 2, i + 3] for i in range(26)], dtype=np.float32)
    order = np.random.default_rng(12).permutation(26)
    table.append([records[i] for i in order], vectors[order])
    keys = [record.source_key for record in records]

    requested = list(reversed(range(26))) + [3, 0]

    assert table.missing(keys) == []
    np.testing.assert_array_equal(table.fetch_vectors([keys[i] for i in requested]), vectors[requested])
    assert table.fetch_vectors([]).shape == (0, 3)
    with pytest.raises(KeyError):
        table.fetch_vectors([("validation_image", "absent")])


def test_record_exposes_its_source_key():
    record = PointRecord("Apple", "training_image", "foodx251:train:0.png", "0.png")

    assert record.source_key == ("training_image", "foodx251:train:0.png")


def test_class_counts_stream_every_row(tmp_path):
    table = EmbeddingTable.create(tmp_path, "counts", "reference", 2, snapshot="", **fingerprints("reference"))
    assert table.class_counts() == {}
    records = [PointRecord("A" if i % 2 else "B", kind, str(i), "")
               for kind in ("class_name", "description", "training_image") for i in range(1100)]

    table.append(records, np.ones((len(records), 2), dtype=np.float32))

    assert table.class_counts() == {"A": 1650, "B": 1650}


def test_source_counts_stream_every_row_including_duplicate_source_keys(tmp_path):
    table = EmbeddingTable.create(tmp_path, "counts", "reference", 2, snapshot="", **fingerprints("reference"))
    assert table.source_counts() == {}
    records = [PointRecord("A", kind, str(i), "")
               for kind, total in (("class_name", 2), ("training_image", 1500)) for i in range(total)]
    duplicates = [PointRecord("A", "class_name", "0", ""), PointRecord("B", "class_name", "0", "")]

    table.append(records + duplicates, np.ones((len(records) + 2, 2), dtype=np.float32))

    expected = {"class_name": 4, "training_image": 1500}
    assert table.source_counts() == expected
    assert EmbeddingTable.open(tmp_path, "counts", "reference").source_counts() == expected


def test_optimize_preserves_rows_header_and_search(tmp_path):
    table = EmbeddingTable.create(tmp_path / "db", "references", "reference", 2, snapshot="kept", **fingerprints("reference"))
    for i in range(32):
        table.append([PointRecord(str(i), "class_name", str(i), str(i))],
                     np.asarray([[1.0, i + 1.0]], dtype=np.float32))
    before = table.rows()
    neighbors = table.query(np.asarray([1.0, 1.0]), 5)

    table.optimize()

    reopened = EmbeddingTable.open(tmp_path / "db", "references", "reference")
    by_id = sorted(reopened.rows(), key=lambda row: row[0].id)
    assert [r for r, _ in by_id] == [r for r, _ in sorted(before, key=lambda row: row[0].id)]
    assert reopened.query(np.asarray([1.0, 1.0]), 5) == neighbors
    assert (reopened.dimension, reopened.creation_snapshot) == (2, "kept")


def test_open_or_create_computes_the_dimension_only_when_creating(tmp_path):
    dimension = Mock(return_value=3)
    created = EmbeddingTable.open_or_create(
        tmp_path / "db", "lazy", "reference", dimension, snapshot="kept", **fingerprints("reference"))
    assert created.dimension == 3
    dimension.assert_called_once_with()
    created.append([PointRecord("A", "class_name", "A", "A")], np.ones((1, 3)))

    reopened = EmbeddingTable.open_or_create(
        tmp_path / "db", "lazy", "reference", Mock(side_effect=AssertionError("encoder")),
        snapshot="replaced", **fingerprints("reference"))

    assert reopened.dimension == 3
    assert len(reopened.rows()) == 1
    assert reopened.creation_snapshot == "kept"


def test_missing_reports_absent_source_keys_once_in_request_order(tmp_path):
    table = EmbeddingTable.create(tmp_path / "db", "keys", "reference", 2, snapshot="", **fingerprints("reference"))
    assert table.missing([]) == []
    table.append(
        [PointRecord("A", "class_name", "A", "A"), PointRecord("A", "training_image", "it's", "f")],
        np.ones((2, 2)),
    )
    requested = [
        ("training_image", "B"), ("class_name", "A"), ("training_image", "A"),
        ("training_image", "B"), ("training_image", "it's"),
    ]

    assert table.missing(requested) == [("training_image", "B"), ("training_image", "A")]


# Completion


def four_rows(tmp_path, role):
    kind = "class_name" if role == "reference" else "validation_image"
    table = EmbeddingTable.create(tmp_path / "db", "table", role, 2, snapshot="", **fingerprints(role))
    for i in range(4):
        table.append([PointRecord("A", kind, str(i), str(i))], np.ones((1, 2)))
    return table, [(kind, str(i)) for i in range(4)]


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_mark_complete_states_the_missing_count_before_optimizing(tmp_path, role):
    table, keys = four_rows(tmp_path, role)
    expected = keys + [(keys[0][0], "absent"), ("training_image", "0"), (keys[0][0], "absent")]

    with patch.object(table, "optimize") as optimize, \
         pytest.raises(IncompleteTableError, match="missing 2 of 6 expected source keys") as error:
        table.mark_complete(expected)

    optimize.assert_not_called()
    assert isinstance(error.value, EmbeddingTableError)
    assert isinstance(error.value, ValueError)
    assert not EmbeddingTable.open(table.uri, "table", role).is_complete


@pytest.mark.parametrize(("role", "optimized"), [("reference", True), ("validation_cache", False)])
def test_mark_complete_optimizes_only_reference_tables(tmp_path, role, optimized):
    table, keys = four_rows(tmp_path, role)
    before = table.rows()

    with patch.object(table, "optimize", wraps=table.optimize) as optimize:
        table.mark_complete(keys[1:])

    assert optimize.called is optimized
    assert sorted(records_of(table), key=lambda r: r.id) == sorted(
        [r for r, _ in before], key=lambda r: r.id)


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_marking_complete_persists_across_reopen(tmp_path, role):
    table, keys = four_rows(tmp_path, role)
    assert not table.is_complete

    table.mark_complete(keys)

    assert table.is_complete
    assert EmbeddingTable.open(table.uri, "table", role).is_complete


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_appending_to_a_complete_table_makes_it_incomplete(tmp_path, role):
    table, keys = four_rows(tmp_path, role)
    table.mark_complete(keys)

    table.append([PointRecord("B", keys[0][0], "late", "late")], np.ones((1, 2)))

    assert not table.is_complete
    reopened = EmbeddingTable.open(table.uri, "table", role)
    assert not reopened.is_complete
    reopened.mark_complete(keys)
    assert EmbeddingTable.open(table.uri, "table", role).is_complete


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_opening_an_incomplete_table_for_reading_states_the_missing_count(tmp_path, role):
    table, keys = four_rows(tmp_path, role)
    expected = Mock(return_value=keys + [(keys[0][0], "absent"), (keys[0][0], "absent")])

    fix = "rerun the index stage" if role == "reference" else "validation caching enabled"
    with pytest.raises(IncompleteTableError, match="'table' is incomplete: missing 1 of 5") as error:
        EmbeddingTable.open_complete(table.uri, "table", role, expected=expected)
    assert fix in str(error.value)

    table.mark_complete(keys)
    table.append([PointRecord("B", keys[0][0], "absent", "")], np.ones((1, 2)))
    with pytest.raises(IncompleteTableError, match="missing 0 of 5"):
        EmbeddingTable.open_complete(table.uri, "table", role, expected=expected)


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_opening_a_complete_table_for_reading_skips_the_key_scan(tmp_path, role):
    table, keys = four_rows(tmp_path, role)
    table.mark_complete(keys)
    expected = Mock(side_effect=AssertionError("expected keys"))

    with patch.object(EmbeddingTable, "missing", side_effect=AssertionError("scan")):
        opened = EmbeddingTable.open_complete(table.uri, "table", role, expected=expected)

    assert (opened.role, opened.is_complete, len(opened.rows())) == (role, True, 4)


def test_optimizing_a_complete_table_makes_it_incomplete(tmp_path):
    table, keys = four_rows(tmp_path, "reference")
    table.mark_complete(keys)

    table.optimize()

    assert not EmbeddingTable.open(table.uri, "table", "reference").is_complete


def test_tables_written_before_completion_markers_open_as_incomplete(tmp_path):
    table, keys = four_rows(tmp_path, "reference")
    table.optimize()  # Earlier code optimized complete tables without marking them.

    reopened = EmbeddingTable.open(table.uri, "table", "reference")
    assert not reopened.is_complete
    with pytest.raises(IncompleteTableError, match="missing 0 of 4"):
        EmbeddingTable.open_complete(table.uri, "table", "reference", expected=lambda: keys)

    reopened.mark_complete(keys)
    assert EmbeddingTable.open_complete(table.uri, "table", "reference", expected=list).is_complete


# Fingerprints


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_fingerprints_round_trip_on_reopen(tmp_path, role):
    EmbeddingTable.create(tmp_path / "db", "table", role, 2, snapshot="", **fingerprints(role))

    reopened = EmbeddingTable.open(tmp_path / "db", "table", role)

    assert reopened.encoder_fingerprint == SYNTHETIC_ENCODER
    expected_inputs = SYNTHETIC_INPUTS if role == "reference" else None
    assert reopened.inputs_fingerprint == expected_inputs


def test_reference_tables_require_inputs_and_caches_refuse_them(tmp_path):
    with pytest.raises(EmbeddingTableError, match="requires an inputs fingerprint"):
        EmbeddingTable.create(
            tmp_path / "db", "reference", "reference", 2, snapshot="",
            encoder=SYNTHETIC_ENCODER, inputs=None,
        )
    with pytest.raises(EmbeddingTableError, match="has no inputs fingerprint"):
        EmbeddingTable.create(
            tmp_path / "db", "cache", "validation_cache", 2, snapshot="",
            encoder=SYNTHETIC_ENCODER, inputs=SYNTHETIC_INPUTS,
        )
    assert not EmbeddingTable.exists(tmp_path / "db", "reference")
    assert not EmbeddingTable.exists(tmp_path / "db", "cache")


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_resuming_with_a_different_encoder_lists_each_differing_setting(tmp_path, role):
    EmbeddingTable.create(tmp_path / "db", "table", role, 2, snapshot="", **fingerprints(role))
    current = fingerprints(role)
    current["encoder"] = dict(SYNTHETIC_ENCODER, **{
        "embedding.model_id": "other", "embedding.truncate_dim": 64,
    })

    with pytest.raises(IncompatibleTableError) as error:
        EmbeddingTable.open_or_create(
            tmp_path / "db", "table", role, Mock(side_effect=AssertionError("encoder")),
            snapshot="", **current,
        )

    assert error.value.differences == {
        "embedding.model_id": ("synthetic", "other"),
        "embedding.truncate_dim": (None, 64),
    }
    message = str(error.value)
    assert "embedding.model_id: stored 'synthetic', required 'other'" in message
    assert "embedding.truncate_dim: stored None, required 64" in message
    assert "new table name" in message
    assert isinstance(error.value, EmbeddingTableError)
    assert not isinstance(error.value, LegacyTableError)


def test_resuming_a_reference_table_with_different_inputs_fails(tmp_path):
    EmbeddingTable.create(
        tmp_path / "db", "table", "reference", 2, snapshot="", **fingerprints("reference")
    )
    changed = dict(SYNTHETIC_INPUTS, **{"n_imgs_per_class": 5, "vocabulary_sha256": "abc"})

    with pytest.raises(IncompatibleTableError) as error:
        EmbeddingTable.open_or_create(
            tmp_path / "db", "table", "reference", lambda: 2, snapshot="",
            encoder=SYNTHETIC_ENCODER, inputs=changed,
        )

    assert error.value.differences == {
        "n_imgs_per_class": (None, 5),
        "vocabulary_sha256": (None, "abc"),
    }


@pytest.mark.parametrize("role", ["reference", "validation_cache"])
def test_resuming_with_matching_fingerprints_reopens_the_table(tmp_path, role):
    EmbeddingTable.create(tmp_path / "db", "table", role, 2, snapshot="kept", **fingerprints(role))

    reopened = EmbeddingTable.open_or_create(
        tmp_path / "db", "table", role, lambda: 3, snapshot="new", **fingerprints(role)
    )

    assert (reopened.dimension, reopened.creation_snapshot) == (2, "kept")


def test_opening_a_complete_table_requires_the_given_encoder_and_dimension(tmp_path):
    table = EmbeddingTable.create(
        tmp_path / "db", "table", "reference", 2, snapshot="", **fingerprints("reference")
    )
    table.mark_complete([])
    other = dict(SYNTHETIC_ENCODER, **{"embedding.model_id": "other"})

    def open_complete(**required):
        return EmbeddingTable.open_complete(
            tmp_path / "db", "table", "reference", expected=list, **required
        )

    with pytest.raises(IncompatibleTableError) as error:
        open_complete(encoder=other, dimension=3)
    assert error.value.differences == {
        "embedding.model_id": ("synthetic", "other"),
        "dimension": (2, 3),
    }
    with pytest.raises(IncompatibleTableError, match="dimension: stored 2, required 3"):
        open_complete(dimension=3)
    assert open_complete(encoder=SYNTHETIC_ENCODER, dimension=2).is_complete
    assert open_complete().is_complete


def test_require_compatible_compares_against_another_tables_values(tmp_path):
    reference = EmbeddingTable.create(
        tmp_path / "db", "reference", "reference", 2, snapshot="", **fingerprints("reference")
    )
    cache = EmbeddingTable.create(
        tmp_path / "db", "cache", "validation_cache", 2, snapshot="",
        encoder=dict(SYNTHETIC_ENCODER, **{"embedding.vertex.classification_prompt": True}),
    )

    reference.require_compatible(dimension=cache.dimension)
    with pytest.raises(IncompatibleTableError) as error:
        reference.require_compatible(encoder=cache.encoder_fingerprint, dimension=cache.dimension)

    assert error.value.differences == {"embedding.vertex.classification_prompt": (None, True)}


# Prefix tables


PREFIX_ROWS = [
    PointRecord("A", "class_name", "apple", "apple"),
    PointRecord("B", "training_image", "train/b.jpg", "train/b.jpg"),
]
PREFIX_VECTORS = np.array([[0.1, 0.2, 0.3, 0.4], [-1.5, 2.25, 1e-8, 7.0]], dtype=np.float32)


@pytest.fixture
def source(tmp_path):
    table = EmbeddingTable.create(tmp_path / "db", "source", "reference", 4, snapshot="snapshot", **fingerprints("reference"))
    table.append(PREFIX_ROWS, PREFIX_VECTORS)
    return table


def test_prefix_keeps_records_and_replaces_vectors_with_float32_prefixes(source):
    derived = source.derive_prefix("source_d2", 2, batch_size=1)

    reopened = EmbeddingTable.open(source.uri, "source_d2", "reference")
    for table in (derived, reopened):
        assert table.name == "source_d2"
        assert records_of(table) == PREFIX_ROWS
        np.testing.assert_array_equal(
            vectors_of(table), np.array([[0.1, 0.2], [-1.5, 2.25]], dtype=np.float32)
        )
        assert (table.role, table.dimension) == ("reference", 2)
        assert table.prefix_source == PrefixSource("source", 4)
    assert (source.dimension, source.prefix_source) == (4, None)


@pytest.mark.parametrize("snapshot", ['{"embedding": {"truncate_dim": 3}}', "{broken", ""])
def test_prefix_creation_snapshot_is_identical_to_its_source(tmp_path, snapshot):
    source = EmbeddingTable.create(tmp_path / "db", "source", "reference", 4, snapshot=snapshot, **fingerprints("reference"))
    source.append(PREFIX_ROWS, PREFIX_VECTORS)

    derived = source.derive_prefix("source_d2", 2, batch_size=1)
    second = derived.derive_prefix("source_d1", 1, batch_size=1)

    assert derived.creation_snapshot == snapshot
    assert EmbeddingTable.open(source.uri, "source_d2", "reference").creation_snapshot == snapshot
    assert second.creation_snapshot == snapshot
    assert second.prefix_source == PrefixSource("source_d2", 2)
    assert EmbeddingTable.open(source.uri, "source", "reference").creation_snapshot == snapshot


def test_prefix_tables_inherit_their_sources_fingerprints(source, tmp_path):
    cache = EmbeddingTable.create(
        tmp_path / "db", "cache", "validation_cache", 4, snapshot="",
        **fingerprints("validation_cache"),
    )

    reference_prefix = source.derive_prefix("source_d2", 2, batch_size=1)
    cache_prefix = cache.derive_prefix("cache_d2", 2, batch_size=1)

    for table in (reference_prefix, EmbeddingTable.open(source.uri, "source_d2", "reference")):
        assert table.encoder_fingerprint == source.encoder_fingerprint == SYNTHETIC_ENCODER
        assert table.inputs_fingerprint == source.inputs_fingerprint == SYNTHETIC_INPUTS
    assert cache_prefix.encoder_fingerprint == SYNTHETIC_ENCODER
    assert cache_prefix.inputs_fingerprint is None
    reference_prefix.require_compatible(
        encoder=cache_prefix.encoder_fingerprint, dimension=cache_prefix.dimension
    )


def test_prefix_of_a_validation_cache_is_a_validation_cache(tmp_path):
    cache = EmbeddingTable.create(tmp_path / "db", "cache", "validation_cache", 4, snapshot="", **fingerprints("validation_cache"))
    keys = [("validation_image", "b"), ("validation_image", "a")]
    cache.append([PointRecord("A", kind, identity, "") for kind, identity in keys], PREFIX_VECTORS)

    derived = cache.derive_prefix("cache_d3", 3, batch_size=1)

    assert EmbeddingTable.open(cache.uri, "cache_d3", "validation_cache").role == "validation_cache"
    np.testing.assert_array_equal(derived.fetch_vectors(keys[::-1]), PREFIX_VECTORS[::-1, :3])


def test_prefix_of_a_complete_source_is_complete(source):
    source.mark_complete(record.source_key for record in PREFIX_ROWS)

    source.derive_prefix("source_d2", 2, batch_size=1)

    assert EmbeddingTable.open(source.uri, "source_d2", "reference").is_complete
    assert source.is_complete


def test_prefix_of_an_incomplete_source_is_incomplete(source):
    source.derive_prefix("source_d2", 2, batch_size=1)

    assert not EmbeddingTable.open(source.uri, "source_d2", "reference").is_complete


def test_prefix_wider_than_source_fails_before_writing(source):
    with pytest.raises(EmbeddingTableError, match="exceeds the stored width 4"):
        source.derive_prefix("too_wide", 5, batch_size=1)

    assert not EmbeddingTable.exists(source.uri, "too_wide")


def test_full_width_prefix_copies_vectors_unchanged(source):
    derived = source.derive_prefix("copy", 4, batch_size=1)

    assert records_of(derived) == PREFIX_ROWS
    np.testing.assert_array_equal(vectors_of(derived), PREFIX_VECTORS)


def test_existing_prefix_is_reused_without_copying_again(source):
    first = source.derive_prefix("source_d2", 2, batch_size=1)
    source.append([PointRecord("C", "class_name", "late", "late")], np.ones((1, 4)))

    reused = source.derive_prefix("source_d2", 2, batch_size=1)

    assert records_of(reused) == records_of(first) == PREFIX_ROWS


def test_legacy_destination_is_refused_and_left_unchanged(source):
    legacy_table(source.uri, "source_d2", None)

    with pytest.raises(LegacyTableError):
        source.derive_prefix("source_d2", 2, batch_size=1)

    with pytest.raises(LegacyTableError):
        EmbeddingTable.open(source.uri, "source_d2", "reference")


def test_prefix_streams_every_row_without_a_result_limit(source):
    many = [PointRecord("A", "training_image", f"extra-{i}", f"extra-{i}") for i in range(23)]
    source.append(many, np.ones((23, 4), dtype=np.float32))

    derived = source.derive_prefix("many_d3", 3, batch_size=3)

    assert records_of(derived) == records_of(source)
    np.testing.assert_array_equal(vectors_of(derived), vectors_of(source)[:, :3])


@pytest.mark.parametrize(
    ("operation", "existing"), [("open_table", True), ("create_table", False)]
)
def test_storage_errors_propagate_without_cleanup(source, monkeypatch, operation, existing):
    if existing:
        source.derive_prefix("destination", 2, batch_size=1)
    drop = Mock()
    monkeypatch.setattr(LanceDBConnection, "drop_table", drop)
    monkeypatch.setattr(
        LanceDBConnection, operation, Mock(side_effect=RuntimeError("storage failure"))
    )

    with pytest.raises(RuntimeError, match="storage failure"):
        source.derive_prefix("destination", 2, batch_size=1)

    drop.assert_not_called()
