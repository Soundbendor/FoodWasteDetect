"""Synthetic retrieval contracts, independent of experiments and model loading."""

import importlib.util
import itertools
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

if any(importlib.util.find_spec(name) is None for name in ("numpy", "lancedb")):
    raise unittest.SkipTest("requires NumPy and LanceDB")

import numpy as np
import pyarrow as pa
import pytest

from classification.retrieval.lancedb import LanceDBIndex
from classification.retrieval.voting import classify
from classification.retrieval.metrics import metrics, ranking_metrics
from classification.schemas import Neighbor, PointRecord, Prediction


def prediction_row(
    label: str, prediction: str | None, nearest: str | None
) -> Prediction:
    return {
        "fname": "synthetic.png",
        "class": label,
        "source_id": "synthetic",
        "source_split": "val",
        "prediction": prediction,
        "nearest": nearest,
        "vote_fraction": 0.0 if prediction is None else 1.0,
        "neighbors": [],
        "retrieval_metrics": ranking_metrics([], label, 1, 0),
    }


class VotingTests(unittest.TestCase):
    def test_vote_priorities_and_order_independence(self):
        self.assertEqual(classify([]), (None, 0.0))
        cases = [
            ([("B", 0.9), ("A", 0.2), ("A", 0.1)], ("A", 2 / 3)),
            ([("B", 0.9), ("A", 0.8)], ("B", 0.5)),
            ([("B", 0.8), ("A", 0.8)], ("A", 0.5)),
            ([("A", -0.8), ("B", -0.2)], ("B", 0.5)),
            ([("B", -0.9), ("A", -0.8), ("B", -0.7), ("A", -0.6)], ("A", 0.5)),
            ([("A", -0.8)], ("A", 1.0)),
        ]
        for values, expected in cases:
            neighbors = [
                Neighbor(id=str(i), label=label, score=score)
                for i, (label, score) in enumerate(values)
            ]
            for reordered in itertools.permutations(neighbors):
                with self.subTest(neighbors=reordered):
                    self.assertEqual(classify(reordered), expected)


def concurrent_reader(uri, barrier, queue):
    index = LanceDBIndex.open(uri, "fixture")
    barrier.wait(timeout=30)
    results = [index.query(np.array([1., 0.]), 10) for _ in range(10)]
    queue.put(results)


class LanceDBTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.uri = Path(self.temp.name) / "db"

    def populate(self, dtype=np.float32, uri=None):
        self.index = LanceDBIndex.create(self.uri if uri is None else uri, "fixture", 2)
        records = [PointRecord(label, "class_name", label, label) for label in ("A", "B", "C")]
        vectors = np.array([[3., 0.], [1., 1.], [-2., 0.]], dtype=dtype)
        self.index.append(records[:2], vectors[:2])
        self.index.append(records[2:], vectors[2:])
        return records, vectors

    def test_length_mismatch_does_not_append(self):
        self.index = LanceDBIndex(self.uri, Mock())
        record = PointRecord("A", "class_name", "A", "A")
        for records, vectors in [([record], np.empty((0, 2))), ([], np.ones((1, 2)))]:
            with self.assertRaises(ValueError):
                self.index.append(records, vectors)
        self.index.table.add.assert_not_called()

    def test_round_trip_and_table_protection(self):
        for dtype in (np.float32, np.float64):
            records, vectors = self.populate(dtype, self.uri / dtype.__name__)
            with self.assertRaises(ValueError):
                LanceDBIndex.create(self.index.uri, "fixture", 2)
            reopened = LanceDBIndex.open(self.index.uri, "fixture")
            stored = reopened.table.to_arrow()
            self.assertEqual(stored.schema.field("vector").type, pa.list_(pa.float32(), 2))
            for record, vector, point in zip(records, vectors, stored.to_pylist(), strict=True):
                self.assertEqual({k: v for k, v in point.items() if k != "vector"}, record.to_payload())
                np.testing.assert_allclose(point["vector"], vector)
            neighbors = reopened.query(np.array([2., 0.], dtype=dtype), 10)
            self.assertEqual([n["label"] for n in neighbors], ["A", "B", "C"])
            np.testing.assert_allclose([n["score"] for n in neighbors], [1., 2**-0.5, -1.], atol=1e-6)
            self.assertEqual(neighbors[0]["id"], records[0].id)
            self.assertEqual(len(reopened.query(vectors[0], 1)), 1)

    def test_empty_and_bad_vectors(self):
        self.index = LanceDBIndex.create(self.uri, "fixture", 2)
        self.assertEqual(self.index.query(np.array([1., 0.]), 10), [])
        for vector in ([[1., 2., 3.]], [[float("nan"), 1.]]):
            with self.assertRaises((ValueError, pa.ArrowInvalid)):
                self.index.append([PointRecord("A", "class_name", "A", "A")], np.array(vector))
            self.assertEqual(self.index.table.count_rows(), 0)
        version = self.index.table.version
        self.index.append([], np.empty((0, 2), dtype=np.float32))
        self.assertEqual(self.index.table.version, version)

    def test_missing_storage_is_not_created(self):
        self.assertFalse(LanceDBIndex.exists(self.uri, "fixture"))
        with self.assertRaises(FileNotFoundError):
            LanceDBIndex.open(self.uri, "fixture")
        self.assertFalse(self.uri.exists())
        self.uri.mkdir()
        with self.assertRaises(ValueError):
            LanceDBIndex.open(self.uri, "fixture")
        self.assertEqual(list(self.uri.iterdir()), [])

    def test_path_resolution(self):
        self.assertEqual(LanceDBIndex("~/fixtures", Mock()).uri, Path.home() / "fixtures")
        self.assertEqual(LanceDBIndex("fixtures", Mock()).uri, Path.cwd() / "fixtures")

    def test_concurrent_readers_preserve_table(self):
        self.populate()
        before = self.index.table.to_arrow()
        version = self.index.table.version
        ctx = multiprocessing.get_context("spawn")
        barrier, queue = ctx.Barrier(3), ctx.Queue()
        processes = [ctx.Process(target=concurrent_reader, args=(str(self.uri), barrier, queue)) for _ in range(3)]
        try:
            for process in processes:
                process.start()
            for _ in processes:
                results = queue.get(timeout=45)
                for result in results:
                    self.assertEqual([n["label"] for n in result], ["A", "B", "C"])
                    np.testing.assert_allclose([n["score"] for n in result], [1., 2**-0.5, -1.], atol=1e-6)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
            queue.close()
        self.index = LanceDBIndex.open(self.uri, "fixture")
        self.assertEqual(self.index.table.version, version)
        self.assertEqual(self.index.table.to_arrow(), before)


if __name__ == "__main__":
    unittest.main()


def test_identity_scan_and_vector_fetch_preserve_source_and_order(tmp_path):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.schemas import PointRecord

    index = LanceDBIndex.create(tmp_path / "db", "vectors", 3)
    records = [PointRecord("Apple", kind, f"quoted'{i}", "value")
               for kind in ("training_image", "validation_image") for i in range(13)]
    vectors = np.asarray([[i + 1, i + 2, i + 3] for i in range(26)], dtype=np.float32)
    order = np.random.default_rng(12).permutation(26)
    index.append([records[i] for i in order], vectors[order])
    identities = [(record.source_kind, record.source_identity) for record in records]
    assert sorted(index.get_record_ids()) == sorted(identities)
    requested = list(reversed(range(26))) + [3, 0]
    np.testing.assert_array_equal(
        index.fetch_vectors([identities[i] for i in requested]), vectors[requested],
    )
    assert index.fetch_vectors([]).shape == (0, 3)


def test_classification_macro_micro_and_display_independence():
    rows = [prediction_row('A', 'A', 'A'), prediction_row('A', 'A', 'A'),
            prediction_row('A', 'B', 'B'), prediction_row('B', 'C', 'C'),
            prediction_row('B', None, 'B')]
    result = metrics(rows, iter(['Unused', 'A', 'B', 'C']))
    assert result['evaluated'] == 5
    assert result['majority_vote_accuracy'] == 0.4
    assert result['nearest_neighbor_accuracy'] == 0.6
    assert result['precision_macro'] == 1 / 3
    assert result['recall_macro'] == (2 / 3) / 3
    assert result['f1_macro'] == 0.8 / 3
    assert result['precision_micro'] == 0.5
    assert result['recall_micro'] == 0.4
    assert result['f1_micro'] == 4 / 9
    assert list(result['per_class']) == ['Unused', 'A', 'B', 'C']
    assert result['per_class']['C'] == dict(support=0, correct=0, recall=None, precision=0, f1=0)
    assert result['per_class']['Unused'] == dict(support=0, correct=0, recall=None, precision=0, f1=0)
    assert {k: v for k, v in result.items() if k != 'per_class'} == {
        k: v for k, v in metrics(rows, []).items() if k != 'per_class'}
    no_abstentions = metrics(rows[:-1], [])
    for key in ('precision_micro', 'recall_micro', 'f1_micro'):
        assert no_abstentions[key] == no_abstentions['majority_vote_accuracy']
    empty = metrics([], iter(['B', 'A']))
    assert empty['evaluated'] == 0
    assert empty['majority_vote_accuracy'] is None
    assert empty['nearest_neighbor_accuracy'] is None
    assert list(empty['per_class']) == ['B', 'A']
    assert empty['f1_macro'] == empty['map_at_r'] == 0
    assert empty['mean_top1_cosine_similarity'] is None


def test_ranking_hand_calculated_and_top5_boundary():
    import math

    import pytest

    def neighbors(labels):
        return [Neighbor(id=str(i), label=label, score=-i / 10) for i, label in enumerate(labels)]

    ranked = neighbors(['B', 'A', 'A', 'B', 'A', 'A', 'A', 'A'])
    result = ranking_metrics(ranked, 'A', 3, 6)
    assert result['top5_hit']
    assert result['precision_at_k'] == 2 / 3
    assert result['reciprocal_rank_at_k'] == 1 / 2
    assert result['ndcg_at_k'] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3) + 0.5))
    assert result['average_precision_at_r'] == pytest.approx((1/2 + 2/3 + 3/5 + 4/6) / 6)
    assert ranking_metrics(neighbors('BBBBAA'), 'A', 1, 2)['top5_hit']
    assert not ranking_metrics(neighbors('BBBBBA'), 'A', 1, 1)['top5_hit']
    for ranked, count in (([], 2), (neighbors('BB'), 0)):
        result = ranking_metrics(ranked, 'A', 10, count)
        assert all(value == 0 for key, value in result.items() if key != 'relevant_reference_count')
    assert ranking_metrics(neighbors('AB'), 'A', 10, 1)['precision_at_k'] == 0.5


def test_support_cosine_is_query_weighted_and_keeps_negative_scores():
    import pytest
    rows = [prediction_row('A', 'A', 'A') for _ in range(3)]
    rows[0]['neighbors'] = [Neighbor(id='1', label='A', score=-0.8)]
    rows[1]['neighbors'] = [Neighbor(id='2', label='A', score=0.6),
                             Neighbor(id='3', label='B', score=-0.2)]
    result = metrics(rows, [])
    assert result['mean_top1_cosine_similarity'] == pytest.approx(-0.1)
    assert result['mean_support_cosine_similarity'] == pytest.approx(-0.3)


def test_class_counts_stream_all_rows_and_reference_types(tmp_path):
    index = LanceDBIndex.create(tmp_path, 'counts', 2)
    assert index.class_counts() == {}
    records = [PointRecord('A' if i % 2 else 'B', kind, str(i), '')
               for kind in ('class_name', 'description', 'training_image') for i in range(1100)]
    index.append(records, np.ones((len(records), 2), dtype=np.float32))
    assert index.class_counts() == {'A': 1650, 'B': 1650}


def test_optimize_preserves_rows_and_reduces_fragments(tmp_path):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.schemas import PointRecord

    index = LanceDBIndex.create(tmp_path / 'db', 'references', 2)
    for i in range(32):
        index.append([PointRecord(str(i), 'class_name', str(i), str(i))],
                     np.asarray([[1.0, i + 1.0]], dtype=np.float32))
    before = index.table.to_arrow().to_pylist()
    fragments = index.table.stats()['fragment_stats']['num_fragments']
    neighbors = index.query(np.asarray([1.0, 1.0]), 5)
    index.optimize()
    assert index.table.stats()['fragment_stats']['num_fragments'] < fragments
    assert sorted(index.table.to_arrow().to_pylist(), key=lambda r: r['id']) == sorted(
        before, key=lambda r: r['id'])
    assert index.query(np.asarray([1.0, 1.0]), 5) == neighbors


def test_dimension_and_prefix_table_derivation(tmp_path):
    from config_helpers import config_data

    from classification.experiments.config import ExperimentConfig

    config = ExperimentConfig.model_validate(config_data())
    config.vector_dimension = 4
    index = LanceDBIndex.create(
        tmp_path / 'db', 'source', 4,
        metadata={'custom.key': 'preserved', 'classification.build': config.model_dump_json()},
    )
    assert index.dimension == 4
    source_meta = index.table.schema.metadata
    assert source_meta[b'custom.key'] == b'preserved'
    assert ExperimentConfig.model_validate_json(source_meta[b'classification.build']).vector_dimension == 4
    records = [PointRecord('A', 'class_name', 'A', 'A'), PointRecord('B', 'class_name', 'B', 'B')]
    vectors = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    index.append(records, vectors)

    derived = index.create_prefix_table('prefix_d2', 2)
    assert derived.dimension == 2
    derived_meta = derived.table.schema.metadata
    assert derived_meta[b'custom.key'] == b'preserved'
    assert derived_meta[b'classification.build'] == config.model_copy(
        update={'vector_dimension': 2}).model_dump_json().encode()
    assert ExperimentConfig.model_validate_json(derived_meta[b'classification.build']).vector_dimension == 2
    prefixes = vectors[:, :2]
    expected = prefixes / np.linalg.norm(prefixes, axis=1, keepdims=True)
    np.testing.assert_allclose(
        derived.fetch_vectors([('class_name', 'A'), ('class_name', 'B')]),
        expected,
        rtol=1e-6,
    )

    second = derived.create_prefix_table('prefix_d1', 1)
    assert second.dimension == 1
    assert ExperimentConfig.model_validate_json(
        second.table.schema.metadata[b'classification.build']
    ).vector_dimension == 1
    reused = derived.create_prefix_table('prefix_d1', 1)
    assert ExperimentConfig.model_validate_json(
        reused.table.schema.metadata[b'classification.build']
    ).vector_dimension == 1

    no_meta = LanceDBIndex.create(tmp_path / 'db', 'no_meta', 4)
    no_meta.append(records, vectors)
    derived_no_meta = no_meta.create_prefix_table('no_meta_d2', 2)
    assert not derived_no_meta.table.schema.metadata
