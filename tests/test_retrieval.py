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

from internvl.retrieval import LanceDBIndex, classify, metrics, ranking_metrics
from internvl.schemas import Neighbor, PointRecord, Prediction


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


def legacy_metrics(predictions, classes):
    result = metrics(predictions, classes)
    return {key: ({label: {k: v for k, v in row.items() if k not in ("precision", "f1")}
                   for label, row in value.items()} if key == "per_class" else value)
            for key, value in result.items() if key in (
                "evaluated", "majority_vote_accuracy", "nearest_neighbor_accuracy", "per_class")}


class VotingTests(unittest.TestCase):
    def test_vote_and_denominator(self):
        self.assertEqual(classify([]), (None, 0.0))
        predictions: list[Prediction] = [
            prediction_row("A", "A" if i < 63 else None, "A")
            for i in range(90)
        ]
        self.assertEqual(
            legacy_metrics(predictions, ["A", "B"]),
            {
                "evaluated": 90,
                "majority_vote_accuracy": 0.7,
                "nearest_neighbor_accuracy": 1.0,
                "per_class": {
                    "A": {"support": 90, "correct": 63, "recall": 0.7},
                    "B": {"support": 0, "correct": 0, "recall": None},
                },
            },
        )

    def test_vote_priorities_and_order_independence(self):
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


class MetricsTests(unittest.TestCase):
    def test_empty_predictions(self):
        self.assertEqual(
            legacy_metrics([], iter(["B", "A"])),
            {
                "evaluated": 0,
                "majority_vote_accuracy": None,
                "nearest_neighbor_accuracy": None,
                "per_class": {
                    "B": {"support": 0, "correct": 0, "recall": None},
                    "A": {"support": 0, "correct": 0, "recall": None},
                },
            },
        )

    def test_abstentions_and_omitted_reporting_classes(self):
        predictions: list[Prediction] = [
            prediction_row("A", None, None),
            prediction_row("A", "A", "B"),
            prediction_row("B", "B", "B"),
            prediction_row("C", "A", "C"),
        ]
        expected = {
            "evaluated": 4,
            "majority_vote_accuracy": 0.5,
            "nearest_neighbor_accuracy": 0.5,
            "per_class": {
                "D": {"support": 0, "correct": 0, "recall": None},
                "A": {"support": 2, "correct": 1, "recall": 0.5},
            },
        }
        result = legacy_metrics(predictions, iter(["D", "A"]))
        self.assertEqual(result, expected)
        self.assertEqual(list(result["per_class"]), ["D", "A"])
        expected["per_class"] = {}
        self.assertEqual(legacy_metrics(predictions, []), expected)


def concurrent_reader(uri, barrier, queue):
    index = LanceDBIndex(uri, "fixture")
    index.open()
    barrier.wait(timeout=30)
    results = [index.query(np.array([1., 0.]), 10) for _ in range(10)]
    queue.put(results)


class LanceDBTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.uri = Path(self.temp.name) / "db"
        self.index = LanceDBIndex(self.uri, "fixture")

    def populate(self, dtype=np.float32):
        self.index.create(2)
        records = [PointRecord(label, "class_name", label, label) for label in ("A", "B", "C")]
        vectors = np.array([[3., 0.], [1., 1.], [-2., 0.]], dtype=dtype)
        self.index.append(records[:2], vectors[:2])
        self.index.append(records[2:], vectors[2:])
        return records, vectors

    def test_length_mismatch_does_not_append(self):
        self.index.table = Mock()
        record = PointRecord("A", "class_name", "A", "A")
        for records, vectors in [([record], np.empty((0, 2))), ([], np.ones((1, 2)))]:
            with self.assertRaisesRegex(ValueError, "Record/vector count mismatch"):
                self.index.append(records, vectors)
        self.index.table.add.assert_not_called()

    def test_round_trip_and_table_protection(self):
        for dtype in (np.float32, np.float64):
            self.index = LanceDBIndex(self.uri / dtype.__name__, "fixture")
            records, vectors = self.populate(dtype)
            with self.assertRaises(ValueError):
                self.index.create(2)
            reopened = LanceDBIndex(self.index.uri, "fixture")
            reopened.open()
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

    def test_full_v4_vectors_round_trip_with_cosine_search(self):
        self.index.create(2048)
        records = [PointRecord(label, "class_name", label, label) for label in ("A", "B")]
        vectors = np.zeros((2, 2048), dtype=np.float32)
        vectors[0, -1] = 3
        vectors[1, 0] = 2
        self.index.append(records, vectors)
        reopened = LanceDBIndex(self.uri, "fixture")
        reopened.open()
        self.assertEqual(
            reopened.table.schema.field("vector").type, pa.list_(pa.float32(), 2048)
        )
        neighbors = reopened.query(vectors[0] * 2, 10)
        self.assertEqual([n["id"] for n in neighbors], [r.id for r in records])
        np.testing.assert_allclose([n["score"] for n in neighbors], [1., 0.], atol=1e-6)
        self.assertEqual(classify(neighbors), ("A", 0.5))

    def test_empty_and_bad_vectors(self):
        self.index.create(2)
        self.assertEqual(self.index.query(np.array([1., 0.]), 10), [])
        for vector in ([[1., 2., 3.]], [[float("nan"), 1.]]):
            with self.assertRaises((ValueError, pa.ArrowInvalid)):
                self.index.append([PointRecord("A", "class_name", "A", "A")], np.array(vector))
            self.assertEqual(self.index.table.count_rows(), 0)

    def test_missing_storage_is_not_created(self):
        self.assertFalse(self.index.exists())
        with self.assertRaises(FileNotFoundError):
            self.index.open()
        self.assertFalse(self.uri.exists())
        self.uri.mkdir()
        with self.assertRaises(ValueError):
            self.index.open()
        self.assertEqual(list(self.uri.iterdir()), [])

    def test_path_resolution(self):
        self.assertEqual(LanceDBIndex("~/fixtures", "a").uri, Path.home() / "fixtures")
        self.assertEqual(LanceDBIndex("fixtures", "a").uri, Path.cwd() / "fixtures")

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
        self.index.open()
        self.assertEqual(self.index.table.version, version)
        self.assertEqual(self.index.table.to_arrow(), before)


if __name__ == "__main__":
    unittest.main()


def test_identity_scan_and_vector_fetch_preserve_source_and_order(tmp_path):
    from internvl.retrieval import LanceDBIndex
    from internvl.schemas import PointRecord

    index = LanceDBIndex(tmp_path / "db", "vectors")
    index.create(3)
    records = [PointRecord("Apple", kind, f"quoted'{i}", "value")
               for kind in ("training_image", "validation_image") for i in range(13)]
    vectors = np.asarray([[i + 1, i + 2, i + 3] for i in range(26)], dtype=np.float32)
    order = np.random.default_rng(12).permutation(26)
    index.append([records[i] for i in order], vectors[order])
    identities = [(record.source_kind, record.source_identity) for record in records]
    assert sorted(index.identities()) == sorted(identities)
    requested = list(reversed(range(26))) + [3, 0]
    np.testing.assert_array_equal(
        index.fetch_vectors([identities[i] for i in requested]), vectors[requested],
    )
    assert index.fetch_vectors([]).shape == (0, 3)


def test_classification_macro_micro_and_display_independence():
    rows = [prediction_row('A', 'A', 'A'), prediction_row('A', 'A', 'A'),
            prediction_row('A', 'B', 'B'), prediction_row('B', 'C', 'C'),
            prediction_row('B', None, None)]
    result = metrics(rows, ['A', 'B', 'C', 'Unused'])
    assert result['precision_macro'] == 1 / 3
    assert result['recall_macro'] == (2 / 3) / 3
    assert result['f1_macro'] == 0.8 / 3
    assert result['precision_micro'] == 0.5
    assert result['recall_micro'] == 0.4
    assert result['f1_micro'] == 4 / 9
    assert result['per_class']['C'] == dict(support=0, correct=0, recall=None, precision=0, f1=0)
    assert {k: v for k, v in result.items() if k != 'per_class'} == {
        k: v for k, v in metrics(rows, []).items() if k != 'per_class'}
    no_abstentions = metrics(rows[:-1], [])
    for key in ('precision_micro', 'recall_micro', 'f1_micro'):
        assert no_abstentions[key] == no_abstentions['majority_vote_accuracy']
    empty = metrics([], [])
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
    index = LanceDBIndex(tmp_path, 'counts')
    index.create(2)
    assert index.class_counts() == {}
    records = [PointRecord('A' if i % 2 else 'B', kind, str(i), '')
               for kind in ('class_name', 'description', 'training_image') for i in range(1100)]
    index.append(records, np.ones((len(records), 2), dtype=np.float32))
    assert index.class_counts() == {'A': 1650, 'B': 1650}
