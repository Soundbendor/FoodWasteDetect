"""Synthetic retrieval contracts, independent of experiments and model loading."""

import importlib.util
import itertools
import unittest
import multiprocessing
from pathlib import Path
import tempfile
from unittest.mock import Mock

if any(importlib.util.find_spec(name) is None for name in ("numpy", "lancedb")):
    raise unittest.SkipTest("requires NumPy and LanceDB")

import numpy as np
import lancedb
import pyarrow as pa

from internvl.retrieval import LanceDBIndex, classify, metrics
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
    }


class VotingTests(unittest.TestCase):
    def test_vote_and_denominator(self):
        self.assertEqual(classify([]), (None, 0.0))
        predictions: list[Prediction] = [
            prediction_row("A", "A" if i < 63 else None, "A")
            for i in range(90)
        ]
        self.assertEqual(
            metrics(predictions, ["A", "B"]),
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
            metrics([], iter(["B", "A"])),
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
        result = metrics(predictions, iter(["D", "A"]))
        self.assertEqual(result, expected)
        self.assertEqual(list(result["per_class"]), ["D", "A"])
        expected["per_class"] = {}
        self.assertEqual(metrics(predictions, []), expected)


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
