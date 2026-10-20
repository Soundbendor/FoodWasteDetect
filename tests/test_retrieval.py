"""Synthetic retrieval contracts, independent of experiments and model loading."""

import importlib.util
import itertools
import unittest
from contextlib import closing
from unittest.mock import Mock

if any(importlib.util.find_spec(name) is None for name in ("numpy", "qdrant_client")):
    raise unittest.SkipTest("requires NumPy and qdrant-client")

import numpy as np
from qdrant_client import QdrantClient

from internvl.retrieval import QdrantIndex, classify, metrics
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


class QdrantTests(unittest.TestCase):
    def test_length_mismatch_does_not_upsert(self):
        client = Mock(spec=QdrantClient)
        index = QdrantIndex(client, "fixture", 2)
        record = PointRecord("A", "class_name", "A", "A")
        for records, vectors in [([record], np.empty((0, 2))), ([], np.ones((1, 2)))]:
            with self.subTest(records=records):
                with self.assertRaisesRegex(
                    ValueError, "^Record/vector count mismatch$"
                ):
                    index.upsert(records, vectors)
                client.upsert.assert_not_called()

    def test_round_trip_and_collection_protection(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype), closing(QdrantClient(":memory:")) as client:
                index = QdrantIndex(client, "fixture", 2)
                index.create()
                records = [
                    PointRecord(label, "class_name", label, label)
                    for label in ("A", "B")
                ]
                vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
                index.upsert(records, vectors)
                with self.assertRaisesRegex(
                    ValueError, "^Refusing to overwrite an existing collection$"
                ):
                    index.create()
                stored = client.retrieve(
                    "fixture", ids=[r.id for r in records], with_vectors=True
                )
                by_id = {str(point.id): point for point in stored}
                for record, vector in zip(records, vectors, strict=True):
                    self.assertEqual(by_id[record.id].payload, record.to_payload())
                    np.testing.assert_allclose(by_id[record.id].vector, vector)
                neighbors = index.query(vectors[0], 10)
                self.assertEqual([n["label"] for n in neighbors], ["A", "B"])
                self.assertEqual(neighbors[0]["id"], records[0].id)
                self.assertAlmostEqual(neighbors[0]["score"], 1.0)
                self.assertEqual(classify(neighbors), ("A", 0.5))


if __name__ == "__main__":
    unittest.main()
