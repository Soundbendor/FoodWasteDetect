"""Synthetic voting and metric contracts, independent of storage and model loading."""

import importlib.util
import itertools
import unittest

if importlib.util.find_spec("numpy") is None:
    raise unittest.SkipTest("requires NumPy")

from classification.retrieval.voting import classify
from classification.retrieval.metrics import metrics, ranking_metrics
from classification.schemas import Neighbor, Prediction


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


if __name__ == "__main__":
    unittest.main()


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
