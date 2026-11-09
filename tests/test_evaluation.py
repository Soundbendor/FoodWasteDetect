"""Evaluation contracts using supplied queries and synthetic embedding batches."""

from io import StringIO
from unittest.mock import Mock

import numpy as np
import pytest
from tqdm import tqdm

from classification.experiments import evaluation
from classification.schemas import Neighbor


def batch(labels):
    return (
        {
            "class": labels,
            "source_id": [f"source-{i}" for i in range(len(labels))],
            "source_split": ["val"] * len(labels),
            "fname": [f"{i}.png" for i in range(len(labels))],
        },
        np.arange(len(labels) * 2, dtype=np.float32).reshape(len(labels), 2),
    )


def test_query_depth_preserves_support_order_and_metadata():
    import json

    metadata, vectors = batch(["A", "B", "C"])
    metadata["extra"] = ["first", "second", "third"]
    snapshot = {key: list(values) for key, values in metadata.items()}
    ranked = [
        Neighbor(id="first", label="B", score=0.9),
        Neighbor(id="second", label="A", score=0.8),
        Neighbor(id="third", label="A", score=0.7),
    ]
    deep_ranked = [
        Neighbor(id=f"n{i}", label="B" if i == 4 else "A", score=0.9 - 0.1 * i)
        for i in range(5)
    ]
    query = Mock(side_effect=[ranked, deep_ranked, []])

    predictions = evaluation.evaluate_batches(
        [(metadata, vectors)], query=query, class_counts={"A": 7, "B": 1},
        k=2, expected_count=3,
    )

    assert metadata == snapshot
    json.dumps(predictions)
    assert [call.args[1] for call in query.call_args_list] == [7, 5, 5]
    for call, vector in zip(query.call_args_list, vectors, strict=True):
        np.testing.assert_array_equal(call.args[0], vector)
    assert [row["extra"] for row in predictions] == ["first", "second", "third"]
    assert predictions[0]["prediction"] == "B"
    assert predictions[0]["nearest"] == "B"
    assert predictions[0]["vote_fraction"] == 0.5
    assert predictions[0]["neighbors"] == ranked[:2]
    assert predictions[0]["retrieval_metrics"]["relevant_reference_count"] == 7
    assert predictions[0]["retrieval_metrics"]["average_precision_at_r"] == pytest.approx(1 / 6)
    assert predictions[1]["prediction"] == "A"
    assert predictions[1]["vote_fraction"] == 1.0
    assert predictions[1]["neighbors"] == deep_ranked[:2]
    assert predictions[1]["retrieval_metrics"]["top5_hit"] is True
    assert predictions[2]["prediction"] is None
    assert predictions[2]["vote_fraction"] == 0
    assert predictions[2]["retrieval_metrics"]["relevant_reference_count"] == 0


def test_metadata_conversion_preserves_input_and_extra_columns():
    metadata, _ = batch(["A"])
    metadata.update(image=[object()], extra=["retained"])
    rows = evaluation.extract_batch_metadata(metadata)
    assert "image" in metadata and "image" not in rows[0]
    assert rows[0]["extra"] == "retained"
    assert rows[0]["source_split"] == "val"


@pytest.mark.parametrize("mismatch", ["columns", "fewer_vectors", "more_vectors"])
def test_misaligned_batches_fail(mismatch):
    metadata, vectors = batch(["A", "B"])
    if mismatch == "columns":
        metadata["source_id"].pop()
    elif mismatch == "fewer_vectors":
        vectors = vectors[:1]
    else:
        vectors = np.concatenate((vectors, vectors[:1]))
    query = Mock(return_value=[])
    with pytest.raises(ValueError):
        evaluation.evaluate_batches(
            [(metadata, vectors)], query=query, class_counts={}, k=1, expected_count=2,
        )
    if mismatch == "columns":
        query.assert_not_called()


@pytest.mark.parametrize("expected_count,labels,message", [
    (0, [], "No evaluation images"),
    (2, ["A"], "Incomplete evaluation"),
    (1, ["A", "B"], "Incomplete evaluation"),
])
def test_empty_or_incomplete_evaluation_fails(expected_count, labels, message):
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate_batches(
            [batch(labels)], query=Mock(return_value=[]), class_counts={},
            k=1, expected_count=expected_count,
        )


@pytest.mark.parametrize("k", [0, -1, True, 1.0, "1"])
def test_invalid_k_fails_before_consuming_batches(k):
    batches = Mock()
    with pytest.raises(ValueError, match="k must be a positive integer"):
        evaluation.evaluate_batches(
            batches, query=Mock(), class_counts={}, k=k, expected_count=1,
        )


@pytest.mark.parametrize("failure", [RuntimeError("query failed"), KeyboardInterrupt()])
def test_query_failure_closes_progress_without_counting_partial_batch(monkeypatch, failure):
    bars = []

    def progress(**kwargs):
        bar = tqdm(file=StringIO(), **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(evaluation, "tqdm", progress)
    query = Mock(side_effect=[[], [], failure])
    with pytest.raises(type(failure)):
        evaluation.evaluate_batches(
            [batch(["A"]), batch(["B", "C"])], query=query,
            class_counts={}, k=1, expected_count=3,
        )
    assert len(bars) == 1
    assert bars[0].n == 1
    assert bars[0].disable
