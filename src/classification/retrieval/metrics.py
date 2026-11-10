"""Classification and ranking metrics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from math import log2
from typing import Literal, TypedDict

from classification.schemas import Neighbor, Prediction, QueryRetrievalMetrics


class ClassMetrics(TypedDict):
    support: int
    correct: int
    recall: float | None
    precision: float
    f1: float


class RetrievalMetrics(TypedDict):
    evaluated: int
    majority_vote_accuracy: float | None
    nearest_neighbor_accuracy: float | None
    top5_hit_rate: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    mean_top1_cosine_similarity: float | None
    mean_support_cosine_similarity: float | None
    retrieval_precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    map_at_r: float
    per_class: dict[str, ClassMetrics]


def ranking_metrics(
    neighbors: Sequence[Neighbor],
    label: str,
    k: int,
    relevant_count: int,
) -> QueryRetrievalMetrics:
    """Score the ranked rows before discarding neighbors beyond the voting support."""
    relevance = [n["label"] == label for n in neighbors] if relevant_count else []
    support = relevance[:k]
    ideal = sum(1 / log2(i + 1) for i in range(1, min(k, relevant_count) + 1))
    dcg = sum(hit / log2(i + 1) for i, hit in enumerate(support, 1))
    hits = 0
    ap = 0.0
    for rank, hit in enumerate(relevance[:relevant_count], 1):
        hits += hit
        if hit:
            ap += hits / rank
    return {
        "top5_hit": any(relevance[:5]),
        "precision_at_k": sum(support) / len(support) if support else 0.0,
        "reciprocal_rank_at_k": next((1 / i for i, hit in enumerate(support, 1) if hit), 0.0),
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "average_precision_at_r": ap / relevant_count if relevant_count else 0.0,
        "relevant_reference_count": relevant_count,
    }


def metrics(predictions: Sequence[Prediction], classes: Iterable[str]) -> RetrievalMetrics:
    """Aggregate majority-vote classification and query-weighted retrieval scores."""
    evaluated = len(predictions)
    support = Counter(p["class"] for p in predictions)
    predicted = Counter(p["prediction"] for p in predictions if p["prediction"] is not None)
    hits = Counter(p["class"] for p in predictions if p["prediction"] == p["class"])
    correct = sum(hits.values())
    labels = support.keys() | predicted.keys()

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    def scores(label: str) -> tuple[float, float, float]:
        return (
            ratio(hits[label], predicted[label]),
            ratio(hits[label], support[label]),
            ratio(2 * hits[label], predicted[label] + support[label]),
        )

    label_scores = {label: scores(label) for label in labels}
    per_class: dict[str, ClassMetrics] = {}
    for label in classes:
        precision, recall, f1 = label_scores.get(label, (0.0, 0.0, 0.0))
        per_class[label] = dict(
            support=support[label],
            correct=hits[label],
            recall=recall if support[label] else None,
            precision=precision,
            f1=f1,
        )
    precision_macro = ratio(
        sum(precision for precision, _, _ in label_scores.values()), len(labels)
    )
    recall_macro = ratio(sum(recall for _, recall, _ in label_scores.values()), len(labels))
    f1_macro = ratio(sum(f1 for _, _, f1 in label_scores.values()), len(labels))
    with_neighbors = [p["neighbors"] for p in predictions if p["neighbors"]]

    def ranking_mean(
        key: Literal[
            "top5_hit",
            "precision_at_k",
            "reciprocal_rank_at_k",
            "ndcg_at_k",
            "average_precision_at_r",
        ],
    ) -> float:
        return ratio(sum(p["retrieval_metrics"][key] for p in predictions), evaluated)

    return {
        "evaluated": evaluated,
        "majority_vote_accuracy": correct / evaluated if evaluated else None,
        "nearest_neighbor_accuracy": (
            sum(p["nearest"] == p["class"] for p in predictions) / evaluated if evaluated else None
        ),
        "per_class": per_class,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": ratio(correct, sum(predicted.values())),
        "recall_micro": ratio(correct, evaluated),
        "f1_micro": ratio(2 * correct, sum(predicted.values()) + evaluated),
        "top5_hit_rate": ranking_mean("top5_hit"),
        "retrieval_precision_at_k": ranking_mean("precision_at_k"),
        "mrr_at_k": ranking_mean("reciprocal_rank_at_k"),
        "ndcg_at_k": ranking_mean("ndcg_at_k"),
        "map_at_r": ranking_mean("average_precision_at_r"),
        "mean_top1_cosine_similarity": (
            sum(n[0]["score"] for n in with_neighbors) / len(with_neighbors)
            if with_neighbors
            else None
        ),
        "mean_support_cosine_similarity": (
            sum(sum(n["score"] for n in neighbors) / len(neighbors) for neighbors in with_neighbors)
            / len(with_neighbors)
            if with_neighbors
            else None
        ),
    }
