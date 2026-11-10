"""Majority voting over retrieved neighbors."""

from collections import Counter
from collections.abc import Sequence

from classification.schemas import Neighbor


def classify(neighbors: Sequence[Neighbor]) -> tuple[str | None, float]:
    """Vote by count, then best similarity, then class name."""
    if not neighbors:
        return None, 0.0
    votes: Counter[str] = Counter()
    best: dict[str, float] = {}
    for neighbor in neighbors:
        label, score = neighbor["label"], neighbor["score"]
        votes[label] += 1
        best[label] = max(best.get(label, score), score)
    winner = min(votes, key=lambda label: (-votes[label], -best[label], label))
    return winner, votes[winner] / len(neighbors)
