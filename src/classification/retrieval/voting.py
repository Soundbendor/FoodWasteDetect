"""Majority voting over retrieved neighbors."""

from collections import Counter
from collections.abc import Sequence

from classification.schemas import Neighbor


def classify(neighbors: Sequence[Neighbor]) -> tuple[str | None, float]:
    """Vote by count, then best similarity, then class name."""
    if not neighbors:
        return None, 0.0
    votes = Counter(n["label"] for n in neighbors)
    best = {label: max(n["score"] for n in neighbors if n["label"] == label) for label in votes}
    winner = min(votes, key=lambda label: (-votes[label], -best[label], label))
    return winner, votes[winner] / len(neighbors)


