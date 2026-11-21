"""Synthetic result tables; no models, datasets, databases, or network calls."""

import csv
import runpy
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

script = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/plot_matroyshka_sweep.py")
)
read_rows, plot = script["read_rows"], script["plot"]


def write_results(path, rows):
    """Write (model, dimension, accuracy) rows; a None accuracy is a missing run."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "model_id", "dimension", "status", "majority_vote_accuracy"])
        for model, dimension, accuracy in rows:
            status = "missing" if accuracy is None else "done"
            writer.writerow([f"{model}_d{dimension}", model, dimension, status, accuracy])
    return path


def test_curves_come_from_the_table(tmp_path):
    models = [*script["LABELS"], "example/unlabeled"]
    results = write_results(
        tmp_path / "results.csv",
        [(model, dimension, 0.75) for model in models for dimension in (256, 128)],
    )
    rows = read_rows(results)
    assert {row["accuracy"] for row in rows} == {0.75}
    output = tmp_path / "plots/plot.svg"
    plot(rows, output)
    content = output.read_text()
    assert "Majority-vote validation accuracy (%)" in content
    assert all(label in content for label in [*script["LABELS"].values(), "example/unlabeled"])
    assert "partial results" not in content


def test_missing_runs_are_labeled_partial(tmp_path):
    results = write_results(
        tmp_path / "results.csv", [("google/ge-v2", 128, 0.7), ("google/ge-v2", 256, None)]
    )
    output = tmp_path / "plot.svg"
    plot(read_rows(results), output)
    assert "partial results" in output.read_text()


def test_table_without_completed_runs_rejected(tmp_path):
    results = write_results(tmp_path / "results.csv", [("google/ge-v2", 128, None)])
    with pytest.raises(ValueError, match="No completed runs"):
        read_rows(results)
