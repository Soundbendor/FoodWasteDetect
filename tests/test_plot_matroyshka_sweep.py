"""Synthetic result tables; no models, datasets, databases, or network calls."""

import csv
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plotting.plot_matroyshka_sweep import plot, read_rows
from scripts.plotting.style import MODEL_STYLES


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
    models = [*MODEL_STYLES, "example/unlabeled"]
    results = write_results(
        tmp_path / "results.csv",
        [(model, dimension, 0.75) for model in models for dimension in (256, 128)],
    )
    rows = read_rows(results)
    assert {row["accuracy"] for row in rows} == {0.75}
    output = tmp_path / "plots/plot.svg"
    plot(rows, output)
    content = output.read_text()
    assert "Majority-vote validation accuracy" in content
    assert all(label in content for label, _, _ in MODEL_STYLES.values())
    assert "example/unlabeled" in content
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


@pytest.mark.parametrize(
    ("low", "high", "limits"),
    [(0.63, 0.93, (60, 95)), (0.75, 0.75, (70, 80)), (0, 0, (0, 5)), (1, 1, (95, 100))],
)
def test_zoom_and_missing_interior_dimension(tmp_path, monkeypatch, low, high, limits):
    results = write_results(
        tmp_path / "results.csv",
        [
            ("google/ge-v2", 512, high),
            ("google/ge-v2", 128, low),
            ("google/ge-v2", 256, None),
            ("example/missing", 128, None),
        ],
    )
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda figure, path: figures.append(figure))
    plot(read_rows(results), tmp_path / "plot.pdf")
    axes = figures[0].axes[0]
    (line,) = axes.lines
    assert list(line.get_xdata()) == [128, 256, 512]
    assert list(line.get_ydata()) == pytest.approx([100 * low, math.nan, 100 * high], nan_ok=True)
    assert axes.get_ylim() == limits
    assert axes.yaxis.get_major_formatter()(100 * high) == f"{100 * high:.0f}%"
