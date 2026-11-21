"""Synthetic report checks; no models, datasets, databases, or network calls."""

import csv
import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

script = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/plot_matroyshka_sweep.py")
)
read_points = script["read_points"]
plot = script["plot"]


def write_report(root, model, dimension, accuracy=0.75, **overrides):
    config = {
        "embedding": {"model_id": model, "truncate_dim": dimension},
        "vector_dimension": dimension,
        "dataset": {"name": "foodx251", "root": "data/foodx"},
        "k": 10,
        "embed_images": True,
        "embed_class_labels": False,
        "embed_descriptions": False,
        "n_imgs_per_class": None,
    }
    config.update(overrides)
    path = (
        root
        / model.replace("/", "_")
        / f"d{dimension}"
        / "evaluation_123"
        / "report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config": config,
                "metrics": {
                    "majority_vote_accuracy": accuracy,
                    "nearest_neighbor_accuracy": 0.25,
                    "evaluated": 100,
                },
            }
        )
    )
    return path


@pytest.fixture
def reports(tmp_path):
    for model, (_, width) in script["MODELS"].items():
        for dimension in reversed(script["DIMENSIONS"]):
            if dimension <= width:
                write_report(tmp_path, model, dimension)
    return tmp_path


def test_all_four_curves_and_csv_use_majority_vote(reports):
    points = read_points([reports])
    assert len(points) == 28
    assert all(point.accuracy == 0.75 for point in points)
    assert [p.dimension for p in points[:9]] == list(script["DIMENSIONS"])
    for model, (_, width) in script["MODELS"].items():
        assert max(p.dimension for p in points if p.model == model) == width
    output = reports / "plot.svg"
    plot(points, output)
    content = output.read_text()
    assert "Majority-vote validation accuracy (%)" in content
    assert all(label in content for label, _ in script["MODELS"].values())
    with output.with_suffix(".csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 28
    assert {r["majority_vote_accuracy"] for r in rows} == {"0.75"}


def test_partial_requires_explicit_opt_in(tmp_path):
    write_report(tmp_path, "google/ge-v2", 768)
    with pytest.raises(ValueError, match="Missing dimensions"):
        read_points([tmp_path])
    with pytest.warns(UserWarning, match="Missing dimensions"):
        assert len(read_points([tmp_path], allow_partial=True)) == 1


def test_duplicate_and_incompatible_reports_rejected(reports):
    extra = write_report(reports / "second_run", "google/ge-v2", 768)
    with pytest.raises(ValueError, match="Duplicate"):
        read_points([reports])
    extra.unlink()
    write_report(reports, "google/ge-v2", 768, k=5)
    with pytest.raises(ValueError, match="differs"):
        read_points([reports])


@pytest.mark.parametrize("accuracy", [None, True, -0.1, 1.1, float("nan")])
def test_invalid_accuracy_rejected(tmp_path, accuracy):
    write_report(tmp_path, "google/ge-v2", 768, accuracy)
    with pytest.raises(ValueError, match="Invalid majority_vote_accuracy"):
        read_points([tmp_path], allow_partial=True)
