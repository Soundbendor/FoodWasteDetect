import csv
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
from matplotlib import rc_context, rcParams
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.plotting.plot_image_count_sweep import CAPS, COUNTS, MODELS, read_reports, render
from scripts.plotting.plot_matroyshka_sweep import plot


def test_shuffled_grid_and_invalid_reports(tmp_path):
    truth = {"val1": "Food"}
    entries = []
    for model in MODELS:
        for cap in reversed(CAPS):
            path = tmp_path / f"{model.split('/')[-1]}-{cap}.json"
            report = {
                "config": {
                    "dataset": {"name": "foodx251"},
                    "k": 10,
                    "n_imgs_per_class": cap,
                    "embed_images": True,
                    "embed_class_labels": False,
                    "embed_descriptions": False,
                    "embedding": {
                        "model_id": model,
                        "vertex": {"classification_prompt": True},
                    },
                },
                "reference_source_counts": {"training_image": COUNTS[cap]},
                "predictions": [
                    {
                        "source_id": "val1",
                        "source_split": "val",
                        "class": "Food",
                        "prediction": "Food",
                    }
                ],
                "metrics": {"evaluated": 1, "majority_vote_accuracy": 1.0},
            }
            path.write_text(json.dumps(report))
            entries.append({"report": path.name, "model": model, "cap": cap})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    render(read_reports(manifest, truth), tmp_path, 1)
    with (tmp_path / "accuracy.csv").open() as handle:
        assert [r["cap"] for r in csv.DictReader(handle)] == [
            "1",
            "5",
            "10",
            "20",
            "50",
            "100",
            "All",
        ] * 3
    for ext in ("png", "pdf"):
        assert (tmp_path / f"accuracy.{ext}").stat().st_size > 0
    for invalid, message in [
        (entries[:-1], "Missing points"),
        (entries + entries[:1], "duplicate"),
    ]:
        manifest.write_text(json.dumps(invalid))
        with pytest.raises(ValueError, match=message):
            read_reports(manifest, truth)


def test_print_style_across_sweeps(tmp_path, monkeypatch):
    figures = {}

    def capture_figure(figure, path):
        figures[path.name] = figure

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    points = {(model, cap): (0.8, tmp_path / "report.json") for model in MODELS for cap in CAPS}
    rows = [
        {"model_id": model, "dimension": 128, "status": "done", "accuracy": 0.8}
        for model in [*MODELS, "example/unknown"]
    ]
    with rc_context({"axes.labelsize": 22}):
        render(points, tmp_path, 1)
        plot(rows, tmp_path / "first.svg")
        plot(rows[::-1], tmp_path / "reversed.svg")
        assert rcParams["axes.labelsize"] == 22

    styles = {}
    for name, figure in figures.items():
        axes = figure.axes[0]
        assert tuple(figure.get_size_inches()) == (6.5, 4)
        assert axes.get_ylabel() == "Majority-vote validation accuracy"
        assert axes.yaxis.label.get_fontsize() == 10
        assert axes.yaxis.get_major_formatter()(80) == "80%"
        assert not axes.spines["top"].get_visible()
        assert all(line.get_visible() for line in axes.get_ygridlines())
        assert not any(line.get_visible() for line in axes.get_xgridlines())
        assert not figure.legends[0].get_frame_on()
        styles[name] = {
            line.get_label(): (line.get_color(), line.get_marker()) for line in axes.lines
        }
    assert styles["accuracy.pdf"] == {
        "EmbeddingGemma2": ("#0072B2", "o"),
        "Gemini Embedding 2": ("#D55E00", "s"),
        "SigLIP2-giant": ("#009E73", "^"),
    }
    assert styles["accuracy.pdf"].items() <= styles["first.svg"].items()
    assert styles["first.svg"] == styles["reversed.svg"]
    assert figures["accuracy.pdf"].axes[0].get_ylim() == (0, 100)
