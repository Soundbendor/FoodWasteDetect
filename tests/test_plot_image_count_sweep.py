import csv
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from plot_image_count_sweep import CAPS, COUNTS, MODELS, read_reports, render


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
