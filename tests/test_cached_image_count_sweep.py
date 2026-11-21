import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cached_image_count_sweep as sweep
from datasets import Dataset
from run_cached_image_count_sweep import checked_rows, verify_replay

from classification.datasets.foodx251 import FoodX251ImageDataset
from classification.retrieval.lancedb import LanceDBIndex


def test_capped_selection_reorders_stored_rows_and_keeps_small_classes():
    dataset = Dataset.from_dict({"class": ["a", "a", "b", "a"], "source_id": ["1", "2", "3", "4"]})
    selected = FoodX251ImageDataset.select_images(dataset, n_imgs_per_class=2)
    table = pa.table(
        {
            "class": ["b", "a", "a"],
            "source_identity": ["3", "2", "1"],
            "source_kind": ["training_image"] * 3,
        }
    )
    assert checked_rows(table, selected, "training_image")["source_identity"].to_pylist() == [
        "1",
        "2",
        "3",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        checked_rows(table.take([1, 1, 2]), selected, "training_image")
    with pytest.raises(ValueError, match="Missing.*3"):
        checked_rows(table.slice(1), selected, "training_image")


def test_replay_checks_identities_and_predictions(tmp_path):
    selection = tmp_path / "selection.json"
    prior = tmp_path / "prior.json"
    report = tmp_path / "report.json"
    original = tmp_path / "historical.json"
    selection.write_text(json.dumps({"reference_ids": ["one"]}))
    prior.write_bytes(selection.read_bytes())
    report.write_text(json.dumps({"predictions": [{"source_id": "two", "prediction": "A"}]}))
    original.write_bytes(report.read_bytes())
    verify_replay(tmp_path, report, original, prior)
    prior.write_text(json.dumps({"reference_ids": ["other"]}))
    with pytest.raises(ValueError, match="training IDs"):
        verify_replay(tmp_path, report, original, prior)
    prior.write_bytes(selection.read_bytes())
    original.write_text(json.dumps({"predictions": [{"source_id": "two", "prediction": "B"}]}))
    with pytest.raises(ValueError, match="predictions"):
        verify_replay(tmp_path, report, original, prior)


def test_manifest_runs_on_cached_vectors(tmp_path, monkeypatch):
    annotations = tmp_path / "raw/annot"
    annotations.mkdir(parents=True)
    (annotations / "class_list.txt").write_text("0 apple\n")
    for split in ("train", "val"):
        (annotations / f"{split}_info.csv").write_text("fruit.jpg,0\n")
    uri = tmp_path / "cache"
    for split, table, kind in [
        ("train", "reference", "training_image"),
        ("val", "queries", "validation_image"),
    ]:
        identity = f"foodx251:{split}:fruit.jpg"
        index = LanceDBIndex.create(uri, table, 2)
        index.table.add(
            [
                {
                    "id": identity,
                    "class": "Apple",
                    "value": "fruit.jpg",
                    "source_kind": kind,
                    "source_identity": identity,
                    "vector": [1.0, 0.0],
                }
            ]
        )
    config = dict(
        dataset=dict(name="foodx251", root=str(annotations.parent)),
        embedding=dict(model_id="synthetic"),
        output_dir=str(tmp_path / "source"),
        lancedb=dict(uri=str(uri), table="reference", validation_table="queries"),
        k=1,
        embed_images=True,
        embed_class_labels=False,
        embed_descriptions=False,
    )
    (tmp_path / "index.json").write_text(json.dumps({"config": config}))
    manifest = tmp_path / "input.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {"model": "synthetic", "width": 2, "index_summary": "index.json", "caps": [1]}
                ]
            }
        )
    )
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv", ["sweep", "--manifest", str(manifest), "--output-dir", str(output)]
    )
    sweep.main()
    (point,) = json.loads((output / "manifest.json").read_text())
    result = json.loads((output / point["report"]).read_text())
    assert result["metrics"]["majority_vote_accuracy"] == 1
    assert result["reference_source_counts"] == {"training_image": 1}
