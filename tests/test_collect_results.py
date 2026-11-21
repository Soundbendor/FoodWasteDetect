"""Synthetic group directories; no models, datasets, databases, or network calls."""

import csv
import json
import runpy
from pathlib import Path

import pytest

script = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/collect_results.py"))
collect, main = script["collect"], script["main"]


def write_group(root, done, missing=(), **config):
    """Write a manifest for every (source, dimension) and records for the done ones."""
    settings = {
        "dataset": {"name": "foodx251", "root": "data/foodx"},
        "k": 10,
        "embed_images": True,
        "embed_class_labels": False,
        "embed_descriptions": False,
        "n_imgs_per_class": None,
    } | config
    (root / "results").mkdir(parents=True, exist_ok=True)
    runs = []
    for source, dimension in [*done, *missing]:
        run = f"{source}_d{dimension}"
        runs.append(dict(run=run, source=source, model_id=f"org/{source}", dimension=dimension))
        if (source, dimension) in done:
            record = dict(
                config=settings,
                metrics={"majority_vote_accuracy": 0.75, "evaluated": 100},
                report=f"/reports/{run}/report.json",
            )
            (root / "results" / f"{run}.json").write_text(json.dumps(record))
    (root / "manifest.json").write_text(json.dumps({"runs": runs}))


def test_table_has_one_row_per_expected_run(tmp_path, capsys):
    write_group(tmp_path, done=[("a", 128), ("a", 256), ("b", 128)], missing=[("b", 256)])
    with pytest.raises(SystemExit) as stopped:
        main([str(tmp_path)])
    assert stopped.value.code == 1
    main([str(tmp_path), "--allow-partial"])
    assert "Missing: b_d256" in capsys.readouterr().out
    with (tmp_path / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["run"] for row in rows] == ["a_d128", "a_d256", "b_d128", "b_d256"]
    assert [row["status"] for row in rows] == ["done", "done", "done", "missing"]
    assert rows[0]["majority_vote_accuracy"] == "0.75" and rows[0]["model_id"] == "org/a"
    assert rows[0]["report"] == "/reports/a_d128/report.json"
    assert rows[3]["majority_vote_accuracy"] == ""


def test_group_without_results_fails(tmp_path):
    write_group(tmp_path, done=[], missing=[("a", 128)])
    with pytest.raises(SystemExit) as stopped:
        main([str(tmp_path), "--allow-partial"])
    assert stopped.value.code == 1
    assert not (tmp_path / "results.csv").exists()


def test_differing_settings_rejected_unless_declared_as_an_axis(tmp_path):
    write_group(tmp_path, done=[("a", 128), ("b", 128)])
    path = tmp_path / "results/b_d128.json"
    record = json.loads(path.read_text())
    record["config"]["k"] = 5
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs"):
        collect(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for run, k in zip(manifest["runs"], [10, 5], strict=True):
        run["k"] = k
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert [row["status"] for row in collect(tmp_path)] == ["done", "done"]
