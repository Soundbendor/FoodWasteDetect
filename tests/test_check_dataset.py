"""Filesystem acceptance uses filenames and metadata, never image decoding."""

import json
from unittest.mock import patch

import pytest

from classification.experiments.config import DatasetConfig
from classification.scripts.check_dataset import check, main


@pytest.fixture
def food(tmp_path):
    files = {
        "meta/classes.txt": "apple\n",
        "meta/train.txt": "apple/1\n",
        "meta/test.txt": "apple/2\n",
        "images/apple/1.jpg": "intentionally not an image",
        "images/apple/2.jpg": "intentionally not an image",
        "images/apple/unused.jpg": "extra split",
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return DatasetConfig(name="food101", root=str(tmp_path))


def test_inventory_does_not_decode_and_allows_extras(food):
    with patch.dict("classification.scripts.check_dataset.EXPECTED_CLASSES", food101=1):
        result = check(food)
    assert result["status"] == "passed"
    assert result["extra_files"] == 1
    assert result["referenced_files"] == 2


def test_missing_image_fails(food, tmp_path):
    (tmp_path / "images/apple/2.jpg").unlink()
    with patch.dict("classification.scripts.check_dataset.EXPECTED_CLASSES", food101=1):
        result = check(food)
    assert result["status"] == "failed"
    assert result["splits"]["validation"]["missing_count"] == 1


def test_incorrect_taxonomy_count_fails(food):
    assert check(food)["status"] == "failed"


def test_unknown_label_fails(food, tmp_path):
    (tmp_path / "meta/test.txt").write_text("pear/2\n")
    with pytest.raises(ValueError, match="missing from classes"):
        check(food)


def test_annotation_count_disagreement_fails(food):
    with (
        patch.dict("classification.scripts.check_dataset.EXPECTED_CLASSES", food101=1),
        patch(
            "classification.scripts.check_dataset.source_counts",
            return_value={"train": 9, "validation": 1},
        ),
    ):
        result = check(food)
    assert result["status"] == "failed"


def test_cli_writes_failure_report(tmp_path):
    config = tmp_path / "dataset.yml"
    config.write_text(f"dataset:\n  name: food101\n  root: {tmp_path / 'missing'}\n")
    report = tmp_path / "reports/check.json"
    with patch(
        "sys.argv",
        ["check_dataset", "--config-file", str(config), "--report-file", str(report)],
    ):
        assert main() == 1
    assert json.loads(report.read_text())["status"] == "failed"
