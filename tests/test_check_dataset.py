"""Filesystem acceptance uses filenames and metadata, never image decoding."""

from unittest.mock import patch

import pytest

from classification.experiments.config import DatasetConfig
from classification.scripts.check_dataset import EXPECTED_CLASSES, check, main


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


@pytest.fixture
def one_class():
    with patch.dict(EXPECTED_CLASSES, food101=1):
        yield


def test_inventory_does_not_decode_and_allows_extras(food, one_class):
    assert check(food) == []


def test_missing_image_fails(food, one_class, tmp_path):
    (tmp_path / "images/apple/2.jpg").unlink()
    (error,) = check(food)
    assert error.startswith("validation: 1 missing image files")


def test_incorrect_taxonomy_count_fails(food):
    assert check(food) == ["Expected 101 classes, got 1"]


def test_cli_reports_failure(tmp_path, capsys):
    config = tmp_path / "dataset.yml"
    config.write_text(f"dataset:\n  name: food101\n  root: {tmp_path / 'missing'}\n")
    with patch("sys.argv", ["check_dataset", "--config-file", str(config)]):
        assert main() == 1
    assert capsys.readouterr().out.endswith(f"{config}: failed\n")
