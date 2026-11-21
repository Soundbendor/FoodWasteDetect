"""Explicit metadata and filesystem acceptance checks to validate datasets downloaded from remote"""

import argparse
import os
from pathlib import Path

import yaml

from classification.datasets.util import DOWNLOAD_DIR
from classification.experiments.config import DatasetConfig
from classification.util import load_dataset

EXPECTED_CLASSES = {
    "foodx251": 251,
    "food101": 101,
    "uecfood256": 255,
    "vegfru": 292,
    "isia500": 500,
}
SPLITS = ("train", "validation")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MISSING_EXAMPLES = 5


def inventory(root: Path) -> set[str]:
    """Walk a resolved ``root`` once, excluding download staging, and collect image paths."""
    paths: set[str] = set()
    for directory, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d != DOWNLOAD_DIR]
        paths.update(
            os.path.join(directory, name)
            for name in files
            if os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES
        )
    return paths


def check(config: DatasetConfig) -> list[str]:
    """Return errors from comparing loader splits with recorded counts and the files on disk."""
    expected_classes = EXPECTED_CLASSES[config.name]
    root = Path(config.root).expanduser().resolve()
    files = inventory(root)
    dataset = load_dataset(config)
    classes = set(dataset.get_class_list())
    errors: list[str] = []
    if len(classes) != expected_classes:
        errors.append(f"Expected {expected_classes} classes, got {len(classes)}")
    referenced: set[str] = set()
    for split, table in zip(SPLITS, (dataset.train_set(), dataset.validation_set()), strict=True):
        # The loader already rejects unknown labels and duplicate source identities.
        columns = table.select_columns(["fname", "class"]).to_dict()
        paths = set(columns["fname"])
        missing = paths - files
        if not len(table):
            errors.append(f"{split}: no rows")
        if split == "train" and classes - set(columns["class"]):
            errors.append("Training references do not cover all classes")
        if missing:
            examples = ", ".join(sorted(missing)[:MISSING_EXAMPLES])
            errors.append(f"{split}: {len(missing)} missing image files, e.g. {examples}")
        if not referenced.isdisjoint(paths):
            errors.append("Train/evaluation paths overlap")
        referenced |= paths
    if config.name == "uecfood256" and (len(files), len(referenced)) != (31395, 27043):
        errors.append(
            "UEC inventory differs from recorded upstream counts (31395 raw, 27043 retained): "
            f"{len(files)} raw, {len(referenced)} retained"
        )
    return errors


def main() -> int:
    """Print errors for the configured dataset and return nonzero when the check fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = DatasetConfig.model_validate(
            yaml.safe_load(args.config_file.read_text())["dataset"]
        )
        errors = check(config)
    except Exception as exc:
        errors = [f"{type(exc).__name__}: {exc}"]
    for error in errors:
        print(error, flush=True)
    print(f"{args.config_file}: {'failed' if errors else 'passed'}", flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
