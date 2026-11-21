"""Explicit metadata and filesystem acceptance checks; never decode images."""

import argparse
import csv
import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from classification.experiments.artifacts import write_json
from classification.experiments.config import DatasetConfig
from classification.util import load_dataset

EXPECTED_CLASSES = {
    "foodx251": 251,
    "food101": 101,
    "uecfood256": 255,
    "vegfru": 292,
    "isia500": 500,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inventory(root: Path) -> set[str]:
    """Walk once, excluding download staging, and collect regular image files."""
    paths = set()
    for directory, directories, files in os.walk(root):
        directories[:] = [d for d in directories if d != ".downloads"]
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                paths.add(str(path.resolve()))
    return paths


def source_counts(name: str, root: Path, files: set[str]) -> dict[str, int]:
    """Count raw annotations independently of loader tables."""
    if name == "uecfood256":
        paths = [
            Path(p) for p in files if Path(p).parent.parent == root and Path(p).suffix == ".jpg"
        ]
        frequencies = Counter(p.name for p in paths)
        per_class = Counter(p.parent.name for p in paths if frequencies[p.name] == 1)
        validation = sum(round(n * 0.2) for n in per_class.values())
        return {"train": sum(per_class.values()) - validation, "validation": validation}
    if name == "foodx251":
        result = {}
        for split, source in (("train", "train"), ("validation", "val")):
            with (root / f"annot/{source}_info.csv").open() as stream:
                result[split] = sum(bool(row) for row in csv.reader(stream))
        return result
    templates = {
        "food101": "meta/{}.txt",
        "vegfru": "vegfru_list/vegfru_{}.txt",
        "isia500": "metadata_ISIAFood_500/{}_finetune.txt",
    }
    return {
        split: sum(
            bool(line.strip())
            for line in (root / templates[name].format(source)).read_text().splitlines()
        )
        for split, source in (("train", "train"), ("validation", "test"))
    }


def check(config: DatasetConfig) -> dict[str, Any]:
    root = Path(config.root).expanduser().resolve()
    files = inventory(root)
    expected = source_counts(config.name, root, files)
    dataset = load_dataset(config)
    errors = []
    classes = dataset.get_class_list()
    if len(classes) != EXPECTED_CLASSES[config.name]:
        errors.append(f"Expected {EXPECTED_CLASSES[config.name]} classes, got {len(classes)}")
    referenced = set()
    split_reports = {}
    for split, table in (("train", dataset.train_set()), ("validation", dataset.validation_set())):
        paths = set(table["fname"])
        labels = set(table["class"])
        missing = paths - files
        source = "val" if config.name == "foodx251" else "test"
        if config.name == "uecfood256":
            source = "validation"
        source = "train" if split == "train" else source
        if not len(table) or len(table) != expected[split]:
            errors.append(f"{split}: {len(table)} rows, expected {expected[split]} nonempty rows")
        if labels - set(classes):
            errors.append(f"{split}: unknown labels")
        if split == "train" and set(classes) - labels:
            errors.append("Training references do not cover all classes")
        if len(set(table["source_id"])) != len(table):
            errors.append(f"{split}: duplicate identities")
        if set(table["source_split"]) != {source}:
            errors.append(f"{split}: incorrect source split")
        if missing:
            errors.append(f"{split}: {len(missing)} missing image files")
        if referenced & paths:
            errors.append("Train/evaluation paths overlap")
        referenced.update(paths)
        split_reports[split] = {
            "identity_digest": hashlib.sha256("\n".join(table["source_id"]).encode()).hexdigest(),
            "rows": len(table),
            "annotation_rows": expected[split],
            "classes": len(labels),
            "missing_count": len(missing),
            "missing_examples": sorted(missing)[:20],
        }
    if config.name == "uecfood256" and (len(files) != 31395 or len(referenced) != 27043):
        errors.append(
            "UEC inventory differs from recorded upstream counts (31395 raw, 27043 retained)"
        )
    return {
        "dataset": config.name,
        "root": str(root),
        "classes": len(classes),
        "image_files": len(files),
        "inventory_digest": hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest(),
        "referenced_files": len(referenced),
        "extra_files": len(files - referenced),
        "splits": split_reports,
        "errors": errors,
        "status": "failed" if errors else "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report: dict[str, Any]
    try:
        config = DatasetConfig.model_validate(
            yaml.safe_load(args.config_file.read_text())["dataset"]
        )
        report = check(config)
    except Exception as exc:
        report = {"status": "failed", "errors": [f"{type(exc).__name__}: {exc}"]}
    report["elapsed_seconds"] = time.monotonic() - started
    report["config_file"] = str(args.config_file.resolve())
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report_file, report)
    print(f"{args.config_file}: {report['status']} ({args.report_file})", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
