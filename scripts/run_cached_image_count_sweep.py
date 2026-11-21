"""Evaluate per-class training image caps using existing full-width vector tables.

Run from the repository root in the classification Pixi environment.
Use a new output directory per sweep. Source databases are opened for reads only.
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
from lancedb.db import LanceDBConnection

from classification.experiments.artifacts import write_json
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig
from classification.retrieval.lancedb import LanceDBIndex
from classification.util import load_dataset


def checked_rows(table, split, kind):
    """Align cache rows to annotation order after checking identity and label."""
    locations = {}
    for i, (identity, label, source) in enumerate(
        zip(
            table["source_identity"].to_pylist(),
            table["class"].to_pylist(),
            table["source_kind"].to_pylist(),
            strict=True,
        )
    ):
        if source != kind or identity in locations:
            raise ValueError(f"Unexpected kind or duplicate {kind}: {identity}")
        locations[identity] = (i, label)
    indices = []
    for identity, label in zip(split["source_id"], split["class"], strict=True):
        if identity not in locations or locations[identity][1] != label:
            raise ValueError(f"Missing or mislabeled {kind}: {identity}")
        indices.append(locations[identity][0])
    if len(indices) != len(locations):
        raise ValueError(f"Unexpected extra rows in {kind} selection")
    return table.take(indices)


def read_source(config, width, dataset, caps, destination):
    options = config.lancedb
    source = LanceDBIndex.open(options.uri, options.table)
    cache = LanceDBIndex.open(options.uri, options.require_validation_table())
    if (source.dimension, cache.dimension) != (width, width):
        raise ValueError("Source dimensions differ from expected full width")
    train = dataset.select_images(dataset.train_set(), n_imgs_per_class=max(caps))
    # Only read the largest needed training prefix, including on fragmented tables.
    predicate = pc.field("source_identity").isin(list(train["source_id"]))
    rows = checked_rows(source.table.to_lance().to_table(filter=predicate), train, "training_image")
    validation = dataset.validation_set()
    queries = checked_rows(cache.table.to_lance().to_table(), validation, "validation_image")
    destination.mkdir(parents=True)
    LanceDBConnection(destination).create_table(options.require_validation_table(), data=queries)
    return rows


def evaluate_cap(base, dataset, rows, cap, folder):
    selected = dataset.select_images(dataset.train_set(), n_imgs_per_class=cap)
    identities = list(selected["source_id"])
    subset = checked_rows(
        rows.filter(pc.is_in(rows["source_identity"], value_set=pa.array(identities))),
        selected,
        "training_image",
    )
    config = base.model_copy(deep=True)
    config.output_dir = str(folder)
    config.n_imgs_per_class = cap
    config.lancedb.table = f"{base.lancedb.table}_n{cap}"
    config.lancedb.uri = str(folder.parent / "lancedb")
    folder.mkdir(parents=True)
    metadata = dict(subset.schema.metadata or {})
    metadata[b"classification.build"] = config.model_dump_json().encode()
    db = LanceDBConnection(config.lancedb.uri)
    db.create_table(config.lancedb.table, data=subset.replace_schema_metadata(metadata))
    write_json(
        folder / "index.json",
        {
            "table": config.lancedb.table,
            "uri": config.lancedb.uri,
            "source_counts": {"training_image": len(subset)},
            "config": config.model_dump(mode="json"),
        },
    )
    write_json(
        folder / "selection.json",
        {"source": base.lancedb.model_dump(), "reference_ids": identities},
    )
    experiment = Experiment(config, dataset=dataset)
    try:
        experiment.evaluate_cached()
    finally:
        experiment.close()
    (path,) = folder.glob("evaluation_*/report.json")
    return path


def verify_replay(folder, report, historical, selection):
    original = json.loads(selection.read_text())
    if (
        json.loads((folder / "selection.json").read_text())["reference_ids"]
        != original["reference_ids"]
    ):
        raise ValueError("Cache replay selected different training IDs")

    def predictions(path):
        return {
            p["source_id"]: p["prediction"] for p in json.loads(path.read_text())["predictions"]
        }

    if predictions(report) != predictions(historical):
        raise ValueError("Cache replay disagrees with historical predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.resolve().parent
    output = args.output_dir.resolve()
    reports = {}
    for entry in manifest.get("reports", []):
        key = (entry["model"], entry["cap"])
        path = (directory / entry["report"]).resolve(strict=True)
        if key in reports:
            raise ValueError(f"Duplicate report {key}")
        reports[key] = path

    sources = manifest["sources"]
    if not sources:
        raise ValueError("Specify at least one cache source")
    output.mkdir(parents=True, exist_ok=False)
    for number, entry in enumerate(sources, start=1):
        path = directory / entry["index_summary"]
        config = ExperimentConfig.model_validate(json.loads(path.read_text())["config"])
        model = config.embedding.model_id
        caps = entry["caps"]
        if (
            not caps
            or any(type(cap) is not int or cap < 1 for cap in caps)
            or len(set(caps)) != len(caps)
        ):
            raise ValueError(f"Expected distinct positive integer caps for {model}")
        width = entry["width"]
        if (
            model != entry["model"]
            or config.n_imgs_per_class is not None
            or not config.embed_images
            or config.embed_class_labels
            or config.embed_descriptions
            or config.embedding.truncate_dim not in (None, width)
            or config.vector_dimension not in (None, width)
        ):
            raise ValueError(f"Expected uncapped full-width image references in {path}")
        replay = entry.get("replay")
        if replay and (replay["cap"] not in caps or (model, replay["cap"]) not in reports):
            raise ValueError("Replay requires a selected cap and matching historical report")
        for cap in caps:
            if (model, cap) in reports and not (replay and cap == replay["cap"]):
                raise ValueError(f"Duplicate evaluation for {model} cap {cap}")
        config.vector_dimension = width
        dataset = load_dataset(config.dataset)
        model_dir = output / f"source_{number}"
        rows = read_source(config, width, dataset, caps, model_dir / "lancedb")
        for cap in caps:
            folder = model_dir / f"n{cap}"
            report = evaluate_cap(config, dataset, rows, cap, folder)
            if replay and cap == replay["cap"]:
                verify_replay(folder, report, reports[model, cap], directory / replay["selection"])
            else:
                reports[model, cap] = report
    write_json(
        output / "manifest.json",
        [
            {
                "model": model,
                "cap": cap,
                "report": os.path.relpath(path.resolve(), output),
            }
            for (model, cap), path in reports.items()
        ],
    )


if __name__ == "__main__":
    main()
