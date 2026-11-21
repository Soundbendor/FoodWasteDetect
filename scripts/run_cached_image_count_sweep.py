"""Evaluate per-class training image caps using existing full-width vector tables.

Run from the repository root in the classification Pixi environment.
Use a new output directory per sweep. Source databases are opened for reads only.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path

import pyarrow.compute as pc
from lancedb.db import LanceDBConnection
from lancedb.index import BTree

from classification.experiments.artifacts import write_json
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig
from classification.experiments.report_types import IndexSummary
from classification.retrieval.lancedb import LanceDBIndex
from classification.util import load_dataset


def load(path):
    return json.loads(path.read_text())


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


def validate_source(entry, config, reports, path):
    """Return (model, caps, replay) after checking the entry against its index config."""
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
    return model, caps, replay


def read_source(config, dataset, caps, destination):
    options = config.lancedb
    validation_table = options.require_validation_table()
    source = LanceDBIndex.open(options.uri, options.table)
    cache = LanceDBIndex.open(options.uri, validation_table)
    if {source.dimension, cache.dimension} != {config.vector_dimension}:
        raise ValueError("Source dimensions differ from expected full width")
    train = dataset.select_images(dataset.train_set(), n_imgs_per_class=max(caps))
    # Only read the largest needed training prefix, including on fragmented tables.
    predicate = pc.field("source_identity").isin(list(train["source_id"]))
    rows = checked_rows(source.table.to_lance().to_table(filter=predicate), train, "training_image")
    # Check validation identities without loading vectors; evaluation fetches by identity.
    keys = cache.table.to_lance().to_table(columns=["source_identity", "class", "source_kind"])
    checked_rows(keys, dataset.validation_set(), "validation_image")
    destination.mkdir(parents=True)
    copy = LanceDBConnection(destination).clone_table(validation_table, str(cache.table.uri))
    # Evaluation filters this table by identity once per query batch.
    copy.create_index("source_identity", config=BTree())
    return rows


def evaluate_cap(base, dataset, rows, cap, folder):
    selected = dataset.select_images(dataset.train_set(), n_imgs_per_class=cap)
    identities = list(selected["source_id"])
    subset = checked_rows(
        rows.filter(pc.field("source_identity").isin(identities)), selected, "training_image"
    )
    settings = base.model_dump()
    settings.update(output_dir=str(folder), n_imgs_per_class=cap)
    settings["lancedb"].update(
        table=f"{base.lancedb.table}_n{cap}", uri=str(folder.parent / "lancedb")
    )
    config = ExperimentConfig.model_validate(settings)
    folder.mkdir(parents=True)
    index = LanceDBIndex.create(
        config.lancedb.uri,
        config.lancedb.table,
        base.vector_dimension,
        metadata={"classification.build": config.model_dump_json()},
    )
    index.table.add(subset)
    summary = IndexSummary(
        table=config.lancedb.table,
        uri=config.lancedb.uri,
        source_counts={"training_image": len(subset)},
        config=config,
    )
    write_json(folder / "index.json", summary.model_dump(mode="json"))
    write_json(
        folder / "selection.json",
        {"source": base.lancedb.model_dump(), "reference_ids": identities},
    )
    with contextlib.closing(Experiment(config, dataset=dataset)) as experiment:
        experiment.evaluate_cached()
    (path,) = folder.glob("evaluation_*/report.json")
    return path


def verify_replay(folder, report, historical, selection):
    if load(folder / "selection.json")["reference_ids"] != load(selection)["reference_ids"]:
        raise ValueError("Cache replay selected different training IDs")

    def predictions(path):
        return {p["source_id"]: p["prediction"] for p in load(path)["predictions"]}

    if predictions(report) != predictions(historical):
        raise ValueError("Cache replay disagrees with historical predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load(args.manifest)
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
        config = ExperimentConfig.model_validate(load(path)["config"])
        model, caps, replay = validate_source(entry, config, reports, path)
        config.vector_dimension = entry["width"]
        dataset = load_dataset(config.dataset)
        model_dir = output / f"source_{number}"
        rows = read_source(config, dataset, caps, model_dir / "lancedb")
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
            {"model": model, "cap": cap, "report": os.path.relpath(path, output)}
            for (model, cap), path in reports.items()
        ],
    )


if __name__ == "__main__":
    main()
