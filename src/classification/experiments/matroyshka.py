"""Evaluate cached embeddings at stored and shorter prefix dimensions."""

import contextlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classification.datasets.base import FoodImageDataset
from classification.experiments.artifacts import write_json
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig
from classification.experiments.report_types import EvaluationReport
from classification.retrieval.lancedb import LanceDBIndex
from classification.tracking import Tracker, tracked_run
from classification.util import load_dataset


def table_names(config: ExperimentConfig, dimension: int, source_width: int) -> tuple[str, str]:
    """Return reference/cache names, retaining names at the stored source width."""
    suffix = f"_d{dimension}" if dimension != source_width else ""
    options = config.lancedb
    validation = options.require_validation_table()
    return options.table + suffix, validation + suffix


def validate_config(config: ExperimentConfig, dimensions: list[int], source_width: int) -> None:
    """Check target widths and distinct source and destination names."""
    if any(dimension > source_width for dimension in dimensions):
        raise ValueError("Sweep dimensions must not exceed the source vector width")
    # table_names returns the source names at the source width.
    all_names = [
        name
        for dimension in {*dimensions, source_width}
        for name in table_names(config, dimension, source_width)
    ]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Sweep source and destination table names must be distinct")


def open_sources(config: ExperimentConfig) -> tuple[LanceDBIndex, LanceDBIndex]:
    """Open the reference and validation tables and check width and stored provenance."""
    uri = config.lancedb.uri
    validation_table = config.lancedb.require_validation_table()
    reference = LanceDBIndex.open(uri, config.lancedb.table)
    validation = LanceDBIndex.open(uri, validation_table)
    if reference.dimension != validation.dimension:
        raise ValueError("Source vector widths must match")
    if reference.table.count_rows() == 0:
        raise ValueError(f"Empty reference table: {reference.name}")
    embedding = config.embedding
    for index in (reference, validation):
        encoded = (index.table.schema.metadata or {}).get(b"classification.build")
        if encoded is None:
            logging.warning("%s: no build metadata; relying on explicit selection", index.name)
            continue
        built = json.loads(encoded)["embedding"]
        if built["model_id"] != embedding.model_id:
            raise ValueError(f"{index.name}: stored model differs from configured model")
        prompt = built.get("vertex", {}).get("classification_prompt", False)
        if prompt != embedding.vertex.classification_prompt:
            raise ValueError(f"{index.name}: stored Vertex prompt differs from config")
    return reference, validation


def check_validation_rows(validation: LanceDBIndex, dataset: FoodImageDataset) -> None:
    """Require one cached vector per validation image."""
    rows, expected = validation.table.count_rows(), len(dataset.validation_set())
    if rows != expected:
        raise ValueError(f"{validation.name}: {rows} rows, validation split has {expected}")


def _evaluate_cached(
    dataset: FoodImageDataset,
) -> Callable[[ExperimentConfig, Tracker], EvaluationReport]:
    """Evaluate one effective configuration against the shared dataset."""

    def execute(config: ExperimentConfig, tracker: Tracker) -> EvaluationReport:
        experiment = Experiment(config, tracker=tracker, dataset=dataset)
        with contextlib.closing(experiment):
            return experiment.evaluate_cached()

    return execute


def run_sweep(config: ExperimentConfig) -> dict[str, list[dict[str, Any]]]:
    """Prepare each dimension and publish a combined summary only after all succeed."""
    dimensions = config.sweep_dimensions
    if dimensions is None:
        raise ValueError("sweep_dimensions is required for sweeps")
    reference_index, validation_index = open_sources(config)
    source_width = reference_index.dimension
    validate_config(config, dimensions, source_width)
    dataset = load_dataset(config.dataset)
    check_validation_rows(validation_index, dataset)
    execute = _evaluate_cached(dataset)
    output = Path(config.output_dir).expanduser().resolve() / datetime.now(UTC).strftime(
        "matroyshka_sweep_%Y%m%dT%H%M%S_%fZ"
    )
    sources = (reference_index, validation_index)
    results = []
    for dimension in dimensions:
        settings = config.model_dump()
        settings["vector_dimension"] = dimension
        reference, validation = table_names(config, dimension, source_width)
        settings["embedding"]["truncate_dim"] = dimension
        settings["lancedb"].update(table=reference, validation_table=validation)
        settings["cache_validation_embeddings"] = True
        settings["output_dir"] = str(output / f"d{dimension}")
        effective = ExperimentConfig.model_validate(settings)
        if dimension != source_width:
            for source, destination in zip(sources, (reference, validation), strict=True):
                source.create_prefix_table(destination, dimension)
        report, run_id = tracked_run(
            effective, "matroyshka-sweep", execute, group=output.name, tags=[f"d{dimension}"]
        )
        results.append(
            dict(
                dimension=dimension,
                table=reference,
                validation_table=validation,
                output_dir=effective.output_dir,
                wandb_run_id=run_id,
                **{key: value for key, value in report.metrics.items() if key != "per_class"},
            )
        )
    summary = {"results": results}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    return summary
