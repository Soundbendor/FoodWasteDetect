"""Evaluate cached embeddings at stored and shorter prefix dimensions."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classification.experiments.artifacts import write_json
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig
from classification.experiments.report_types import EvaluationReport
from classification.retrieval.lancedb import LanceDBIndex
from classification.tracking import Tracker


def table_names(config: ExperimentConfig, dimension: int, source_width: int) -> tuple[str, str]:
    """Return reference/cache names, retaining names at the stored source width."""
    suffix = f"_d{dimension}" if dimension != source_width else ""
    options = config.lancedb
    validation = options.require_validation_table()
    return options.table + suffix, validation + suffix


def validate_config(config: ExperimentConfig, dimensions: list[int], source_width: int) -> None:
    """Check target widths and distinct source and destination names."""
    validation = config.lancedb.require_validation_table()
    if any(dimension > source_width for dimension in dimensions):
        raise ValueError("Sweep dimensions must not exceed the source vector width")
    all_names = [config.lancedb.table, validation]
    for dimension in dimensions:
        if dimension != source_width:
            all_names.extend(table_names(config, dimension, source_width))
    if len(set(all_names)) != len(all_names):
        raise ValueError("Sweep source and destination table names must be distinct")


def evaluate_cached(config: ExperimentConfig, tracker: Tracker) -> EvaluationReport:
    """Evaluate cached vectors in validation order and publish the complete report."""
    return Experiment(config, tracker=tracker).evaluate_cached()


def run_sweep(
    config: ExperimentConfig,
    evaluate: Callable[[ExperimentConfig], tuple[EvaluationReport, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Prepare each dimension and publish a combined summary only after all succeed."""
    dimensions = config.sweep_dimensions
    if dimensions is None:
        raise ValueError("sweep_dimensions is required for sweeps")
    uri = config.lancedb.uri
    validation_source = config.lancedb.require_validation_table()
    reference_index = LanceDBIndex.open(uri, config.lancedb.table)
    validation_index = LanceDBIndex.open(uri, validation_source)
    if reference_index.dimension != validation_index.dimension:
        raise ValueError("Source vector widths must match")
    source_width = reference_index.dimension
    validate_config(config, dimensions, source_width)
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
        report, run_id = evaluate(effective)
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
