"""Evaluate cached embeddings at stored and shorter prefix dimensions."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classification.experiments.artifacts import write_json
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig
from classification.experiments.report_types import EvaluationReport
from classification.retrieval.embedding_table import EmbeddingTable
from classification.tracking import Tracker


def table_names(sources: tuple[str, str], dimension: int, source_width: int) -> tuple[str, str]:
    """Return derived reference/cache names, retaining names at the stored source width."""
    suffix = f"_d{dimension}" if dimension != source_width else ""
    reference, validation = sources
    return reference + suffix, validation + suffix


def validate_sweep(sources: tuple[str, str], dimensions: list[int], source_width: int) -> None:
    """Check target widths and that derived table names collide with no other sweep table."""
    if any(dimension > source_width for dimension in dimensions):
        raise ValueError("Sweep dimensions must not exceed the source vector width")
    all_names = list(sources)
    for dimension in dimensions:
        if dimension != source_width:
            all_names.extend(table_names(sources, dimension, source_width))
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
    validation_source = config.lancedb.validation_table
    # ExperimentConfig requires a usable validation table whenever sweep_dimensions is set.
    assert validation_source is not None
    sources = (config.lancedb.table, validation_source)
    validation_index = EmbeddingTable.open(
        config.lancedb.uri, validation_source, "validation_cache"
    )
    reference_index = EmbeddingTable.open(config.lancedb.uri, config.lancedb.table, "reference")
    source_width = reference_index.dimension
    if validation_index.dimension != source_width:
        raise ValueError(
            f"Source vector widths must match: reference has {source_width}, "
            f"validation cache has {validation_index.dimension}"
        )
    validate_sweep(sources, dimensions, source_width)
    output = Path(config.output_dir).expanduser().resolve() / datetime.now(UTC).strftime(
        "matroyshka_sweep_%Y%m%dT%H%M%S_%fZ"
    )
    results = []
    # TODO: review continues here
    for dimension in dimensions:
        reference, validation = table_names(sources, dimension, source_width)
        settings = config.model_dump()
        settings["lancedb"].update(table=reference, validation_table=validation)
        settings["cache_validation_embeddings"] = True
        settings["output_dir"] = str(output / f"d{dimension}")
        effective = ExperimentConfig.model_validate(settings)
        if dimension != source_width:
            pairs = ((reference_index, reference), (validation_index, validation))
            for source, destination in pairs:
                source.derive_prefix(
                    destination, dimension, batch_size=effective.embedding.batch_size
                )
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
