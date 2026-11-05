"""Evaluate cached embeddings at stored and shorter prefix dimensions."""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from lancedb.table import Table

from classification.experiments.artifacts import write_json
from classification.experiments.batches import cached_batches
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig
from classification.experiments.report_types import EvaluationReport
from classification.retrieval.lancedb import LanceDBIndex
from classification.tracking import Tracker


def table_names(config: ExperimentConfig, dimension: int, source_width: int) -> tuple[str, str]:
    """Return reference/cache names, retaining names at the stored source width."""
    suffix = f"_d{dimension}" if dimension != source_width else ""
    options = config.lancedb
    validation = options.validation_table
    if validation is None:
        raise ValueError("lancedb.validation_table is required for sweeps")
    return options.table + suffix, validation + suffix


def validate_config(config: ExperimentConfig, dimensions: list[int], source_width: int) -> None:
    """Check target widths and distinct source and destination names."""
    names = [config.lancedb.table, config.lancedb.validation_table]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Reference and validation table names must be nonempty")
    if any(dimension > source_width for dimension in dimensions):
        raise ValueError("Sweep dimensions must not exceed the source vector width")
    all_names = list(names)
    for dimension in dimensions:
        if dimension != source_width:
            all_names.extend(table_names(config, dimension, source_width))
    if len(set(all_names)) != len(all_names):
        raise ValueError("Sweep source and destination table names must be distinct")


def vector_width(table: Table) -> int:
    """Read the positive fixed-size vector width from a source schema."""
    position = table.schema.get_field_index("vector")
    if position < 0:
        raise ValueError("Source table must have a fixed-size vector field")
    vector_type = table.schema.field(position).type
    if not pa.types.is_fixed_size_list(vector_type) or vector_type.list_size <= 0:
        raise ValueError("Source table must have a positive fixed-size vector field")
    return vector_type.list_size


def prefix_batches(
    source: Table,
    schema: pa.Schema,
    dimension: int,
    batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Stream every row, retaining metadata and replacing only the vector column."""
    position = schema.get_field_index("vector")
    for batch in source.search().limit(None).to_batches(batch_size=batch_size):
        vectors = np.asarray(batch.column(position).to_pylist(), dtype=np.float32)
        arrays = list(batch.columns)
        arrays[position] = pa.array(
            vectors[:, :dimension].tolist(), type=schema.field(position).type
        )
        yield pa.RecordBatch.from_arrays(arrays, schema=schema)


def prepare_table(
    connection: LanceDBConnection,
    source_name: str,
    destination: str,
    dimension: int,
    batch_size: int,
) -> None:
    """Reuse readable destinations; otherwise create once from a streaming reader."""
    if destination in connection:
        connection.open_table(destination)
        return
    source = connection.open_table(source_name)
    schema = source.schema
    position = schema.get_field_index("vector")
    metadata = dict(schema.metadata or {})
    encoded = metadata.get(b"classification.build")
    if encoded is not None:
        snapshot = ExperimentConfig.model_validate_json(encoded)
        snapshot.vector_dimension = dimension
        metadata[b"classification.build"] = snapshot.model_dump_json().encode("utf-8")
    schema = schema.set(
        position, schema.field(position).with_type(pa.list_(pa.float32(), dimension))
    ).with_metadata(metadata)
    reader = pa.RecordBatchReader.from_batches(
        schema,
        prefix_batches(source, schema, dimension, batch_size),
    )
    connection.create_table(destination, data=reader, schema=schema, mode="create", exist_ok=False)


def evaluate_cached(config: ExperimentConfig, tracker: Tracker) -> EvaluationReport:
    """Evaluate cached vectors in validation order and publish the complete report."""
    experiment = Experiment(config, tracker=tracker)
    options = config.lancedb
    assert options.validation_table is not None
    cache = LanceDBIndex.open(options.uri, options.validation_table)
    return experiment.evaluate_from_batches(
        cached_batches(
            cache,
            experiment.dataset.iter_batches("val", experiment.batch_size, decode=False),
        )
    )


def run_sweep(
    config: ExperimentConfig,
    evaluate: Callable[[ExperimentConfig], tuple[EvaluationReport, str]],
) -> dict[str, list[dict[str, Any]]]:
    """Prepare each dimension and publish a combined summary only after all succeed."""
    dimensions = config.sweep_dimensions
    # TODO: refactor overly defensive checks
    if dimensions is None:
        raise ValueError("sweep_dimensions is required for sweeps")
    uri = Path(config.lancedb.uri).expanduser().resolve()
    if not uri.is_dir():
        raise FileNotFoundError(f"LanceDB database does not exist: {uri}")
    connection = LanceDBConnection(uri)
    validation_source = config.lancedb.validation_table
    if not validation_source or not validation_source.strip():
        raise ValueError("Reference and validation table names must be nonempty")
    validation_vector_dim = vector_width(connection.open_table(validation_source))
    training_vector_dim = vector_width(connection.open_table(config.lancedb.table))
    if training_vector_dim != validation_vector_dim:
        raise ValueError("Source vector widths must match")
    source_width = training_vector_dim
    validate_config(config, dimensions, source_width)
    output = Path(config.output_dir).expanduser().resolve() / datetime.now(UTC).strftime(
        "matroyshka_sweep_%Y%m%dT%H%M%S_%fZ"
    )
    results = []
    # TODO: review continues here
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
                prepare_table(
                    connection,
                    source,
                    destination,
                    dimension,
                    effective.embedding.batch_size,
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
