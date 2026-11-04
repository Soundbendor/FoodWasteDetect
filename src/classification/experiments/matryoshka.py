"""Evaluate cached Gemini embeddings at native and shorter prefix dimensions."""

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

DIMENSIONS = (768, 1536, 3072)


def table_names(config: ExperimentConfig, dimension: int) -> tuple[str, str]:
    """Return reference/cache names, leaving native-dimension names unchanged."""
    suffix = f"_d{dimension}" if dimension != 3072 else ""
    options = config.lancedb
    validation = options.validation_table
    if validation is None:
        raise ValueError("lancedb.validation_table is required for sweeps")
    return options.table + suffix, validation + suffix


def validate_config(config: ExperimentConfig) -> None:
    """Require Gemini source tables and distinct source/destination names."""
    if config.embedding.model_id != "google/ge-v2":
        raise ValueError("matroyshka-sweep requires google/ge-v2 source tables")
    names = [config.lancedb.table, config.lancedb.validation_table]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Reference and validation table names must be nonempty")
    all_names = [name for dimension in DIMENSIONS for name in table_names(config, dimension)]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Sweep source and destination table names must be distinct")


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
    validate_config(config)
    uri = Path(config.lancedb.uri).expanduser().resolve()
    if not uri.is_dir():
        raise FileNotFoundError(f"LanceDB database does not exist: {uri}")
    connection = LanceDBConnection(uri)
    sources = table_names(config, 3072)
    for name in sources:
        connection.open_table(name)
    output = Path(config.output_dir).expanduser().resolve() / datetime.now(UTC).strftime(
        "matroyshka_sweep_%Y%m%dT%H%M%S_%fZ"
    )
    results = []
    for dimension in DIMENSIONS:
        settings = config.model_dump()
        settings["vector_dimension"] = dimension
        reference, validation = table_names(config, dimension)
        settings["embedding"]["truncate_dim"] = dimension
        settings["lancedb"].update(table=reference, validation_table=validation)
        settings["cache_validation_embeddings"] = True
        settings["output_dir"] = str(output / f"d{dimension}")
        effective = ExperimentConfig.model_validate(settings)
        if dimension != 3072:
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
