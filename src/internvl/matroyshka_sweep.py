"""Evaluate reference prefixes with optional reuse of cached validation vectors."""

from collections.abc import Callable, Iterator
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from lancedb.table import Table

from .experiment import Experiment, _write_json
from .retrieval import LanceDBIndex
from .tracking import Tracker
from .vertex_utils import VertexExperiment, cached_batches


def sweep_settings(config: dict) -> tuple[int, list[int]]:
    """Resolve Gemini defaults and validate explicitly configured sweep dimensions."""
    gemini = config["embedding"].get("model_id") == "google/ge-v2"
    options = config.get("dimension_sweep", {})
    if not isinstance(options, dict):
        raise ValueError("dimension_sweep must be a mapping with source_dim and dimensions")
    source_dim = options.get("source_dim", 3072 if gemini else None)
    dimensions = options.get("dimensions", [768, 1536, 3072] if gemini else None)
    if type(source_dim) is not int or source_dim <= 0:
        raise ValueError("dimension_sweep.source_dim must be a positive integer")
    if (
        not isinstance(dimensions, list) or not dimensions
        or any(type(d) is not int or d <= 0 or d > source_dim for d in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise ValueError("dimension_sweep.dimensions must be a nonempty list of unique positive integers <= source_dim")
    return source_dim, dimensions


def table_names(config: dict, dimension: int) -> tuple[str, str | None]:
    source_dim, _ = sweep_settings(config)
    suffix = f"_d{dimension}" if dimension != source_dim else ""
    options = config["lancedb"]
    validation = options.get("validation_table")
    return options["table"] + suffix, validation + suffix if validation is not None else None


def validate_config(config: dict) -> None:
    Experiment.validate_config(config)
    source_dim, dimensions = sweep_settings(config)
    options = config["lancedb"]
    names = [options.get("table")]
    if options.get("validation_table") is not None:
        names.append(options["validation_table"])
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Reference and validation table names must be nonempty")
    all_names = [
        name for dimension in dict.fromkeys([source_dim, *dimensions])
        for name in table_names(config, dimension) if name is not None
    ]
    if len(set(all_names)) != len(all_names):
        raise ValueError("Sweep source and destination table names must be distinct")


def prefix_batches(
    source: Table, schema: pa.Schema, dimension: int, batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Stream every row, retaining metadata and replacing only the vector column."""
    position = schema.get_field_index("vector")
    for batch in source.search().limit(None).to_batches(batch_size=batch_size):
        vectors = np.asarray(batch.column(position).to_pylist(), dtype=np.float32)
        arrays = list(batch.columns)
        arrays[position] = pa.array(vectors[:, :dimension].tolist(), type=schema.field(position).type)
        yield pa.RecordBatch.from_arrays(arrays, schema=schema)


def prepare_table(
    connection: LanceDBConnection, source_name: str, destination: str,
    dimension: int, batch_size: int,
) -> None:
    """Reuse readable destinations; otherwise create once from a streaming reader."""
    if destination in connection:
        connection.open_table(destination)
        return
    source = connection.open_table(source_name)
    schema = source.schema
    position = schema.get_field_index("vector")
    schema = schema.set(position, schema.field(position).with_type(pa.list_(pa.float32(), dimension)))
    reader = pa.RecordBatchReader.from_batches(
        schema, prefix_batches(source, schema, dimension, batch_size),
    )
    connection.create_table(destination, data=reader, schema=schema, mode="create", exist_ok=False)


def evaluate_cached(config: dict, tracker: Tracker) -> dict:
    experiment = Experiment(config, tracker=tracker)
    options = config["lancedb"]
    cache = LanceDBIndex(options["uri"], options["validation_table"])
    cache.open()
    return experiment.evaluate_from_batches(cached_batches(
        cache, experiment.dataset.iter_batches("val", experiment.batch_size, decode=False),
    ))


def evaluate_dimension(config: dict, tracker: Tracker) -> dict:
    """Read a supplied cache strictly, or encode validation images for this dimension."""
    if config["lancedb"].get("validation_table") is not None:
        return evaluate_cached(config, tracker)
    experiment_class = (
        VertexExperiment if config["embedding"].get("model_id") == "google/ge-v2" else Experiment
    )
    return experiment_class(config, tracker=tracker).evaluate()


def run_sweep(config: dict, evaluate: Callable[[dict], tuple[dict, str]]) -> dict:
    """Prepare each dimension and publish a combined summary only after all succeed."""
    validate_config(config)
    source_dim, dimensions = sweep_settings(config)
    uri = Path(config["lancedb"]["uri"]).expanduser().resolve()
    if not uri.is_dir():
        raise FileNotFoundError(f"LanceDB database does not exist: {uri}")
    connection = cast(LanceDBConnection, lancedb.connect(uri))
    sources = [name for name in table_names(config, source_dim) if name is not None]
    for name in sources:
        connection.open_table(name)
    output = Path(config["output_dir"]).expanduser().resolve() / datetime.now(UTC).strftime(
        "matroyshka_sweep_%Y%m%dT%H%M%S_%fZ"
    )
    results = []
    for dimension in dimensions:
        effective = deepcopy(config)
        effective["dimension_sweep"] = dict(source_dim=source_dim, dimensions=dimensions.copy())
        reference, validation = table_names(config, dimension)
        effective["embedding"]["truncate_dim"] = dimension
        effective["lancedb"].update(table=reference, validation_table=validation)
        effective["cache_validation_embeddings"] = validation is not None
        effective["output_dir"] = str(output / f"d{dimension}")
        if dimension != source_dim:
            destinations = [name for name in (reference, validation) if name is not None]
            for source, destination in zip(sources, destinations, strict=True):
                prepare_table(connection, source, destination, dimension,
                              effective["embedding"].get("batch_size", 32))
        report, run_id = evaluate(effective)
        results.append(dict(
            dimension=dimension, table=reference, validation_table=validation,
            output_dir=effective["output_dir"], wandb_run_id=run_id,
            **{key: value for key, value in report["metrics"].items() if key != "per_class"},
        ))
    summary = {"results": results}
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    return summary
