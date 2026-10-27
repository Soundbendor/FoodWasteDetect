"""Evaluate float32 prefixes of complete Gemini reference and validation tables."""

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
from .vertex_utils import cached_batches

DIMENSIONS = (768, 1536, 3072)


def table_names(config: dict, dimension: int) -> tuple[str, str]:
    suffix = f"_d{dimension}" if dimension != 3072 else ""
    options = config["lancedb"]
    return options["table"] + suffix, options["validation_table"] + suffix


def validate_config(config: dict) -> None:
    Experiment.validate_config(config)
    if config["embedding"].get("model_id") != "google/ge-v2":
        raise ValueError("matroyshka-sweep requires google/ge-v2 source tables")
    names = [config["lancedb"].get(key) for key in ("table", "validation_table")]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("Reference and validation table names must be nonempty")
    all_names = [name for dimension in DIMENSIONS for name in table_names(config, dimension)]
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


def run_sweep(config: dict, evaluate: Callable[[dict], tuple[dict, str]]) -> dict:
    """Prepare each dimension and publish a combined summary only after all succeed."""
    validate_config(config)
    uri = Path(config["lancedb"]["uri"]).expanduser().resolve()
    if not uri.is_dir():
        raise FileNotFoundError(f"LanceDB database does not exist: {uri}")
    connection = cast(LanceDBConnection, lancedb.connect(uri))
    sources = table_names(config, 3072)
    for name in sources:
        connection.open_table(name)
    output = Path(config["output_dir"]).expanduser().resolve() / datetime.now(UTC).strftime(
        "matroyshka_sweep_%Y%m%dT%H%M%S_%fZ"
    )
    results = []
    for dimension in DIMENSIONS:
        effective = deepcopy(config)
        reference, validation = table_names(config, dimension)
        effective["embedding"]["truncate_dim"] = dimension
        effective["lancedb"].update(table=reference, validation_table=validation)
        effective["cache_validation_embeddings"] = True
        effective["output_dir"] = str(output / f"d{dimension}")
        if dimension != 3072:
            for source, destination in zip(sources, (reference, validation), strict=True):
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
