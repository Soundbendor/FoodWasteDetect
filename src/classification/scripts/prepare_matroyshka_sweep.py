"""Read existing tables and prepare configs for separate CPU Slurm sweeps."""

import argparse
import json
import re
from pathlib import Path

import pyarrow as pa
from lancedb.db import LanceDBConnection

from classification.experiments.artifacts import write_json
from classification.experiments.config import ExperimentConfig

DIMENSIONS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]


def prepare(source: dict, output: Path) -> ExperimentConfig:
    """Validate table widths and provenance without loading models or vectors."""
    summary = json.loads(Path(source["index_summary"]).read_text())
    config = ExperimentConfig.model_validate(summary["config"])
    if config.embedding.model_id != source["model_id"]:
        raise ValueError("Index summary model does not match the selected model")
    uri = Path(summary["uri"]).expanduser().resolve()
    reference = summary["table"]
    validation = source["validation_table"]
    if not validation or validation == reference:
        raise ValueError(
            "No distinct validation cache selected; set validation_table in sources JSON"
        )
    for name in (reference, validation):
        if not (uri / f"{name}.lance").is_dir():
            raise ValueError(f"Missing cached table: {uri / name}")
    connection = LanceDBConnection(uri)
    for name in (reference, validation):
        table = connection.open_table(name)
        vector_type = table.schema.field("vector").type
        if (
            not pa.types.is_fixed_size_list(vector_type)
            or vector_type.list_size != source["max_dimension"]
        ):
            raise ValueError(
                f"{name}: expected full width {source['max_dimension']}, got {vector_type}"
            )
        if table.count_rows() == 0:
            raise ValueError(f"Empty table: {name}")
        snapshot = (table.schema.metadata or {}).get(b"classification.build")
        if snapshot:
            built = json.loads(snapshot)
            if built["embedding"]["model_id"] != source["model_id"]:
                raise ValueError(f"{name}: stored model differs from selected model")
            if (
                built["embedding"].get("vertex", {}).get("classification_prompt", False)
                != config.embedding.vertex.classification_prompt
            ):
                raise ValueError(f"{name}: stored Vertex prompt differs from reference config")
        else:
            print(f"NOTE {name}: no build metadata; relying on explicit source selection")
    config.lancedb.uri = str(uri)
    config.lancedb.table = reference
    config.lancedb.validation_table = validation
    config.embedding.device = "cpu"
    config.embedding.truncate_dim = None
    config.vector_dimension = source["max_dimension"]
    config.sweep_dimensions = [d for d in DIMENSIONS if d <= source["max_dimension"]]
    config.cache_validation_embeddings = True
    config.output_dir = str(output / source["name"])
    return ExperimentConfig.model_validate(config.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    sources = json.loads(Path(args.sources).read_text())
    names = [source["name"] for source in sources]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[a-z0-9_]+", name) for name in names
    ):
        parser.error("Source names must be unique lowercase letters, digits, or underscores")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for source in sources:
        try:
            config = prepare(source, output)
        except (ValueError, OSError, KeyError) as error:
            print(f"SKIP {source['name']}: {error}", flush=True)
            continue
        path = output / f"{source['name']}.json"
        write_json(path, config.model_dump(mode="json"))
        jobs.append(f"{source['name']}\t{path}\n")
        print(
            f"READY {source['name']}: {config.lancedb.table}, dimensions={config.sweep_dimensions}",
            flush=True,
        )
        print(
            f"  References: images={config.embed_images}, labels={config.embed_class_labels}, "
            f"descriptions={config.embed_descriptions}, cap={config.n_imgs_per_class}, k={config.k}"
        )
    (output / "jobs.tsv").write_text("".join(jobs))
    if not jobs:
        parser.exit(1, "No models have usable full-width reference and validation tables.\n")


if __name__ == "__main__":
    main()
