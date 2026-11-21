"""Read existing tables and prepare configs for separate CPU Slurm sweeps."""

import argparse
import json
import re
from pathlib import Path

from classification.experiments.artifacts import write_json
from classification.experiments.config import ExperimentConfig
from classification.experiments.matroyshka import check_validation_rows, open_sources
from classification.util import load_dataset

DIMENSIONS = [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]


def prepare(source: dict, output: Path) -> ExperimentConfig:
    """Validate the cached tables and return a sweep config at their stored width."""
    summary = json.loads(Path(source["index_summary"]).read_text())
    config = ExperimentConfig.model_validate(summary["config"])
    if config.embedding.model_id != source["model_id"]:
        raise ValueError("Index summary model does not match the selected model")
    config.lancedb.uri = str(Path(summary["uri"]).expanduser().resolve())
    config.lancedb.table = summary["table"]
    config.lancedb.validation_table = source.get("validation_table")
    reference, validation = open_sources(config)
    width = reference.dimension
    expected = source.get("max_dimension")
    if expected is not None and expected != width:
        raise ValueError(f"{reference.name}: expected width {expected}, stored width {width}")
    check_validation_rows(validation, load_dataset(config.dataset))
    config.embedding.device = "cpu"
    config.embedding.truncate_dim = None
    config.sweep_dimensions = [d for d in DIMENSIONS if d < width] + [width]
    config.cache_validation_embeddings = True
    config.output_dir = str(output / source["name"])
    return ExperimentConfig.model_validate(config.model_dump())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    sources = json.loads(Path(args.sources).read_text())
    names = [source.get("name") for source in sources]
    if len(set(names)) != len(names) or not all(
        isinstance(name, str) and re.fullmatch(r"[a-z0-9_]+", name) for name in names
    ):
        parser.error("Source names must be unique lowercase letters, digits, or underscores")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = []
    # Concurrent jobs sharing a source table would race on its derived prefix tables.
    owners: dict[tuple[str, str], str] = {}
    for source in sources:
        name = source["name"]
        try:
            config = prepare(source, output)
            tables = [
                (config.lancedb.uri, table)
                for table in (config.lancedb.table, config.lancedb.require_validation_table())
            ]
            if shared := [owners[key] for key in tables if key in owners]:
                raise ValueError(f"shares a source table with {shared[0]}")
        except Exception as error:
            print(f"SKIP {name}: {type(error).__name__}: {error}", flush=True)
            continue
        owners.update(dict.fromkeys(tables, name))
        path = output / f"{name}.json"
        write_json(path, config.model_dump(mode="json"))
        jobs.append(f"{name}\t{path}\n")
        print(
            f"READY {name}: {config.lancedb.table}, dimensions={config.sweep_dimensions}",
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
