"""CPU-only preflight and fake Slurm submission checks."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pytest

from classification.scripts import prepare_matroyshka_sweep as preparation


@pytest.fixture
def source(tmp_path, monkeypatch):
    uri = tmp_path / "db"
    for name in ["reference", "validation"]:
        (uri / f"{name}.lance").mkdir(parents=True)
    config = dict(
        dataset=dict(name="foodx251", root=str(tmp_path / "raw")),
        embedding=dict(model_id="model"),
        lancedb=dict(uri=str(uri), table="reference"),
        output_dir=str(tmp_path / "old"),
        k=10,
        embed_images=True,
        embed_class_labels=True,
        embed_descriptions=False,
        n_imgs_per_class=20,
    )
    index = tmp_path / "index.json"
    index.write_text(json.dumps(dict(config=config, uri=str(uri), table="reference")))
    table = Mock()
    table.count_rows.return_value = 2
    connection = Mock()
    connection.open_table.return_value = table
    monkeypatch.setattr(preparation, "LanceDBConnection", Mock(return_value=connection))
    return dict(
        name="model",
        model_id="model",
        max_dimension=768,
        index_summary=str(index),
        validation_table="validation",
    ), table


@pytest.mark.parametrize(
    "width,expected",
    [
        (768, [128, 256, 512, 768]),
        (2048, [128, 256, 512, 768, 1024, 1536, 2048]),
        (3072, [128, 256, 512, 768, 1024, 1536, 2048, 3072]),
        (4096, [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]),
    ],
)
def test_native_grid_preserves_experiment(source, tmp_path, width, expected):
    settings, table = source
    settings["max_dimension"] = width
    table.schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), width))])
    config = preparation.prepare(settings, tmp_path / "output")
    assert config.sweep_dimensions == expected
    assert config.embedding.device == "cpu"
    assert config.embedding.truncate_dim is None
    assert config.embed_class_labels and config.n_imgs_per_class == 20
    assert config.k == 10 and config.cache_validation_embeddings


def test_missing_cache_and_wrong_width_rejected(source, tmp_path):
    settings, table = source
    settings["validation_table"] = None
    with pytest.raises(ValueError, match="No distinct validation"):
        preparation.prepare(settings, tmp_path)
    preparation.LanceDBConnection.assert_not_called()
    settings["validation_table"] = "validation"
    table.schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), 512))])
    with pytest.raises(ValueError, match="expected full width"):
        preparation.prepare(settings, tmp_path)


def test_wrong_model_metadata_rejected(source, tmp_path):
    settings, table = source
    table.schema = pa.schema(
        [pa.field("vector", pa.list_(pa.float32(), 768))],
        metadata={"classification.build": json.dumps({"embedding": {"model_id": "other"}})},
    )
    with pytest.raises(ValueError, match="stored model differs"):
        preparation.prepare(settings, tmp_path)


@pytest.mark.parametrize("dry_run", [False, True])
def test_submitter_uses_separate_cpu_wrapper_jobs(tmp_path, dry_run):
    root = Path(__file__).resolve().parents[1]
    binaries = tmp_path / "bin"
    binaries.mkdir()
    pixi = binaries / "pixi"
    pixi.write_text("""#!/usr/bin/env bash
set -eu
while [[ "$1" != --output-dir ]]; do shift; done
mkdir -p "$2"
printf 'qwen\\t/a path/qwen.json\\ngemini_v2\\t/a path/gemini.json\\n' > "$2/jobs.tsv"
""")
    sbatch = binaries / "sbatch"
    sbatch.write_text("""#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$TEST_CALLS"
echo 12345
""")
    for file in [pixi, sbatch]:
        file.chmod(0o755)
    calls = tmp_path / "calls"
    env = dict(
        os.environ,
        PATH=f"{binaries}:{os.environ['PATH']}",
        TEST_CALLS=str(calls),
        MATROYSHKA_OUTPUT_ROOT=str(tmp_path / "output"),
    )
    result = subprocess.run(
        [
            "bash",
            str(root / "scripts/submit_matroyshka_sweep.sh"),
            *(["--dry-run"] if dry_run else []),
            "cfg/matroyshka_sources.example.json",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Resolved configs" in result.stdout
    if dry_run:
        assert not calls.exists()
    else:
        recorded = calls.read_text()
        assert recorded.count("sbatch/run_matroyshka_sweep.sbatch") == 2
        assert "/a path/qwen.json" in recorded and "/a path/gemini.json" in recorded
