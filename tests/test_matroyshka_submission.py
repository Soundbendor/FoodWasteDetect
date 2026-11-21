"""CPU-only preflight and fake Slurm submission checks."""

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from classification.retrieval.lancedb import LanceDBIndex
from classification.scripts import prepare_matroyshka_sweep as preparation


def write_tables(uri, width, validation_rows=2, metadata=None, prefix=""):
    """Create reference and validation tables with constant vectors."""
    names = {}
    for kind, rows in [("reference", 2), ("validation", validation_rows)]:
        name = names[kind] = f"{prefix}{kind}"
        index = LanceDBIndex.create(uri, name, width, metadata=metadata)
        index.table.add([
            {"id": f"{name}-{i}", "class": "A", "source_kind": kind, "source_identity": str(i),
             "value": "", "vector": [1.0] * width}
            for i in range(rows)
        ])
    return names


@pytest.fixture
def source(tmp_path, monkeypatch):
    uri = tmp_path / "db"
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
    dataset = Mock()
    dataset.validation_set.return_value = [0, 1]
    monkeypatch.setattr(preparation, "load_dataset", Mock(return_value=dataset))
    return dict(
        name="model",
        model_id="model",
        index_summary=str(index),
        validation_table="validation",
    ), uri


@pytest.mark.parametrize(
    "width,expected",
    [
        (768, [128, 256, 512, 768]),
        (1152, [128, 256, 512, 768, 1024, 1152]),
        (2048, [128, 256, 512, 768, 1024, 1536, 2048]),
        (3072, [128, 256, 512, 768, 1024, 1536, 2048, 3072]),
        (4096, [128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]),
    ],
)
def test_native_grid_preserves_experiment(source, tmp_path, width, expected):
    settings, uri = source
    write_tables(uri, width)
    config = preparation.prepare(settings, tmp_path / "output")
    assert config.sweep_dimensions == expected
    assert config.embedding.device == "cpu"
    assert config.embedding.truncate_dim is None
    assert config.embed_class_labels and config.n_imgs_per_class == 20
    assert config.k == 10 and config.cache_validation_embeddings


def test_missing_cache_and_wrong_width_rejected(source, tmp_path):
    settings, uri = source
    settings["validation_table"] = None
    with pytest.raises(ValueError, match="validation_table must be nonempty"):
        preparation.prepare(settings, tmp_path)
    settings["validation_table"] = "validation"
    with pytest.raises(FileNotFoundError):
        preparation.prepare(settings, tmp_path)
    write_tables(uri, 512)
    settings["max_dimension"] = 768
    with pytest.raises(ValueError, match="expected width 768, stored width 512"):
        preparation.prepare(settings, tmp_path)


def test_partial_validation_cache_rejected(source, tmp_path):
    settings, uri = source
    write_tables(uri, 128, validation_rows=1)
    with pytest.raises(ValueError, match="1 rows, validation split has 2"):
        preparation.prepare(settings, tmp_path)


def test_wrong_model_metadata_rejected(source, tmp_path):
    settings, uri = source
    snapshot = json.dumps({"embedding": {"model_id": "other"}})
    write_tables(uri, 768, metadata={"classification.build": snapshot})
    with pytest.raises(ValueError, match="stored model differs"):
        preparation.prepare(settings, tmp_path)


def test_unusable_and_shared_sources_skipped(source, tmp_path, capsys):
    settings, uri = source
    write_tables(uri, 128)
    write_tables(uri, 128, prefix="other_")
    sources = [
        settings,
        dict(settings, name="shared"),
        dict(settings, name="string_width", max_dimension="128"),
        dict(settings, name="other", validation_table="other_validation"),
    ]
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(sources))
    output = tmp_path / "output"
    preparation.main(["--sources", str(manifest), "--output-dir", str(output)])
    printed = capsys.readouterr().out
    assert "SKIP shared: ValueError: shares a source table with model" in printed
    assert "SKIP string_width: ValueError" in printed
    jobs = (output / "jobs.tsv").read_text().splitlines()
    # "other" reuses the reference table, so only the first source is ready.
    assert [line.split("\t")[0] for line in jobs] == ["model"]


def test_unnamed_source_is_a_usage_error(source, tmp_path):
    settings, _ = source
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps([{k: v for k, v in settings.items() if k != "name"}]))
    with pytest.raises(SystemExit, match="2"):
        preparation.main(["--sources", str(manifest), "--output-dir", str(tmp_path / "out")])


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
