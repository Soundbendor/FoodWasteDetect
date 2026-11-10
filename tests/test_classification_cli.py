"""CPU checks for schema-defined CLI overrides and dispatch."""

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, call, mock_open, patch

import pytest
import yaml
from config_helpers import config_data

from classification.cli import _load_config, main
from classification.experiments.config import ExperimentConfig, OllamaConfig


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config_data()))
    return path


@pytest.mark.parametrize(
    "command",
    [[], ["experiment"], ["experiment", "matroyshka-sweep"], ["download_model"]],
)
def test_help_before_loading(command):
    with patch("builtins.open", side_effect=AssertionError("file load")), pytest.raises(SystemExit) as error:
        main([*command, "--help"])
    assert error.value.code == 0


@pytest.mark.parametrize("stage,method", [
    ("describe", "generate_descriptions"), ("index", "build_vector_database"),
    ("evaluate", "evaluate"), ("run", "run"),
])
def test_typed_dispatch_before_tracking(config_path, stage, method):
    content = config_path.read_bytes()
    with patch("classification.cli.Experiment") as experiment, patch("classification.cli.wandb.init") as initialize, patch("classification.cli.Tracker"):
        def track(**kwargs):
            assert kwargs["config"]["embedding"]["batch_size"] == 16
            return MagicMock()
        initialize.side_effect = track
        main(["experiment", "--set", "embedding.batch_size=16", stage, "--config_file", str(config_path)])
    config = experiment.call_args.args[0]
    assert isinstance(config, ExperimentConfig)
    assert config.embedding.batch_size == 16
    experiment.assert_called_once_with(config, tracker=ANY)
    assert experiment.return_value.method_calls == [getattr(call, method)(), call.close()]
    assert config_path.read_bytes() == content


def test_overrides_defaults_order_and_mapping_merges(config_path):
    result = _load_config(str(config_path), [
        "embedding.vertex=null", "embedding.vertex.max_attempts=2",
        "embedding={batch_size: 16, vertex: {project: test}}",
        "embedding.vertex={classification_prompt: true}",
        "k=3", "k=7", 'output_dir="a=b"', "n_imgs_per_class=2",
    ])
    assert result.k == 7 and result.output_dir == "a=b"
    assert result.n_imgs_per_class == 2
    assert result.embedding.batch_size == 16
    assert result.embedding.vertex.project == "test"
    assert result.embedding.vertex.max_attempts == 2
    assert result.embedding.vertex.classification_prompt is True


@pytest.mark.parametrize("assignment,field,value", [
    ('output_dir="001"', "output_dir", "001"),
    ("output_dir=path with spaces", "output_dir", "path with spaces"),
    ("output_dir=a=b=c", "output_dir", "a=b=c"),
    ("cache_validation_embeddings=false", "cache_validation_embeddings", False),
    ("n_imgs_per_class=null", "n_imgs_per_class", None),
])
def test_yaml_values(config_path, assignment, field, value):
    assert getattr(_load_config(str(config_path), [assignment]), field) == value


def test_overrides_can_repair_invalid_yaml_values(config_path):
    config_path.write_text(yaml.safe_dump(config_data(k="invalid")))
    assert _load_config(str(config_path), ["k=5"]).k == 5


@pytest.mark.parametrize("assignment", [
    "k", "=1", " =1", "embedding..device=cpu", ".k=1",
    "embedding.batch_size.value=2", "embedding.unknown=1", "unknown=1",
    "embedding=[", "embedding={unknown: 1}", "k=true",
    "cache_validation_embeddings=1", 'embedding.batch_size="32"',
])
def test_invalid_assignments_before_tracking(config_path, assignment):
    with patch("classification.cli.wandb.init") as initialize, patch("classification.cli.Experiment") as experiment, pytest.raises(SystemExit) as error:
        main(["experiment", "run", "--config_file", str(config_path), "--set", assignment])
    assert error.value.code == 2
    initialize.assert_not_called()
    experiment.assert_not_called()


@pytest.mark.parametrize("content", ["- sequence", "null", "embedding: [", "k: invalid"])
def test_invalid_yaml_before_tracking(config_path, content):
    config_path.write_text(content)
    with patch("classification.cli.wandb.init") as initialize, pytest.raises(SystemExit) as error:
        main(["experiment", "run", "--config_file", str(config_path)])
    assert error.value.code == 2
    initialize.assert_not_called()


def test_missing_section_created(config_path):
    settings = dict(binary_path="ollama", host="localhost:11434", startup_timeout=2, request_timeout=3)
    overrides = [f"ollama.{key}={value}" for key, value in settings.items()]
    result = _load_config(str(config_path), overrides)
    assert result.ollama == OllamaConfig.model_validate(settings)


@pytest.mark.parametrize("failure", [RuntimeError("stage failed"), KeyboardInterrupt()])
def test_stage_failures_propagate(config_path, failure):
    with patch("classification.cli.wandb.init"), patch("classification.cli.Tracker"), patch("classification.cli.Experiment") as experiment, pytest.raises(type(failure)) as error:
        experiment.return_value.run.side_effect = failure
        main(["experiment", "run", "--config_file", str(config_path)])
    assert error.value is failure


@pytest.mark.parametrize("failure", [None, RuntimeError("pull failed"), KeyboardInterrupt()])
def test_download_validates_only_ollama_section(failure):
    settings = dict(binary_path="ollama", host="localhost:11434", startup_timeout=2, request_timeout=3)
    with (
        patch("builtins.open", mock_open(read_data=yaml.safe_dump(dict(ollama=settings, unrelated="legacy")))),
        patch("classification.cli.wandb.init") as initialize,
        patch("classification.llm.ollama.OllamaSession") as session,
    ):
        session.return_value.__enter__.return_value.download_model.side_effect = failure
        if failure is None:
            main(["download_model", "requested:tag"])
        else:
            with pytest.raises(type(failure)):
                main(["download_model", "requested:tag"])
        initialize.assert_not_called()
        session.assert_called_once_with(OllamaConfig.model_validate(settings))
        session.return_value.__enter__.return_value.download_model.assert_called_once_with("requested:tag")
        session.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("args", [[], ["unknown"], ["evaluate"], ["experiment"], ["experiment", "unknown"], ["download_model"], ["experiment", "run"]])
def test_invalid_commands(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2

class SlurmOverrideTests(unittest.TestCase):
    def test_sweep_uses_separate_tables_and_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            capture = temporary / "arguments"
            sbatch = temporary / "sbatch"
            sbatch.write_text(
                '#!/bin/bash\n'
                '[[ "$INTERNVL_PIXI_ENV" == jina-cu126 || "$INTERNVL_PIXI_ENV" == cu126 ]] || exit 1\n'
                '{ printf "%s\\0" "$@"; printf "\\0"; } >> "$CAPTURE_ARGUMENTS"\n'
            )
            sbatch.chmod(0o755)
            env = dict(os.environ, PATH=f"{temporary}:{os.environ.get('PATH', '')}",
                       CAPTURE_ARGUMENTS=str(capture), INTERNVL_PIXI_ENV="wrong-inherited-environment")
            script = root / "scripts/submit_embedding_sweep.sh"
            subprocess.run(["bash", "-n", str(script)], check=True)
            custom_config = temporary / "config with spaces.yml"
            custom_config.write_text((root / "cfg/classification.yml").read_text())
            models = [
                "jinaai/jina-clip-v2",
                "jinaai/jina-embeddings-v4",
                "google/embeddinggemma-2",
                "openai/clip-vit-base-patch32",
                "Qwen/Qwen3-VL-Embedding-8B",
            ]
            tables, outputs = [], []
            for extra in ([], [str(custom_config)]):
                capture.write_bytes(b"")
                subprocess.run(["bash", str(script), *extra], cwd=root, env=env, check=True)
                submissions = capture.read_bytes().decode().removesuffix("\0\0").split("\0\0")
                self.assertEqual(len(submissions), 5)
                for model, submission in zip(models, submissions, strict=True):
                    args = submission.split("\0")
                    table = next(a.removeprefix("lancedb.table=") for a in args if a.startswith("lancedb.table="))
                    output = next(a.removeprefix("output_dir=") for a in args if a.startswith("output_dir="))
                    self.assertEqual(output, f"assets/sweeps/{table}")
                    expected = [
                        f"--job-name={table}",
                        "--export=ALL,INTERNVL_PIXI_ENV=" + ("jina-cu126" if model.startswith("jinaai/") else "cu126"),
                        "sbatch/run_embedding.sbatch",
                        *(extra or ["cfg/classification.yml"]),
                        "--set", f"embedding.model_id={model}",
                        "--set", "embedding.device=cuda",
                        "--set", f"output_dir={output}",
                        "--set", f"lancedb.table={table}",
                    ]
                    if model == "jinaai/jina-embeddings-v4":
                        expected.extend(["--set", "embedding.batch_size=1"])
                    elif model == "Qwen/Qwen3-VL-Embedding-8B":
                        expected.extend(["--set", "embedding.batch_size=8"])
                    self.assertEqual(args, expected)
                    tables.append(table)
                    outputs.append(output)
            self.assertEqual(len(set(tables)), 10)
            self.assertEqual(len(set(outputs)), 10)

    def test_wrappers_forward_arguments_and_keep_defaults(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            capture = temporary / "arguments"
            pixi = temporary / "pixi"
            pixi.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$CAPTURE_ARGUMENTS"\n')
            pixi.chmod(0o755)
            base_env = dict(
                os.environ,
                PATH=f"{temporary}:{os.environ.get('PATH', '')}",
                SLURM_SUBMIT_DIR=str(root),
                CAPTURE_ARGUMENTS=str(capture),
            )
            base_env.pop("INTERNVL_PIXI_ENV", None)
            overrides = [
                "config with spaces.yml",
                "--set",
                "output_dir=output with spaces",
                "--set=k=3",
                "--set=lancedb.uri=shared db",
                "--set=lancedb.table=fixture",
            ]
            for script, stage, envs in (
                ("scripts/generate_descriptions.sbatch", "describe", (None,)),
                ("sbatch/test_embedding.sbatch", "evaluate", (None, "jina-cu126", "cu126", "jina-cpu")),
                ("sbatch/run_embedding.sbatch", "run", (None, "jina-cu126", "cu126", "jina-cpu")),
            ):
                subprocess.run(["bash", "-n", str(root / script)], check=True)
                for environment in envs:
                    env = dict(base_env)
                    if environment is not None:
                        env["INTERNVL_PIXI_ENV"] = environment
                    for extra in ([], overrides) if environment is None else (overrides,):
                        with self.subTest(script=script, env=environment, extra=extra):
                            subprocess.run(
                                ["bash", str(root / script), *extra], env=env, check=True
                            )
                            actual = capture.read_bytes().decode().rstrip("\0").split("\0")
                            self.assertEqual(
                                actual,
                                [
                                    "run",
                                    "--manifest-path",
                                    "src/classification/pyproject.toml",
                                    "--as-is",
                                    "-e",
                                    environment or "cu126",
                                    "classification",
                                    "experiment",
                                    stage,
                                    "--config_file",
                                    *(extra or ["cfg/classification.yml"]),
                                ],
                            )


if __name__ == "__main__":
    unittest.main()
