"""CPU checks for CLI dispatch and Ollama downloads, with no model or service calls."""

import contextlib
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, mock_open, patch

import yaml

from internvl.cli import main


class CommandTests(unittest.TestCase):
    def test_command_help_does_not_dispatch(self):
        for command in ([], ["experiment"], ["download_model"]):
            with (
                self.subTest(command=command),
                contextlib.redirect_stdout(io.StringIO()),
                patch("internvl.cli._run_experiment") as run,
                patch("internvl.cli.download_model") as download,
                self.assertRaises(SystemExit) as error,
            ):
                main([*command, "--help"])
                self.assertEqual(error.exception.code, 0)
                run.assert_not_called()
                download.assert_not_called()

    def test_configuration_failures_propagate(self):
        for command, loader in (
            (
                ["experiment", "run", "--config_file", "selected.yml"],
                "internvl.cli.OmegaConf.load",
            ),
            (["download_model", "requested:tag"], "yaml.safe_load"),
        ):
            for source in ("open", "yaml"):
                failure = (
                    OSError("read failed")
                    if source == "open"
                    else ValueError("parse failed")
                )
                with (
                    self.subTest(command=command, source=source),
                    patch("builtins.open", mock_open()) as opened,
                    patch(loader) as load,
                    patch("internvl.cli.Experiment") as experiment,
                    patch("internvl.generate_descriptors.DescriberLLM") as descriptors,
                    self.assertRaises(type(failure)) as error,
                ):
                    if source == "open":
                        opened.side_effect = failure
                    else:
                        load.side_effect = failure
                    main(command)
                self.assertIs(error.exception, failure)
                experiment.assert_not_called()
                descriptors.assert_not_called()

    def test_experiment_failures_and_interrupts_propagate(self):
        for source in ("constructor", "run"):
            for failure in (RuntimeError("backend failed"), KeyboardInterrupt()):
                with (
                    self.subTest(source=source, failure=type(failure).__name__),
                    patch("internvl.cli.Experiment") as experiment,
                    patch(
                        "builtins.open", mock_open(read_data="output_dir: synthetic")
                    ),
                    self.assertRaises(type(failure)) as error,
                ):
                    target = (
                        experiment
                        if source == "constructor"
                        else experiment.return_value.run
                    )
                    target.side_effect = failure
                    main(["experiment", "run", "--config_file", "selected.yml"])
                self.assertIs(error.exception, failure)

    def test_download_reads_working_directory_config_and_exact_model(self):
        backend = SimpleNamespace()
        backend.DescriberLLM = MagicMock()
        yaml = SimpleNamespace()
        cfg = {"ollama": {"llm": "configured:tag", "host": "synthetic"}}

        def load(stream):
            self.assertEqual(stream.read(), content)
            return cfg

        yaml.safe_load = Mock(side_effect=load)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cfg").mkdir()
            path = root / "cfg/experiment_config_internvl.yml"
            content = "ollama:\n  llm: configured:tag\n  host: synthetic\n"
            path.write_text(content)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(
                    sys.modules,
                    {
                        "internvl.generate_descriptors": backend,
                        "yaml": yaml,
                        "util": None,
                    },
                ):
                    main(["download_model", "requested:exact-tag"])
            finally:
                os.chdir(previous)
            self.assertEqual(path.read_text(), content)
        backend.DescriberLLM.assert_called_once_with(
            {"llm": "configured:tag", "host": "synthetic"}
        )
        context = backend.DescriberLLM.return_value
        context.__enter__.return_value.download_model.assert_called_once_with(
            "requested:exact-tag"
        )
        context.__exit__.assert_called_once_with(None, None, None)

    def test_download_cleanup_on_failure_and_interrupt(self):
        for failure in (RuntimeError("pull failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                backend = SimpleNamespace()
                backend.DescriberLLM = MagicMock()
                context = backend.DescriberLLM.return_value
                context.__enter__.return_value.download_model.side_effect = failure
                yaml = SimpleNamespace()
                yaml.safe_load = Mock(return_value={"ollama": {"llm": "fake"}})
                with (
                    patch.dict(
                        sys.modules,
                        {
                            "internvl.generate_descriptors": backend,
                            "yaml": yaml,
                        },
                    ),
                    patch("builtins.open", mock_open()),
                    self.assertRaises(type(failure)) as error,
                ):
                    main(["download_model", "requested:tag"])
                self.assertIs(error.exception, failure)
                context.__exit__.assert_called_once()
                self.assertIs(context.__exit__.call_args.args[0], type(failure))
                self.assertIs(context.__exit__.call_args.args[1], failure)

    def test_argument_errors_do_not_dispatch(self):
        cases = [
            [],
            ["unknown"],
            ["evaluate"],
            ["download_model"],
            ["experiment"],
            ["experiment", "unknown"],
            ["experiment", "run"],
            ["experiment", "unknown", "--config_file", "unused.yml"],
        ]
        cases += [
            [command]
            for command in (
                "caption-images",
                "merge-captions",
                "convert-captions",
                "generate-descriptions",
                "build-db",
                "finetune-embeddings",
            )
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with (
                    patch("internvl.cli.download_model") as download,
                    patch("internvl.cli._run_experiment") as run,
                    self.assertRaises(SystemExit) as error,
                ):
                    main(argv)
                self.assertEqual(error.exception.code, 2)
                download.assert_not_called()
                run.assert_not_called()

    def test_help_lists_supported_commands(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as error:
            main(["--help"])
        self.assertEqual(error.exception.code, 0)
        for command in ("experiment", "download_model"):
            self.assertIn(command, output.getvalue())
        for command in (
            "build-db",
            "evaluate",
            "caption-images",
            "merge-captions",
            "convert-captions",
            "generate-descriptions",
        ):
            self.assertNotIn(command, output.getvalue())

    def test_removed_retrieval_commands_are_invalid(self):
        for command in ("build-db", "evaluate"):
            output = io.StringIO()
            with (
                self.subTest(command=command),
                contextlib.redirect_stderr(output),
                self.assertRaises(SystemExit) as error,
            ):
                main([command, "--config_file", "unused.yml"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("invalid choice", output.getvalue())

    def test_experiment_dispatch(self):
        for stage, method in (
            ("describe", "generate_descriptions"),
            ("index", "build_vector_database"),
            ("evaluate", "evaluate"),
            ("run", "run"),
        ):
            with (
                self.subTest(stage=stage),
                patch("internvl.cli.Experiment") as experiment,
                patch(
                    "builtins.open", mock_open(read_data="output_dir: synthetic")
                ) as opened,
            ):
                main(["experiment", stage, "--config_file", "selected.yml"])
            opened.assert_called_once_with("selected.yml", encoding="utf-8")
            experiment.assert_called_once_with({"output_dir": "synthetic"})
            self.assertEqual(
                experiment.return_value.method_calls, [getattr(call, method)()]
            )

    @unittest.skipUnless(
        importlib.util.find_spec("yaml") is not None, "requires PyYAML"
    )
    def test_cli_dispatches_experiment_stages(self):
        backend = SimpleNamespace()
        experiment = backend.Experiment = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yml"
            config_path.write_text("k: 7")
            for stage, method in [
                ("index", "build_vector_database"),
                ("describe", "generate_descriptions"),
                ("evaluate", "evaluate"),
                ("run", "run"),
            ]:
                with (
                    self.subTest(stage=stage),
                    patch("internvl.cli.Experiment", experiment),
                ):
                    main(["experiment", stage, "--config_file", str(config_path)])
                experiment.assert_called_once_with({"k": 7})
                getattr(experiment.return_value, method).assert_called_once_with()
                experiment.reset_mock()


class ConfigOverrideTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(directory) / "config.yml"
        self.config = {
            "ollama": {"llm": "original:tag", "host": "http://localhost:11434"},
            "embedding": {"batch_size": 32},
            "lancedb": {"uri": "assets/lancedb", "table": "fixture"},
            "embed_images": True,
            "k": 10,
            "threshold": 0.5,
            "labels": ["old"],
            "optional": None,
            "output_dir": "original",
            "command": "a config key, not the subcommand",
        }
        self.content = yaml.safe_dump(self.config)
        self.path.write_text(self.content)
        self.experiment = self.enterContext(patch("internvl.cli.Experiment"))

    def invoke(self, *overrides, stage="run"):
        main(["experiment", stage, "--config_file", str(self.path), *overrides])
        self.assertEqual(self.path.read_text(), self.content)
        return self.experiment.call_args.args[0]

    def test_nested_overrides_reach_every_stage(self):
        for stage, method in (
            ("describe", "generate_descriptions"),
            ("index", "build_vector_database"),
            ("evaluate", "evaluate"),
            ("run", "run"),
        ):
            with self.subTest(stage=stage):
                self.experiment.reset_mock()
                actual = self.invoke(
                    "--set",
                    "ollama.llm=gemma3:27b",
                    "--set",
                    "embedding.batch_size=16",
                    "--set",
                    "embed_images=false",
                    "--set",
                    "threshold=0.75",
                    "--set",
                    "lancedb.uri=assets/other",
                    "--set",
                    "lancedb.table=other_table",
                    "--set",
                    "output_dir=path with spaces",
                    "--set",
                    "command=literal",
                    stage=stage,
                )
                self.assertIs(type(actual), dict)
                self.assertIs(type(actual["embedding"]), dict)
                self.assertEqual(actual["ollama"]["llm"], "gemma3:27b")
                self.assertEqual(
                    actual["ollama"]["host"], self.config["ollama"]["host"]
                )
                self.assertEqual(actual["embedding"]["batch_size"], 16)
                self.assertIs(actual["embed_images"], False)
                self.assertEqual(actual["threshold"], 0.75)
                self.assertEqual(
                    actual["lancedb"]["uri"], "assets/other"
                )
                self.assertEqual(actual["lancedb"]["table"], "other_table")
                self.assertEqual(actual["output_dir"], "path with spaces")
                self.assertEqual(actual["command"], "literal")
                self.assertEqual(actual["k"], 10)
                self.assertEqual(
                    self.experiment.return_value.method_calls, [getattr(call, method)()]
                )

    def test_omitted_overrides_preserve_config(self):
        self.assertEqual(self.invoke(), self.config)

    def test_last_override_wins_and_values_can_contain_equals(self):
        actual = self.invoke("--set", "k=3", "--set=k=7", "--set", "output_dir=a=b")
        self.assertEqual(actual["k"], 7)
        self.assertEqual(actual["output_dir"], "a=b")

    def test_yaml_values_and_explicit_string_quotes(self):
        for raw, expected in (
            ("null", None),
            ("[1, two]", [1, "two"]),
            ("false", False),
            ("9", 9),
            ("1.5", 1.5),
            ('"001"', "001"),
            ('"false"', "false"),
        ):
            with self.subTest(raw=raw):
                actual = self.invoke(
                    "--set", "labels=[new, other]", "--set", f"optional={raw}"
                )
                self.assertEqual(actual["labels"], ["new", "other"])
                self.assertEqual(actual["optional"], expected)
                self.assertIs(type(actual["optional"]), type(expected))

    def test_type_changes_reach_application_validation(self):
        failure = ValueError("application validation failed")
        self.experiment.side_effect = failure
        with self.assertRaises(ValueError) as error:
            self.invoke("--set", "embedding.batch_size=invalid")
        self.assertIs(error.exception, failure)
        self.assertEqual(
            self.experiment.call_args.args[0]["embedding"]["batch_size"], "invalid"
        )

    def test_interpolations_resolve_before_dispatch(self):
        actual = self.invoke("--set", "output_dir=${ollama.llm}")
        self.assertEqual(actual["output_dir"], "original:tag")

    def test_override_order_relative_to_stage_and_config(self):
        for arguments in (
            ["experiment", "--set", "k=5", "run", "--config_file", str(self.path)],
            ["experiment", "--config_file", str(self.path), "--set", "k=5", "run"],
            ["experiment", "run", "--set", "k=5", "--config_file", str(self.path)],
        ):
            with self.subTest(arguments=arguments):
                main(arguments)
                self.assertEqual(self.experiment.call_args.args[0]["k"], 5)

    def test_invalid_overrides_fail_before_construction(self):
        for overrides in (
            ["--set", "ollama.lm=typo"],
            ["--set", "new_key=1"],
            ["--set", "qdrant.collection_prefix=old"],
            ["--set", "k"],
            ["--set", "=1"],
            ["--set", " =1"],
            ["--set"],
            ["--set", "labels=["],
            ["--ollama.llm", "old syntax"],
            ["--se", "k=1"],
        ):
            with (
                self.subTest(overrides=overrides),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    self.invoke(*overrides)
                self.assertEqual(error.exception.code, 2)
                self.experiment.assert_not_called()
                self.assertEqual(self.path.read_text(), self.content)

    def test_help_never_reads_config(self):
        for arguments in (
            ["experiment", "--help"],
            ["experiment", "run", "--config_file", "nonexistent.yml", "--help"],
        ):
            output = io.StringIO()
            with (
                self.subTest(arguments=arguments),
                patch("builtins.open") as opened,
                contextlib.redirect_stdout(output),
                self.assertRaises(SystemExit) as error,
            ):
                main(arguments)
            self.assertEqual(error.exception.code, 0)
            self.assertIn("--set PATH=VALUE", output.getvalue())
            opened.assert_not_called()
            self.experiment.assert_not_called()

    def test_non_mapping_config_is_rejected(self):
        self.path.write_text("[one, two]")
        with self.assertRaisesRegex(ValueError, "YAML mapping"):
            main(["experiment", "run", "--config_file", str(self.path)])
        self.experiment.assert_not_called()


@unittest.skipUnless(shutil.which("bash"), "requires bash")
class SlurmOverrideTests(unittest.TestCase):
    def test_sweep_uses_separate_tables_and_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            capture = temporary / "arguments"
            sbatch = temporary / "sbatch"
            sbatch.write_text('#!/bin/bash\nprintf "%s\\0" "$@" >> "$CAPTURE_ARGUMENTS"\n')
            sbatch.chmod(0o755)
            env = dict(os.environ, PATH=f"{temporary}:{os.environ.get('PATH', '')}",
                       CAPTURE_ARGUMENTS=str(capture))
            script = root / "scripts/submit_embedding_sweep.sh"
            subprocess.run(["bash", "-n", str(script)], check=True)
            tables, outputs = [], []
            for _ in range(2):
                capture.write_bytes(b"")
                subprocess.run(["bash", str(script)], cwd=root, env=env, check=True)
                args = capture.read_bytes().decode().rstrip("\0").split("\0")
                table = next(a.removeprefix("lancedb.table=") for a in args if a.startswith("lancedb.table="))
                output = next(a.removeprefix("output_dir=") for a in args if a.startswith("output_dir="))
                self.assertEqual(output, f"assets/sweeps/{table}")
                self.assertIn("sbatch/run_embedding.sbatch", args)
                self.assertIn("cfg/classification.yml", args)
                self.assertFalse(any("qdrant" in a for a in args))
                tables.append(table)
                outputs.append(output)
            self.assertEqual(len(set(tables)), 2)
            self.assertEqual(len(set(outputs)), 2)

    def test_wrappers_forward_arguments_and_keep_defaults(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            capture = temporary / "arguments"
            pixi = temporary / "pixi"
            pixi.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$CAPTURE_ARGUMENTS"\n')
            pixi.chmod(0o755)
            env = dict(
                os.environ,
                PATH=f"{temporary}:{os.environ.get('PATH', '')}",
                SLURM_SUBMIT_DIR=str(root),
                CAPTURE_ARGUMENTS=str(capture),
            )
            for script, stage in (
                ("scripts/generate_descriptions.sbatch", "describe"),
                ("sbatch/test_embedding.sbatch", "evaluate"),
                ("sbatch/run_embedding.sbatch", "run"),
            ):
                subprocess.run(["bash", "-n", str(root / script)], check=True)
                for extra in (
                    [],
                    [
                        "config with spaces.yml",
                        "--set",
                        "output_dir=output with spaces",
                        "--set=k=3",
                        "--set=lancedb.uri=shared db",
                        "--set=lancedb.table=fixture",
                    ],
                ):
                    with self.subTest(script=script, extra=extra):
                        subprocess.run(
                            ["bash", str(root / script), *extra], env=env, check=True
                        )
                        actual = capture.read_bytes().decode().rstrip("\0").split("\0")
                        self.assertEqual(
                            actual,
                            [
                                "run",
                                "--manifest-path",
                                "src/internvl/pyproject.toml",
                                "--as-is",
                                "-e",
                                "cu126",
                                "internvl",
                                "experiment",
                                stage,
                                "--config_file",
                                *(extra or ["cfg/classification.yml"]),
                            ],
                        )


if __name__ == "__main__":
    unittest.main()
