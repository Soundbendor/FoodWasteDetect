"""CPU checks for CLI dispatch and Ollama downloads, with no model or service calls."""

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call, mock_open, patch

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
        for command in (
            ["experiment", "run", "--config_file", "selected.yml"],
            ["download_model", "requested:tag"],
        ):
            for source in ("open", "yaml"):
                with self.subTest(command=command, source=source):
                    backend = SimpleNamespace()
                    backend.Experiment = MagicMock()
                    descriptors = SimpleNamespace()
                    descriptors.DescriberLLM = MagicMock()
                    yaml = SimpleNamespace()
                    yaml.safe_load = Mock()
                    failure = (
                        OSError("read failed")
                        if source == "open"
                        else ValueError("parse failed")
                    )
                    opened = mock_open()
                    if source == "open":
                        opened.side_effect = failure
                    else:
                        yaml.safe_load.side_effect = failure
                    with (
                        patch.dict(
                            sys.modules,
                            {
                                "yaml": yaml,
                                "internvl.experiment": backend,
                                "internvl.generate_descriptors": descriptors,
                            },
                        ),
                        patch("internvl.cli.Experiment", backend.Experiment),
                        patch("internvl.cli.yaml", yaml),
                        patch("builtins.open", opened),
                        self.assertRaises(type(failure)) as error,
                    ):
                        main(command)
                    self.assertIs(error.exception, failure)
                    backend.Experiment.assert_not_called()
                    descriptors.DescriberLLM.assert_not_called()

    def test_experiment_failures_and_interrupts_propagate(self):
        for source in ("constructor", "run"):
            for failure in (RuntimeError("backend failed"), KeyboardInterrupt()):
                with self.subTest(source=source, failure=type(failure).__name__):
                    backend = SimpleNamespace()
                    backend.Experiment = MagicMock()
                    target = (
                        backend.Experiment
                        if source == "constructor"
                        else backend.Experiment.return_value.run
                    )
                    target.side_effect = failure
                    yaml = SimpleNamespace()
                    yaml.safe_load = Mock(return_value={"output_dir": "synthetic"})
                    with (
                        patch.dict(
                            sys.modules, {"yaml": yaml, "internvl.experiment": backend}
                        ),
                        patch("internvl.cli.Experiment", backend.Experiment),
                        patch("internvl.cli.yaml", yaml),
                        patch("builtins.open", mock_open()),
                        self.assertRaises(type(failure)) as error,
                    ):
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
            with self.subTest(stage=stage):
                backend = SimpleNamespace()
                backend.Experiment = MagicMock()
                yaml = SimpleNamespace()
                config = {"output_dir": "synthetic"}
                yaml.safe_load = Mock(return_value=config)
                with (
                    patch.dict(
                        sys.modules, {"internvl.experiment": backend, "yaml": yaml}
                    ),
                    patch("internvl.cli.Experiment", backend.Experiment),
                    patch("internvl.cli.yaml", yaml),
                    patch(
                        "builtins.open",
                        mock_open(read_data="output_dir: synthetic"),
                    ) as opened,
                ):
                    main(["experiment", stage, "--config_file", "selected.yml"])
                opened.assert_called_once_with("selected.yml", encoding="utf-8")
                backend.Experiment.assert_called_once_with(config)
                self.assertIs(backend.Experiment.call_args.args[0], config)
                self.assertEqual(
                    backend.Experiment.return_value.method_calls,
                    [getattr(call, method)()],
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


if __name__ == "__main__":
    unittest.main()
