"""CPU checks for the text-pipeline extraction, with no model or service calls."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from internvl.cli import main
from internvl import workflows


def available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


class CommandTests(unittest.TestCase):
    def test_every_command_dispatches_without_changing_configuration(self):
        cfg = {"paths": {"dataset": "synthetic"}, "llm": "fake"}
        fake_util = types.ModuleType("util")
        fake_util.parse_cfg = Mock(return_value=cfg)
        cases = [
            ("caption-images", "caption_imgs", True),
            ("generate-descriptions", "generate_descriptions", True),
            ("merge-captions", "merge_files", False),
            ("convert-captions", "convert_csv_to_dictionary", False),
        ]
        for command, function, configured in cases:
            with self.subTest(command=command):
                fake_util.parse_cfg.reset_mock()
                argv = [command]
                expected = {}
                if configured:
                    argv += ["--config_file", "example.yml"]
                    expected["cfg"] = cfg
                if command == "caption-images":
                    argv += ["--partition", "3"]
                    expected["partition"] = 3
                with patch.dict(sys.modules, {"util": fake_util}), patch.object(
                    workflows, function
                ) as execute:
                    main(argv)
                execute.assert_called_once_with(**expected)
                if configured:
                    fake_util.parse_cfg.assert_called_once_with("example.yml")
                else:
                    fake_util.parse_cfg.assert_not_called()

    def test_download_reads_working_directory_config_and_exact_model(self):
        backend = types.ModuleType("internvl.generate_descriptors")
        backend.DescriberLLM = MagicMock()
        yaml = types.ModuleType("yaml")
        cfg = {'ollama': {'llm': 'configured:tag', 'host': 'synthetic'}}
        def load(stream):
            self.assertEqual(stream.read(), content)
            return cfg
        yaml.safe_load = Mock(side_effect=load)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'cfg').mkdir()
            path = root / 'cfg/experiment_config_internvl.yml'
            content = 'ollama:\n  llm: configured:tag\n  host: synthetic\n'
            path.write_text(content)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(sys.modules, {
                    'internvl.generate_descriptors': backend, 'yaml': yaml,
                    'util': None,
                }):
                    main(['download_model', 'requested:exact-tag'])
            finally:
                os.chdir(previous)
            self.assertEqual(path.read_text(), content)
        backend.DescriberLLM.assert_called_once_with(
            {'llm': 'configured:tag', 'host': 'synthetic'})
        context = backend.DescriberLLM.return_value
        context.__enter__.return_value.download_model.assert_called_once_with('requested:exact-tag')
        context.__exit__.assert_called_once_with(None, None, None)

    def test_argument_errors_do_not_dispatch(self):
        for argv in [[], ["unknown"], ["evaluate"], ["caption-images", "--config_file", "x"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(workflows, "test_classification") as evaluate:
                    with self.assertRaises(SystemExit) as error:
                        main(argv)
                self.assertEqual(error.exception.code, 2)
                evaluate.assert_not_called()


@unittest.skipUnless(available("pandas", "numpy"), "requires pandas and numpy")
class ArtifactTests(unittest.TestCase):
    def test_merge_and_conversion_preserve_csv_and_class_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            expected = {}
            for partition in range(1, 5):
                label = "Macaron" if partition % 2 else "Apple"
                caption = f'A red, "round" food {partition}'
                expected.setdefault(label, []).append(caption)
                with (root / "assets" / f"foodx251_captions_{partition}.txt").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["idx", "class", "caption", "unused"])
                    writer.writerow([partition * 10, label, caption, "ignored"])
            previous = Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["merge-captions"])
                    main(["convert-captions"])
            finally:
                os.chdir(previous)
            with (root / "assets/foodx251_captions.txt").open(newline="") as stream:
                reader = csv.DictReader(stream)
                records = list(reader)
                self.assertEqual(reader.fieldnames, ["idx", "class", "caption"])
            self.assertEqual([row["idx"] for row in records], ["10", "20", "30", "40"])
            self.assertEqual(json.loads((root / "foodx251_captions.json").read_text()), expected)

    @unittest.skipUnless(available("yaml", "torch", "qdrant_client"), "requires PyYAML and legacy database dependencies")
    def test_yaml_config_is_forwarded_to_selected_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "experiment.yml"
            config.write_text("llm: fake\nsettings:\n  eval_samples: 7\n")
            with patch.object(workflows, "generate_descriptions") as evaluate:
                main(["generate-descriptions", "--config_file", str(config)])
            evaluate.assert_called_once_with(cfg={"llm": "fake", "settings": {"eval_samples": 7}})


if __name__ == "__main__":
    unittest.main()
