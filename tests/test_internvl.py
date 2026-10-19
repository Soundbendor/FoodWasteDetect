"""CPU checks for the text-pipeline extraction, with no model or service calls."""

import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import MagicMock, Mock, patch

from internvl.cli import main
from internvl import workflows


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


def run_python(code):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(SRC), env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )


# Run isolation checks in fresh interpreters so cached imports cannot hide a leak.
IMPORT_GUARD = """
import importlib.abc
import sys

class RejectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('Unexpected dependency: ' + fullname)

sys.meta_path.insert(0, RejectImports())
"""


class CommandTests(unittest.TestCase):
    def test_help_and_workflow_imports_do_not_load_backends(self):
        result = run_python(
            "forbidden = {'torch', 'torchvision', 'transformers', 'sentence_transformers', "
            "'datasets', 'ollama', 'pydantic', 'ultralytics', 'qdrant_client', "
            "'requests', 'pandas', 'numpy', 'yaml', 'PIL'}\n"
            + IMPORT_GUARD
            + """
import runpy
import internvl
import internvl.workflows
from internvl.cli import main

for command in [None, 'caption-images', 'generate-descriptions', 'build-db',
                'evaluate', 'merge-captions', 'convert-captions',
                'download_model']:
    try:
        main(([command] if command else []) + ['--help'])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError('Help did not exit')
try:
    main(['download_model'])
except SystemExit as exc:
    assert exc.code == 2
else:
    raise AssertionError('Missing model name did not exit')
sys.argv = ['internvl', '--help']
try:
    runpy.run_module('internvl', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("caption-images", result.stdout)

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


class ImportBoundaryTests(unittest.TestCase):
    @unittest.skipUnless(available("pandas", "numpy", "PIL", "tqdm"), "requires CPU dataframe/image dependencies")
    def test_shared_dataset_and_metrics_do_not_import_model_or_detection_backends(self):
        result = run_python(
            "forbidden = {'ultralytics', 'database.img_crop', 'database.clip_embedding', "
            "'internvl', 'ollama', 'torch', 'transformers', 'sentence_transformers', "
            "'qdrant_client', 'mean_average_precision', 'sklearn'}\n"
            + IMPORT_GUARD
            + """
from data_wrappers.foodx251 import FoodX251
from util import EvalMetric

# Importing these helpers must not construct a model or load an external dataset.
metric = EvalMetric()
assert metric.len == 0
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_image_app_does_not_import_text_backends(self):
        result = run_python(
            "forbidden = {'internvl', 'vlm', 'llm', 'database.embedding', 'ollama'}\n"
            + IMPORT_GUARD
            + """
import types

# Stand-ins for the app's image path, not for the forbidden text dependencies.
for module_name, attributes in {
    'data_wrappers.foodx251': ['FoodX251'],
    'database.clip_embedding': ['CLIPEmbedding'],
    'database.vecdb': ['VectorDB'],
    'util': ['EvalMetric', 'parse_args', 'parse_cfg'],
}.items():
    module = types.ModuleType(module_name)
    for attribute in attributes:
        setattr(module, attribute, object)
    sys.modules[module_name] = module
import app
assert callable(app.build_img_vdb)
assert callable(app.eval_img_vdb)
assert not hasattr(app, 'caption_imgs')
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_qdrant_without_reranking_does_not_import_model_backends(self):
        result = run_python(
            "forbidden = {'torch', 'sentence_transformers', 'internvl', 'ollama'}\n"
            + IMPORT_GUARD
            + """
import types
from unittest.mock import Mock, patch

requests = types.ModuleType('requests')
requests.get = Mock(side_effect=AssertionError('Unexpected HTTP call'))
sys.modules['requests'] = requests
qdrant = types.ModuleType('qdrant_client')
qdrant.QdrantClient = Mock(side_effect=AssertionError('Unexpected client construction'))
models = types.ModuleType('qdrant_client.models')
for name in ['Distance', 'PointStruct', 'ScoredPoint', 'VectorParams']:
    setattr(models, name, object)
sys.modules['qdrant_client'] = qdrant
sys.modules['qdrant_client.models'] = models

from database.vecdb import VectorDB
client = Mock()
client.query_points.return_value = types.SimpleNamespace(points=[
    types.SimpleNamespace(score=0.2, payload={'class': 'Apple'}),
    types.SimpleNamespace(score=0.8, payload={'class': 'Pear'}),
])
with patch.object(VectorDB, 'connect', return_value=client):
    db = VectorDB('unused', 'synthetic', None, 2)
assert db.reranker is None
matches = db.query(None, [1.0, 0.0])
assert [match.payload['class'] for match in matches] == ['Pear', 'Apple']
client.query_points.assert_called_once_with(
    collection_name='synthetic', query=[1.0, 0.0], limit=10)
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)


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

    def test_caption_utilities_do_not_import_model_backends(self):
        result = run_python(
            "forbidden = {'internvl.model', 'internvl.generate_descriptors', 'internvl.embeddings', "
            "'ultralytics', 'ollama', 'torch', 'transformers', 'sentence_transformers', 'qdrant_client'}\n"
            + IMPORT_GUARD
            + """
import os
import tempfile
from pathlib import Path
from internvl.cli import main

with tempfile.TemporaryDirectory() as directory:
    os.chdir(directory)
    Path('assets').mkdir()
    for partition in range(1, 5):
        Path(f'assets/foodx251_captions_{partition}.txt').write_text(
            f'idx,class,caption\\n{partition},Apple,A red apple\\n')
    main(['merge-captions'])
    main(['convert-captions'])
"""
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(available("yaml"), "requires PyYAML")
    def test_yaml_config_is_forwarded_to_selected_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "experiment.yml"
            config.write_text("llm: fake\nsettings:\n  eval_samples: 7\n")
            with patch.object(workflows, "generate_descriptions") as evaluate:
                main(["generate-descriptions", "--config_file", str(config)])
            evaluate.assert_called_once_with(cfg={"llm": "fake", "settings": {"eval_samples": 7}})


if __name__ == "__main__":
    unittest.main()
