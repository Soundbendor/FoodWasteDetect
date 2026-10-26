"""Synthetic W&B contracts: fake SDK publishing plus isolated native offline runs."""

import io
import itertools
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
import test_classification
import yaml

from internvl.cli import main
from internvl.tracking import Tracker


class Artifact:
    def __init__(self, name, type):
        self.name, self.type = name, type
        self.files = {}

    @contextmanager
    def new_file(self, name, *, encoding):
        assert encoding == "utf-8"
        with io.StringIO() as stream:
            yield stream
            self.files[name] = stream.getvalue().encode(encoding)

    def add_file(self, path, *, name):
        self.files[name] = Path(path).read_bytes()


@pytest.fixture
def sdk(monkeypatch):
    run = MagicMock(id="synthetic", summary={})
    run.__enter__.return_value = run
    run.__exit__.return_value = False
    monkeypatch.setattr("internvl.cli.wandb.init", Mock(return_value=run))
    monkeypatch.setattr("internvl.tracking.wandb.Artifact", Artifact)
    return run


@pytest.fixture
def experiment_case():
    case = test_classification.ExperimentTests()
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    source = """# original comment
embed_images: false
embed_class_labels: true
embed_descriptions: false
output_dir: synthetic
embedding:
  batch_size: 2
k: ${embedding.batch_size}
"""
    path.write_text(source)
    return path, source


@pytest.mark.parametrize(
    "stage,method",
    [
        ("describe", "generate_descriptions"),
        ("index", "build_vector_database"),
        ("evaluate", "evaluate"),
        ("run", "run"),
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_cli_owns_one_run_and_exact_config(sdk, tmp_path, stage, method, newline):
    path, source = config_file(tmp_path)
    source = source.replace("\n", newline)
    path.write_bytes(source.encode())
    overrides = ["embedding.batch_size=4", "embedding.batch_size=3"]
    with (
        patch("internvl.cli.Experiment") as experiment,
        patch("internvl.cli.wandb.init", return_value=sdk) as initialize,
    ):
        def assert_active(*args, **kwargs):
            sdk.__enter__.assert_called_once_with()
            sdk.__exit__.assert_not_called()

        sdk.log_artifact.side_effect = assert_active

        def construct(*args, **kwargs):
            assert_active()
            sdk.log_artifact.assert_called_once()
            return experiment.return_value

        experiment.side_effect = construct
        getattr(experiment.return_value, method).side_effect = assert_active
        main(
            [
                "experiment",
                stage,
                "--config_file",
                str(path),
                "--set",
                overrides[0],
                "--set",
                overrides[1],
            ]
        )
    initialize.assert_called_once()
    cfg = experiment.call_args.args[0]
    assert cfg == initialize.call_args.kwargs["config"]
    assert cfg["k"] == cfg["embedding"]["batch_size"] == 3
    assert initialize.call_args.kwargs["job_type"] == stage
    settings = initialize.call_args.kwargs["settings"]
    assert settings.disable_git and settings.disable_code and not settings.save_code
    assert settings.x_disable_meta and not settings.x_save_requirements
    assert "mode" not in initialize.call_args.kwargs
    assert experiment.call_args.kwargs["tracker"].run is sdk
    getattr(experiment.return_value, method).assert_called_once_with()
    sdk.__enter__.assert_called_once_with()
    sdk.__exit__.assert_called_once_with(None, None, None)
    sdk.finish.assert_not_called()
    artifact = sdk.log_artifact.call_args.args[0]
    assert artifact.name == "configuration-synthetic"
    assert set(artifact.files) == {"resolved_config.yaml"}
    assert yaml.safe_load(artifact.files["resolved_config.yaml"]) == cfg
    assert path.read_bytes() == source.encode()


@pytest.mark.parametrize(
    "kind", ["help", "unknown", "malformed", "mapping", "switches", "download"]
)
def test_no_run_for_non_experiments_or_invalid_input(sdk, tmp_path, kind):
    path, _ = config_file(tmp_path)
    args = ["experiment", "run", "--config_file", str(path)]
    if kind == "help":
        args = ["--help"]
    elif kind == "unknown":
        args += ["--set", "embedding.unknown=1"]
    elif kind == "malformed":
        args += ["--set", "broken"]
    elif kind == "mapping":
        path.write_text("- not a mapping")
    elif kind == "switches":
        args += ["--set", "embed_class_labels=false"]
    elif kind == "download":
        args = ["download_model", "fake"]
    with (
        patch("internvl.cli.wandb.init") as initialize,
        patch("internvl.cli.download_model"),
    ):
        if kind == "download":
            main(args)
        else:
            with pytest.raises((SystemExit, ValueError)):
                main(args)
        initialize.assert_not_called()


@pytest.mark.parametrize(
    "source", ["init", "artifact", "constructor", "stage", "interrupt", "success"]
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_native_context_exit_on_errors_and_success(sdk, tmp_path, source, cleanup_fails):
    path, _ = config_file(tmp_path)
    failure = KeyboardInterrupt() if source == "interrupt" else RuntimeError(source)
    cleanup_error = RuntimeError("cleanup")
    with (
        patch("internvl.cli.Experiment") as experiment,
        patch("internvl.cli.wandb.init", return_value=sdk) as initialize,
    ):
        if cleanup_fails:
            sdk.__exit__.side_effect = cleanup_error
        if source == "init":
            initialize.side_effect = failure
        elif source == "artifact":
            sdk.log_artifact.side_effect = failure
        elif source == "constructor":
            experiment.side_effect = failure
        elif source in ("stage", "interrupt"):
            experiment.return_value.run.side_effect = failure
        if source == "success" and not cleanup_fails:
            main(["experiment", "run", "--config_file", str(path)])
        else:
            with pytest.raises(BaseException) as caught:
                main(["experiment", "run", "--config_file", str(path)])
            expected = cleanup_error if cleanup_fails and source != "init" else failure
            assert caught.value is expected
        if source == "init":
            sdk.__enter__.assert_not_called()
            sdk.__exit__.assert_not_called()
        else:
            sdk.__enter__.assert_called_once_with()
            sdk.__exit__.assert_called_once()
            if source == "success":
                assert sdk.__exit__.call_args.args == (None, None, None)
            else:
                exc_type, error, traceback = sdk.__exit__.call_args.args
                assert exc_type is type(failure)
                assert error is failure
                assert traceback is not None
        sdk.finish.assert_not_called()
        if source in ("init", "artifact"):
            experiment.assert_not_called()


@pytest.mark.parametrize("flags", [x for x in itertools.product([False, True], repeat=3) if any(x)])
def test_pipeline_counts_and_reuse(experiment_case, sdk, flags):
    exp = experiment_case.experiment(flags)
    exp.batch_size = 3
    exp.tracker = Tracker(sdk)
    report = exp.run()
    history = [c.args[0] for c in sdk.log.call_args_list]
    totals = {
        "reference_embedding": sum(n for flag, n in zip(flags, [2, 2, 10], strict=True) if flag),
        "evaluation": 2,
    }
    if flags[2]:
        totals["descriptions"] = 10
    for stage, total in totals.items():
        points = [p for p in history if f"{stage}/completed" in p]
        assert points[0][f"{stage}/completed"] == 0
        assert points[-1][f"{stage}/completed"] == total
        assert all(p[f"{stage}/total"] == total for p in points)
        assert points[-1][f"{stage}/fraction"] == 1
    if flags[2]:
        assert [
            p["descriptions/completed"] for p in history if "descriptions/completed" in p
        ] == list(range(11))
    else:
        assert not any("descriptions/completed" in p for p in history)
    assert set(sdk.summary) == {
        "reference_source_counts", "evaluation/evaluated",
        "evaluation/majority_vote_accuracy", "evaluation/nearest_neighbor_accuracy",
    }
    assert sdk.summary["reference_source_counts"] == report["reference_source_counts"]
    for key in ("evaluated", "majority_vote_accuracy", "nearest_neighbor_accuracy"):
        assert sdk.summary[f"evaluation/{key}"] == report["metrics"][key]
    artifact = sdk.log_artifact.call_args.args[0]
    saved = next(exp.output.glob("evaluation_*/report.json"))
    assert artifact.files["report.json"] == saved.read_bytes()
    table = next(p["evaluation/per_class"] for p in history if "evaluation/per_class" in p)
    assert table.columns == ["class", "support", "correct", "recall"]
    assert table.data == [
        [label, v["support"], v["correct"], v["recall"]]
        for label, v in report["metrics"]["per_class"].items()
    ]
    # A fresh invocation reuses inputs without loading references or decoding images.
    reused = Mock(id="reused", summary={})
    exp.tracker = Tracker(reused)
    with (
        patch.object(exp, "_get_text_inputs", side_effect=AssertionError("read references")),
        patch.object(
            exp.dataset,
            "train_set",
            side_effect=AssertionError("read training metadata"),
        ),
    ):
        exp.generate_descriptions()
        exp.build_vector_database()
    assert reused.summary == {}
    reused.define_metric.assert_not_called()
    reused.log.assert_not_called()


def test_direct_python_is_untracked_and_full_cli_run_shares_tracker(experiment_case, sdk, tmp_path):
    exp = experiment_case.experiment()
    with patch("internvl.cli.wandb.init") as initialize:
        with patch.object(exp.dataset, "train_set", side_effect=AssertionError("tracking count")):
            exp.run()
        initialize.assert_not_called()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(experiment_case.experiment(name="cli").config))
    with patch("internvl.cli.wandb.init", return_value=sdk) as initialize:
        main(["experiment", "run", "--config_file", str(path)])
    initialize.assert_called_once()
    sdk.__enter__.assert_called_once_with()
    sdk.__exit__.assert_called_once_with(None, None, None)
    sdk.finish.assert_not_called()
    assert sdk.summary["evaluation/evaluated"] == 2
    assert [c.args[0].type for c in sdk.log_artifact.call_args_list] == [
        "configuration",
        "evaluation",
    ]


def test_unknown_counts_null_recall_and_saved_report_survives_logging_failure(experiment_case, sdk):
    exp = experiment_case.experiment()
    exp.build_vector_database()
    (exp.output / "index.json").unlink()
    exp.classes.append("Absent")
    exp.tracker = Tracker(sdk)
    failure = RuntimeError("upload failed")
    sdk.log_artifact.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        exp.evaluate()
    assert caught.value is failure
    report = json.loads(next(exp.output.glob("evaluation_*/report.json")).read_text())
    assert report["metrics"]["evaluated"] == 2
    assert sdk.summary["reference_source_counts"] is None
    table = next(
        c.args[0]["evaluation/per_class"]
        for c in sdk.log.call_args_list
        if "evaluation/per_class" in c.args[0]
    )
    assert table.data[-1] == ["Absent", 0, 0, None]


def test_progress_failure_aborts_after_completed_append(experiment_case, sdk):
    exp = experiment_case.experiment((True, True, False))
    exp.tracker = Tracker(sdk)
    failure = RuntimeError("progress")
    sdk.log.side_effect = [None, failure]
    with pytest.raises(RuntimeError) as caught:
        exp.build_vector_database()
    assert caught.value is failure
    assert sdk.summary == {}
    assert experiment_case.client.open_table(exp.config["lancedb"]["table"]).count_rows() == 2
    assert not (exp.output / "index.json").exists()


@pytest.mark.parametrize("total,completed", [(3, [0, 2, 3]), (0, [0])])
def test_progress_payloads_axes_and_no_lifecycle_reporting(sdk, total, completed):
    tracker = Tracker(sdk)
    for count in completed:
        tracker.progress("evaluation", count, total)
    assert sdk.log.call_args_list == [
        call({
            "evaluation/completed": count,
            "evaluation/total": total,
            "evaluation/fraction": count / total if total else 0.0,
        })
        for count in completed
    ]
    assert sdk.define_metric.call_args_list == [
        call("evaluation/completed"),
        call("evaluation/*", step_metric="evaluation/completed"),
    ]
    assert sdk.summary == {}
    assert vars(tracker) == {"run": sdk}


@pytest.mark.parametrize("mode", ["disabled", "offline"])
def test_native_sdk_smoke(tmp_path, mode):
    # Fresh process keeps native global SDK state and environment changes isolated.
    script = r"""
import json
from pathlib import Path
from unittest.mock import patch
import wandb
import yaml
from internvl.cli import _run_experiment
from internvl.tracking import Tracker
report = {"metrics": {"evaluated": 1, "majority_vote_accuracy": 1.,
    "nearest_neighbor_accuracy": 1., "per_class": {
        "A": {"support": 1, "correct": 1, "recall": 1.},
        "B": {"support": 0, "correct": 0, "recall": None}}}}
path = Path("report.json")
path.write_text(json.dumps(report))
config = {"k": 1, "embed_images": False, "embed_class_labels": True,
          "embed_descriptions": False}
original_configuration = Tracker.configuration

def publish(tracker, config):
    assert wandb.run is tracker.run
    log_artifact = tracker.run.log_artifact
    def inspect(artifact):
        assert set(artifact.manifest.entries) == {"resolved_config.yaml"}
        entry = artifact.manifest.entries["resolved_config.yaml"]
        assert yaml.safe_load(Path(entry.local_path).read_text()) == config
        return log_artifact(artifact)
    with patch.object(tracker.run, "log_artifact", side_effect=inspect):
        original_configuration(tracker, config)

class Experiment:
    def __init__(self, config, *, tracker):
        assert wandb.run is tracker.run
        assert dict(tracker.run.config) == config
        self.tracker = tracker

    def generate_descriptions(self):
        raise AssertionError("unexpected stage")

    build_vector_database = generate_descriptions
    run = generate_descriptions

    def evaluate(self):
        tracker = self.tracker
        tracker.progress("evaluation", 0, 1)
        tracker.progress("evaluation", 1, 1)
        tracker.reference_counts(None)
        tracker.evaluation(report, path)

with patch("internvl.cli.Experiment", Experiment), patch.object(Tracker, "configuration", publish):
    _run_experiment(config, "evaluate")
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WANDB_")}
    env.update(
        WANDB_MODE=mode,
        WANDB_DIR=str(tmp_path),
        WANDB_PROJECT="synthetic",
        WANDB_CACHE_DIR=str(tmp_path / "cache"),
        WANDB_DATA_DIR=str(tmp_path / "data"),
        WANDB_CONFIG_DIR=str(tmp_path / "config"),
        CUDA_VISIBLE_DEVICES="",
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == "offline":
        assert list(tmp_path.glob("wandb/offline-run-*/*.wandb"))
        assert not list(tmp_path.rglob("requirements.txt"))
        assert not list(tmp_path.rglob("diff.patch"))
        assert not list(tmp_path.rglob("wandb-metadata.json"))


@pytest.mark.parametrize("failure", [RuntimeError("query failed"), KeyboardInterrupt()])
def test_evaluation_partial_batch_only_advances_when_complete(experiment_case, sdk, failure):
    from datasets import concatenate_datasets

    exp = experiment_case.experiment()
    exp.build_vector_database()
    split = exp.dataset.splits["validation"]
    exp.dataset.splits["validation"] = concatenate_datasets([split, split.select([0])])
    exp.tracker = Tracker(sdk)
    exp.evaluate()
    points = [
        c.args[0]["evaluation/completed"]
        for c in sdk.log.call_args_list
        if "evaluation/completed" in c.args[0]
    ]
    assert points == [0, 2, 3]
    sdk.log.reset_mock()
    reports = list(exp.output.glob("evaluation_*/report.json"))
    with (
        patch("internvl.experiment.LanceDBIndex.query", side_effect=[[], failure]),
        pytest.raises(type(failure)) as caught,
    ):
        exp.evaluate()
    assert caught.value is failure
    points = [
        c.args[0]["evaluation/completed"]
        for c in sdk.log.call_args_list
        if "evaluation/completed" in c.args[0]
    ]
    assert points == [0]
    assert list(exp.output.glob("evaluation_*/report.json")) == reports
    assert not any(key.endswith("/state") for key in sdk.summary)


def test_description_callback_counts_returned_requests_and_propagates_errors():
    from internvl.generate_descriptors import DescriberLLM

    llm = DescriberLLM.__new__(DescriberLLM)
    callback = Mock()
    failure = RuntimeError("request failed")
    with (
        patch.object(
            llm, "describe_class", side_effect=[Mock(description="one"), failure]
        ) as request,
        pytest.raises(RuntimeError) as caught,
    ):
        llm.generate_descriptions(["A"], 3, on_description=callback)
    assert caught.value is failure
    assert request.call_count == 2
    callback.assert_called_once_with(1, 3)
    callback.reset_mock()
    callback.side_effect = failure
    with (
        patch.object(llm, "describe_class", return_value=Mock(description="one")) as request,
        pytest.raises(RuntimeError) as caught,
    ):
        llm.generate_descriptions(["A"], 3, on_description=callback)
    assert caught.value is failure
    request.assert_called_once_with("A")
    callback.assert_called_once_with(1, 3)


def test_description_callback_counts_across_classes_before_validation():
    from internvl.generate_descriptors import DescriberLLM

    llm = DescriberLLM.__new__(DescriberLLM)
    callback = Mock()
    with patch.object(llm, "describe_class", return_value=Mock(description="one")):
        assert llm.generate_descriptions(["A", "B"], 2, on_description=callback) == {
            "A": ["one", "one"], "B": ["one", "one"],
        }
    assert callback.call_args_list == [call(1, 4), call(2, 4), call(3, 4), call(4, 4)]
    callback.reset_mock()
    with (
        patch.object(llm, "describe_class", return_value=Mock(description=" ")),
        pytest.raises(ValueError, match="Empty generated description for A"),
    ):
        llm.generate_descriptions(["A", "B"], 2, on_description=callback)
    assert callback.call_args_list == [call(1, 4), call(2, 4)]


@pytest.mark.parametrize("limit", [None, 1, 2, 20])
@pytest.mark.parametrize("images_enabled", [False, True])
def test_capped_reference_progress_matches_inserted_records(
    experiment_case, sdk, limit, images_enabled
):
    experiment_case.interleaved_training()
    exp = experiment_case.experiment((images_enabled, True, True))
    exp.config["n_imgs_per_class"] = limit
    exp.batch_size = 3
    exp.tracker = Tracker(sdk)
    exp.generate_descriptions()
    artifact = exp.build_vector_database()
    points = [
        call.args[0]
        for call in sdk.log.call_args_list
        if "reference_embedding/completed" in call.args[0]
    ]
    expected_images = {None: 5, 1: 2, 2: 4, 20: 5}[limit] if images_enabled else 0
    expected_total = 12 + expected_images
    assert sum(artifact["source_counts"].values()) == expected_total
    assert all(point["reference_embedding/total"] == expected_total for point in points)
    assert points[-1]["reference_embedding/completed"] == expected_total
    assert points[-1]["reference_embedding/fraction"] == 1.0
    expected_completed = [0, 3, 6, 9, 12]
    expected_completed += [12 + n for n in range(3, expected_images, 3)]
    if expected_images:
        expected_completed.append(expected_total)
    assert [p["reference_embedding/completed"] for p in points] == expected_completed


@pytest.mark.parametrize("images", [False, True])
@pytest.mark.parametrize("failure", [RuntimeError("insert failed"), KeyboardInterrupt()])
def test_failed_reference_batch_does_not_advance(experiment_case, sdk, images, failure):
    exp = experiment_case.experiment((images, not images, False))
    exp.batch_size = 1
    exp.tracker = Tracker(sdk)
    with (
        patch("internvl.experiment.LanceDBIndex.append", side_effect=[None, failure]),
        pytest.raises(type(failure)) as caught,
    ):
        exp.build_vector_database()
    assert caught.value is failure
    assert [c.args[0]["reference_embedding/completed"] for c in sdk.log.call_args_list] == [0, 1]
    assert sdk.summary == {}
    assert not (exp.output / "index.json").exists()


@pytest.mark.parametrize("source", ["backend", "logging"])
@pytest.mark.parametrize("failure", [RuntimeError("description failed"), KeyboardInterrupt()])
def test_description_failure_has_no_cleanup_progress(experiment_case, sdk, source, failure):
    exp = experiment_case.experiment((False, False, True))
    exp.tracker = Tracker(sdk)
    if source == "logging":
        sdk.log.side_effect = [None, failure]
    with patch.object(experiment_case.Describer, "describe_class") as request:
        request.side_effect = [Mock(description="one"), failure]
        with pytest.raises(type(failure)) as caught:
            exp.generate_descriptions()
    assert caught.value is failure
    assert [c.args[0]["descriptions/completed"] for c in sdk.log.call_args_list] == [0, 1]
    assert request.call_count == (1 if source == "logging" else 2)
    assert sdk.summary == {}
    assert not (exp.output / "descriptions.json").exists()


def test_truncated_v4_cli_build_and_evaluation(experiment_case, sdk, tmp_path):
    from sentence_transformers import SentenceTransformer
    from test_embeddings import SyntheticEncoder
    from internvl.embeddings import EmbeddingModel

    exp = experiment_case.experiment((True, True, True), 'truncated-v4')
    exp.config['embedding'].update(
        model_id='jinaai/jina-embeddings-v4', device='cpu', truncate_dim=None,
    )
    path = tmp_path / 'truncated.yaml'
    path.write_text(yaml.safe_dump(exp.config))
    experiment_case.embedder_constructor.side_effect = EmbeddingModel
    encoder = SyntheticEncoder(2048)
    backend = SentenceTransformer(
        modules=[encoder], device='cpu', prompts={'passage': 'Passage: '},
    )
    backend.module_kwargs = {'0': ['task', 'truncate_dim']}
    with patch('internvl.embeddings._load_backend', return_value=backend):
        main(['experiment', 'run', '--config_file', str(path),
              '--set', 'embedding.truncate_dim=1024'])
    experiment_case.embedder_constructor.assert_called_once_with(
        model_id='jinaai/jina-embeddings-v4', device='cpu', batch_size=2,
        truncate_dim=1024,
    )
    table = experiment_case.client.open_table(exp.config['lancedb']['table'])
    assert table.schema.field('vector').type.list_size == 1024
    assert table.count_rows() == 14
    report = json.loads(next(exp.output.glob('evaluation_*/report.json')).read_text())
    assert report['metrics']['evaluated'] == 2
    assert report['config']['embedding']['truncate_dim'] == 1024
    from internvl.cli import wandb
    assert wandb.init.call_args.kwargs['config'] == report['config']
    artifacts = [c.args[0] for c in sdk.log_artifact.call_args_list]
    config_artifact = next(a for a in artifacts if 'resolved_config.yaml' in a.files)
    assert yaml.safe_load(config_artifact.files['resolved_config.yaml']) == report['config']
