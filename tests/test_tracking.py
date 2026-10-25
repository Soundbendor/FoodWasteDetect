"""Synthetic W&B contracts: fake SDK publishing plus isolated native offline runs."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import test_classification
import yaml
from internvl.cli import main
from internvl.tracking import StageProgress, Tracker


class Artifact:
    def __init__(self, name, type):
        self.name, self.type = name, type
        self.files = {}

    def add_file(self, path, *, name):
        self.files[name] = Path(path).read_bytes()


@pytest.fixture
def sdk(monkeypatch):
    run = Mock(id="synthetic", summary={})
    monkeypatch.setattr("internvl.tracking.wandb.init", Mock(return_value=run))
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
        patch("internvl.tracking.wandb.init", return_value=sdk) as initialize,
    ):
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
    sdk.finish.assert_called_once_with(exit_code=0)
    artifact = sdk.log_artifact.call_args.args[0]
    assert artifact.name == "configuration-synthetic"
    assert artifact.files["source_config.yaml"] == source.encode()
    assert yaml.safe_load(artifact.files["resolved_config.yaml"]) == cfg
    assert json.loads(artifact.files["invocation.json"]) == {
        "stage": stage,
        "config_path": str(path),
        "overrides": overrides,
    }
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
        patch("internvl.tracking.wandb.init") as initialize,
        patch("internvl.cli.download_model"),
    ):
        if kind == "download":
            main(args)
        else:
            with pytest.raises((SystemExit, ValueError)):
                main(args)
        initialize.assert_not_called()


@pytest.mark.parametrize(
    "source", ["init", "artifact", "constructor", "stage", "interrupt", "finish"]
)
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_lifecycle_errors_preserve_original(sdk, tmp_path, source, cleanup_fails):
    path, _ = config_file(tmp_path)
    failure = KeyboardInterrupt() if source == "interrupt" else RuntimeError(source)
    with (
        patch("internvl.cli.Experiment") as experiment,
        patch("internvl.tracking.wandb.init", return_value=sdk) as initialize,
    ):
        if cleanup_fails:
            sdk.finish.side_effect = RuntimeError("cleanup")
        if source == "init":
            initialize.side_effect = failure
        elif source == "artifact":
            sdk.log_artifact.side_effect = failure
        elif source == "constructor":
            experiment.side_effect = failure
        elif source in ("stage", "interrupt"):
            experiment.return_value.run.side_effect = failure
        else:
            sdk.finish.side_effect = failure
        with pytest.raises(BaseException) as caught:
            main(["experiment", "run", "--config_file", str(path)])
        assert caught.value is failure
        if source == "init":
            sdk.finish.assert_not_called()
        else:
            sdk.finish.assert_called_once_with(exit_code=0 if source == "finish" else 1)
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
        assert sdk.summary[f"{stage}/state"] == "completed"
        assert sdk.summary[f"{stage}/throughput"] >= 0
    if flags[2]:
        assert [
            p["descriptions/completed"] for p in history if "descriptions/completed" in p
        ] == list(range(11))
    else:
        assert sdk.summary["descriptions/state"] == "disabled"
        assert "descriptions/throughput" not in sdk.summary
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
    assert reused.summary["reference_embedding/state"] == "reused"
    assert reused.summary["descriptions/state"] == ("reused" if flags[2] else "disabled")
    assert not any("throughput" in key for key in reused.summary)
    reused.log.assert_not_called()


def test_direct_python_is_untracked_and_full_cli_run_shares_tracker(experiment_case, sdk, tmp_path):
    exp = experiment_case.experiment()
    with patch("internvl.tracking.wandb.init") as initialize:
        exp.run()
        initialize.assert_not_called()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(experiment_case.experiment(name="cli").config))
    with patch("internvl.tracking.wandb.init", return_value=sdk) as initialize:
        main(["experiment", "run", "--config_file", str(path)])
    initialize.assert_called_once()
    sdk.finish.assert_called_once_with(exit_code=0)
    assert (
        sdk.summary["reference_embedding/state"] == sdk.summary["evaluation/state"] == "completed"
    )
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
    assert sdk.summary["reference_embedding/state"] == "failed"
    assert experiment_case.client.open_table(exp.config["lancedb"]["table"]).count_rows() == 2
    assert not (exp.output / "index.json").exists()


def test_stage_timing_axes_and_cleanup_preserve_interrupt(sdk):
    with (
        patch("internvl.tracking.perf_counter", side_effect=[10.0, 14.0]),
        StageProgress(Tracker(sdk), "evaluation") as progress,
    ):
        progress.start(3)
        progress.advance(2)
        progress.advance(1)
    assert sdk.summary["evaluation/elapsed_seconds"] == 4
    assert sdk.summary["evaluation/throughput"] == 0.75
    sdk.define_metric.assert_any_call("evaluation/*", step_metric="evaluation/completed")
    assert all(not c.kwargs for c in sdk.log.call_args_list)
    sdk.summary = Mock()
    sdk.summary.update.side_effect = RuntimeError("cleanup")
    failure = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught, StageProgress(Tracker(sdk), "evaluation"):
        raise failure
    assert caught.value is failure
    assert "cleanup" in failure.__notes__[0]


@pytest.mark.parametrize("mode", ["disabled", "offline"])
def test_native_sdk_smoke(tmp_path, mode):
    # Fresh process keeps native global SDK state and environment changes isolated.
    script = r"""
import json
from pathlib import Path
from internvl.tracking import StageProgress, track_invocation
report = {"metrics": {"evaluated": 1, "majority_vote_accuracy": 1.,
    "nearest_neighbor_accuracy": 1., "per_class": {
        "A": {"support": 1, "correct": 1, "recall": 1.},
        "B": {"support": 0, "correct": 0, "recall": None}}}}
path = Path("report.json")
path.write_text(json.dumps(report))
with track_invocation({"k": 1}, "evaluate", "config.yaml", "# source\nk: 1\n", []) as tracker:
    assert dict(tracker.run.config) == {"k": 1}
    with StageProgress(tracker, "evaluation") as progress:
        progress.start(1)
        progress.advance()
    tracker.reference_counts(None)
    tracker.evaluation(report, path)
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
        patch("internvl.experiment.LanceDBIndex.query", side_effect=failure),
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
    assert sdk.summary["evaluation/state"] == (
        "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
    )


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
    callback.assert_called_once_with()
    callback.side_effect = failure
    with (
        patch.object(llm, "describe_class", return_value=Mock(description="one")) as request,
        pytest.raises(RuntimeError) as caught,
    ):
        llm.generate_descriptions(["A"], 3, on_description=callback)
    assert caught.value is failure
    request.assert_called_once_with("A")
