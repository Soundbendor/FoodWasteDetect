"""Explicit W&B ownership and stage progress for the experiment CLI."""

from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import wandb
import yaml


@contextmanager
def track_invocation(
    config: dict[str, Any],
    stage: str,
    config_path: str,
    source_yaml: str,
    overrides: Sequence[str],
) -> Generator[Tracker]:
    """Own one SDK run; preserve the primary failure if finishing also fails."""
    run = wandb.init(
        config=config,
        job_type=stage,
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            save_code=False,
            x_disable_meta=True,
            x_save_requirements=False,
        ),
    )
    tracker = Tracker(run)
    try:
        tracker.configuration(
            config,
            source_yaml,
            {
                "stage": stage,
                "config_path": config_path,
                "overrides": list(overrides),
            },
        )
        yield tracker
    except BaseException as error:
        try:
            run.finish(exit_code=1)
        except BaseException as cleanup_error:
            error.add_note(f"W&B cleanup also failed: {cleanup_error!r}")
        raise
    else:
        run.finish(exit_code=0)


class Tracker:
    """Publish only explicit experiment data through an owned run object."""

    def __init__(self, run: wandb.Run):
        self.run = run

    def configuration(self, config: dict, source_yaml: str, invocation: dict) -> None:
        artifact = wandb.Artifact(f"configuration-{self.run.id}", type="configuration")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = {
                "source_config.yaml": source_yaml,
                "resolved_config.yaml": yaml.safe_dump(config, sort_keys=False),
                "invocation.json": json.dumps(invocation, indent=2) + "\n",
            }
            for name, content in files.items():
                path = root / name
                path.write_text(content, encoding="utf-8")
                artifact.add_file(str(path), name=name)
            self.run.log_artifact(artifact)

    def reference_counts(self, counts: dict | None) -> None:
        self.run.summary["reference_source_counts"] = counts

    def evaluation(self, report: dict, report_path: Path) -> None:
        metrics = report["metrics"]
        self.run.summary.update(
            {
                f"evaluation/{key}": metrics[key]
                for key in ("evaluated", "majority_vote_accuracy", "nearest_neighbor_accuracy")
            }
        )
        self.run.log(
            {
                "evaluation/per_class": wandb.Table(
                    columns=["class", "support", "correct", "recall"],
                    data=[
                        [label, values["support"], values["correct"], values["recall"]]
                        for label, values in metrics["per_class"].items()
                    ],
                )
            }
        )
        artifact = wandb.Artifact(f"evaluation-{self.run.id}", type="evaluation")
        artifact.add_file(str(report_path), name="report.json")
        self.run.log_artifact(artifact)


class StageProgress:
    """Count completed work; a missing tracker keeps direct Python calls untracked."""

    def __init__(self, tracker: Tracker | None, name: str):
        self.tracker = tracker
        self.name = name
        self.completed = 0
        self.total: int | None = None
        self.state = "running"

    def __enter__(self):
        self.started = perf_counter()
        return self

    def start(self, total: int) -> None:
        self.total = total
        if self.tracker:
            run = self.tracker.run
            run.summary[f"{self.name}/state"] = "running"
            run.define_metric(f"{self.name}/completed")
            run.define_metric(f"{self.name}/*", step_metric=f"{self.name}/completed")
            self.advance(0)

    def advance(self, count: int = 1) -> None:
        self.completed += count
        if self.tracker:
            self.tracker.run.log(
                {
                    f"{self.name}/completed": self.completed,
                    f"{self.name}/total": self.total,
                    f"{self.name}/fraction": self.completed / self.total if self.total else 0.0,
                }
            )

    def _finish(self, state: str) -> None:
        if not self.tracker:
            return
        elapsed = perf_counter() - self.started
        summary: dict[str, Any] = {f"{self.name}/state": state}
        if self.state not in ("reused", "disabled"):
            summary[f"{self.name}/elapsed_seconds"] = elapsed
            if self.total is not None:
                summary[f"{self.name}/throughput"] = self.completed / elapsed if elapsed else 0.0
        self.tracker.run.summary.update(summary)

    def __exit__(self, exc_type, error, traceback):
        if error is None:
            self._finish("completed" if self.state == "running" else self.state)
        else:
            try:
                self._finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed")
            except BaseException as cleanup_error:
                error.add_note(f"W&B stage cleanup also failed: {cleanup_error!r}")
