"""Weights and Biases monitoring and tracking for image recognition experiments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import wandb
import yaml

from classification.experiments.config import ExperimentConfig
from classification.experiments.report_types import EvaluationReport


class Tracker:
    """Publish configuration and results through a caller-owned W&B run."""

    def __init__(self, run: wandb.Run) -> None:
        self.run = run

    def configuration(self, config: ExperimentConfig) -> None:
        """Publish the resolved configuration, including overrides."""
        artifact = wandb.Artifact(f"configuration-{self.run.id}", type="configuration")
        with artifact.new_file("resolved_config.yaml", encoding="utf-8") as stream:
            yaml.safe_dump(config.model_dump(mode="json"), stream, sort_keys=False)
        self.run.log_artifact(artifact)

    def reference_counts(self, counts: dict[str, int] | None) -> None:
        """Publish source counts, or null when no matching index summary exists."""
        self.run.summary["reference_source_counts"] = counts

    def evaluation(self, report: EvaluationReport, report_path: Path) -> None:
        """Publish completed metrics and the report already saved at ``report_path``."""
        metrics = report.metrics
        self.run.summary.update(
            {f"evaluation/{key}": value for key, value in metrics.items() if key != "per_class"}
        )
        self.run.log(
            {
                "evaluation/per_class": wandb.Table(
                    columns=["class", "support", "correct", "recall", "precision", "f1"],
                    data=[
                        [
                            label,
                            values["support"],
                            values["correct"],
                            values["recall"],
                            values["precision"],
                            values["f1"],
                        ]
                        for label, values in metrics["per_class"].items()
                    ],
                )
            }
        )
        artifact = wandb.Artifact(f"evaluation-{self.run.id}", type="evaluation")
        artifact.add_file(str(report_path), name="report.json")
        self.run.log_artifact(artifact)


def tracked_run(
    config: ExperimentConfig,
    stage: str,
    execute: Callable[[ExperimentConfig, Tracker], Any],
) -> tuple[Any, str]:
    """Run ``execute`` inside one native W&B context and return its result and run ID."""
    with wandb.init(
        config=config.model_dump(mode="json"),
        job_type=stage,
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            save_code=False,
            x_disable_meta=True,
            x_save_requirements=False,
        ),
    ) as run:
        tracker = Tracker(run)
        tracker.configuration(config)
        return execute(config, tracker), run.id
