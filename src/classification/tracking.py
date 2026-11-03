"""Weights and Biases monitoring and tracking for image recognition experiments."""

from __future__ import annotations

from pathlib import Path

import wandb
import yaml


class Tracker:
    """Track an experiment, uploading configuration and results artifacts to wandb."""

    def __init__(self, run: wandb.Run):
        self.run = run

    def configuration(self, config: dict) -> None:
        """Publish the resolved configuration, including overrides."""
        artifact = wandb.Artifact(f"configuration-{self.run.id}", type="configuration")
        with artifact.new_file("resolved_config.yaml", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        self.run.log_artifact(artifact)

    def reference_counts(self, counts: dict | None) -> None:
        self.run.summary["reference_source_counts"] = counts

    def evaluation(self, report: dict, report_path: Path) -> None:
        metrics = report["metrics"]
        self.run.summary.update(
            {
                f"evaluation/{key}": metrics[key]
                for key in metrics if key != "per_class"
            }
        )
        self.run.log(
            {
                "evaluation/per_class": wandb.Table(
                    columns=["class", "support", "correct", "recall", "precision", "f1"],
                    data=[
                        [label, values["support"], values["correct"], values["recall"],
                         values["precision"], values["f1"]]
                        for label, values in metrics["per_class"].items()
                    ],
                )
            }
        )
        artifact = wandb.Artifact(f"evaluation-{self.run.id}", type="evaluation")
        artifact.add_file(str(report_path), name="report.json")
        self.run.log_artifact(artifact)
