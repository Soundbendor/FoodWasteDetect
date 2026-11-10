"""Parse commands and merge experiment configuration overrides."""

import argparse
import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import wandb
import yaml

from classification.datasets.image_dataset import load_dataset
from classification.experiments.classification import Experiment
from classification.experiments.config import ExperimentConfig, OllamaConfig
from classification.experiments.matroyshka import run_sweep
from classification.tracking import Tracker

# Method names, not bound methods, so tests that replace cli.Experiment keep working.
STAGES = {
    "describe": "generate_descriptions",
    "index": "build_vector_database",
    "evaluate": "evaluate",
    "run": "run",
}


def download_model(model_name: str) -> None:
    """Pull a model using the working directory's legacy Ollama configuration.

    Read ``cfg/experiment_config_internvl.yml`` and close the owned server when
    the pull completes or fails. The requested name overrides ``ollama.llm``.
    """
    from classification.llm.ollama import OllamaSession

    with open("cfg/experiment_config_internvl.yml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    settings = OllamaConfig.model_validate(cfg["ollama"])
    with OllamaSession(settings) as session:
        session.download_model(model_name)


def _parse_override(assignment: str) -> dict[str, Any]:
    """Turn a dotted PATH=VALUE assignment into a nested mapping."""
    path, separator, raw = assignment.partition("=")
    if not separator or not path.strip():
        raise ValueError("expected a nonempty PATH followed by =VALUE")
    value = yaml.safe_load(raw)
    for part in reversed(path.split(".")):
        value = {part: value}
    return value


def _merge_mapping(target: dict, value: dict) -> None:
    for key, item in value.items():
        if isinstance(item, dict) and isinstance(target.get(key), dict):
            _merge_mapping(target[key], item)
        else:
            target[key] = item


def _load_config(config_file: str, overrides: Sequence[str]) -> ExperimentConfig:
    try:
        with open(config_file, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Experiment configuration must be a YAML mapping")
        for assignment in overrides:
            _merge_mapping(config, _parse_override(assignment))
        return ExperimentConfig.model_validate(config)
    except (yaml.YAMLError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classification",
        description="FoodWasteDetect utilities and food image recognition experiments",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser(
        "experiment",
        help="Run a food image classification experiment.",
        allow_abbrev=False,
    )
    experiment.add_argument(
        "stage",
        choices=(*STAGES, "matroyshka-sweep"),
        help="Run a specific stage of the food image classification experiment.",
    )
    experiment.add_argument("--config_file", required=True, help="Path to experiment config YAML")
    experiment.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override a schema field; repeat as needed (e.g. --set embedding.batch_size=16). Values use YAML syntax.",
    )
    download = commands.add_parser(
        "download_model",
        help="Download weights for an Ollama model.",
        allow_abbrev=False,
    )
    download.add_argument("model_name", help="Ollama model name to download")
    return parser


def _run_experiment(config: ExperimentConfig, stage: str) -> None:
    if stage == "matroyshka-sweep":
        dataset = load_dataset(config.dataset)

        def evaluate_cached(effective: ExperimentConfig, tracker: Tracker) -> Any:
            experiment = Experiment(effective, tracker=tracker, dataset=dataset)
            with contextlib.closing(experiment):
                return experiment.evaluate_cached()

        run_sweep(config, lambda effective: _tracked_run(effective, stage, evaluate_cached))
        return

    def execute(effective: ExperimentConfig, tracker: Tracker) -> Any:
        with contextlib.closing(Experiment(effective, tracker=tracker)) as experiment:
            return getattr(experiment, STAGES[stage])()

    _tracked_run(config, stage, execute)


def _tracked_run(
    config: ExperimentConfig,
    stage: str,
    execute: Callable[[ExperimentConfig, Tracker], Any],
) -> tuple[Any, str]:
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


def main(argv: Sequence[str] | None = None) -> None:
    """Parse Ollama download or whole-image experiment commands and dispatch them."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "experiment":
        try:
            config = _load_config(args.config_file, args.overrides)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        _run_experiment(config, args.stage)
    elif args.command == "download_model":
        download_model(args.model_name)
