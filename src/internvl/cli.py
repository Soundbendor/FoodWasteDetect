"""Parse commands and load runtime dependencies only when dispatching work."""

import argparse
import yaml
from collections.abc import Sequence

from .experiment import Experiment


def download_model(model_name: str) -> None:
    import yaml

    from .generate_descriptors import DescriberLLM

    with open("cfg/experiment_config_internvl.yml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    with DescriberLLM(cfg["ollama"]) as llm:
        llm.download_model(model_name)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="internvl",
        description="FoodWasteDetect utilities and food image recognition experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser(
        "experiment", help="Run a food image classification experiment."
    )
    experiment.add_argument(
        "stage",
        choices=("describe", "index", "evaluate", "run"),
        help="Run a specific stage of the food image classification experiment.",
    )
    experiment.add_argument(
        "--config_file", required=True, help="Path to experiment config YAML"
    )
    download = commands.add_parser(
        "download_model", help="Download weights for an Ollama model."
    )
    download.add_argument("model_name", help="Ollama model name to download")
    return parser


def _run_experiment(config_file: str, stage: str) -> None:

    with open(config_file, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    experiment = Experiment(config)
    stages = {
        "describe": experiment.generate_descriptions,
        "index": experiment.build_vector_database,
        "evaluate": experiment.evaluate,
        "run": experiment.run,
    }
    stages[stage]()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse Ollama download or whole-image experiment commands and dispatch them."""
    args = _build_parser().parse_args(argv)
    if args.command == "experiment":
        _run_experiment(args.config_file, args.stage)
    elif args.command == "download_model":
        download_model(args.model_name)
