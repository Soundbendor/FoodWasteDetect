"""Parse commands and merge experiment configuration overrides."""

import argparse
from collections.abc import Sequence
from typing import Any, cast

import yaml
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .experiment import Experiment, validate_config
from .tracking import track_invocation


def download_model(model_name: str) -> None:
    import yaml

    from .generate_descriptors import DescriberLLM

    with open("cfg/experiment_config_internvl.yml", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    with DescriberLLM(cfg["ollama"]) as llm:
        llm.download_model(model_name)


def _override_assignment(value: str) -> str:
    path, separator, _ = value.partition("=")
    if not separator or not path.strip():
        raise argparse.ArgumentTypeError("expected a nonempty PATH followed by =VALUE")
    return value


def _load_config(config_file: str, overrides: Sequence[str]) -> tuple[dict[str, Any], str]:
    with open(config_file, encoding="utf-8", newline="") as stream:
        source_yaml = stream.read()
        config = OmegaConf.create(source_yaml)
    if not isinstance(config, DictConfig):
        raise ValueError("Experiment configuration must be a YAML mapping")
    OmegaConf.set_struct(config, True)
    try:
        config.merge_with_dotlist(list(overrides))
    except (OmegaConfBaseException, yaml.YAMLError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True)), source_yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="internvl",
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
        choices=("describe", "index", "evaluate", "run"),
        help="Run a specific stage of the food image classification experiment.",
    )
    experiment.add_argument("--config_file", required=True, help="Path to experiment config YAML")
    experiment.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        type=_override_assignment,
        metavar="PATH=VALUE",
        help="Override an existing config path; repeat as needed (e.g. --set embedding.batch_size=16). Values use YAML syntax.",
    )
    download = commands.add_parser(
        "download_model",
        help="Download weights for an Ollama model.",
        allow_abbrev=False,
    )
    download.add_argument("model_name", help="Ollama model name to download")
    return parser


def _run_experiment(
    config: dict[str, Any],
    stage: str,
    *,
    config_path: str,
    source_yaml: str,
    overrides: Sequence[str],
) -> None:
    validate_config(config)
    with track_invocation(config, stage, config_path, source_yaml, overrides) as tracker:
        experiment = Experiment(config, tracker=tracker)
        stages = {
            "describe": experiment.generate_descriptions,
            "index": experiment.build_vector_database,
            "evaluate": experiment.evaluate,
            "run": experiment.run,
        }
        stages[stage]()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse Ollama download or whole-image experiment commands and dispatch them."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "experiment":
        try:
            config, source_yaml = _load_config(args.config_file, args.overrides)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
        _run_experiment(
            config,
            args.stage,
            config_path=args.config_file,
            source_yaml=source_yaml,
            overrides=args.overrides,
        )
    elif args.command == "download_model":
        download_model(args.model_name)
