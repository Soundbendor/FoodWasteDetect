"""Command dispatch without importing model, dataset, or service backends."""

import argparse
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="internvl",
        description="FoodWasteDetect caption and text-retrieval experiments",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    workflows = (
        ("caption-images", "caption_imgs", True,
         "Caption FoodX251 training images"),
        ("generate-descriptions", "generate_descriptions", True,
         "Describe food class names through Ollama"),
        ("build-db", "build_vdb", True,
         "Embed and index reference descriptions"),
        ("evaluate", "test_classification", True,
         "Evaluate caption-based image classification"),
        ("finetune-embeddings", "finetune_embeddings", True,
         "Fine-tune the text embedding model"),
        ("merge-captions", "merge_files", False,
         "Merge the four FoodX251 caption partitions"),
        ("convert-captions", "convert_csv_to_dictionary", False,
         "Convert merged captions to a description dictionary"),
    )
    for command, workflow, needs_config, help_text in workflows:
        subparser = commands.add_parser(command, help=help_text, description=help_text)
        subparser.set_defaults(workflow=workflow, needs_config=needs_config)
        if needs_config:
            subparser.add_argument(
                "--config_file", required=True, help="Path to experiment YAML"
            )
        if command == "caption-images":
            subparser.add_argument(
                "--partition", required=True, type=int,
                help="Legacy FoodX251 partition number, 1 through 4",
            )

    args = parser.parse_args(argv)

    # Help and argument errors return before loading any runtime dependencies.
    from . import workflows as experiments

    kwargs = {}
    if args.needs_config:
        from util import parse_cfg

        kwargs["cfg"] = parse_cfg(args.config_file)
    if args.command == "caption-images":
        kwargs["partition"] = args.partition
    getattr(experiments, args.workflow)(**kwargs)
