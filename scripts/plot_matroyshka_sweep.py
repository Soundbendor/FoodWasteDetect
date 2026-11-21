"""Plot majority-vote validation accuracy from Matryoshka evaluation reports.

Pass one submission directory (assets/matroyshka/<run>) or individual model
sweep directories. By default all 28 model/dimension points are required.
"""

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

from matplotlib.figure import Figure

MODELS = {
    "Qwen/Qwen3-VL-Embedding-8B": ("Qwen3-VL-Embedding-8B", 4096),
    "jinaai/jina-embeddings-v4": ("Jina Embeddings v4", 2048),
    "google/embeddinggemma-2": ("EmbeddingGemma2", 768),
    "google/ge-v2": ("Gemini Embedding 2", 3072),
}
DIMENSIONS = (128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096)


@dataclass(frozen=True)
class Point:
    model: str
    dimension: int
    accuracy: float
    report: Path


def read_points(inputs: list[Path], *, allow_partial: bool = False) -> list[Point]:
    """Read reports, rejecting duplicate points and incompatible experiment settings."""
    paths = set()
    for path in inputs:
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")
        paths.update(
            p.resolve()
            for p in (path.rglob("report.json") if path.is_dir() else [path])
        )
    points = {}
    comparison = None
    for path in sorted(paths):
        report = json.loads(path.read_text())
        config = report["config"]
        model = config["embedding"]["model_id"]
        if model not in MODELS:
            continue
        dimension = config.get("vector_dimension") or config["embedding"].get(
            "truncate_dim"
        )
        if (
            type(dimension) is not int
            or dimension not in DIMENSIONS
            or dimension > MODELS[model][1]
        ):
            raise ValueError(
                f"Missing or unsupported embedding dimension in {path}: {dimension}"
            )
        accuracy = report["metrics"]["majority_vote_accuracy"]
        if (
            type(accuracy) not in (int, float)
            or not math.isfinite(accuracy)
            or not 0 <= accuracy <= 1
        ):
            raise ValueError(f"Invalid majority_vote_accuracy in {path}: {accuracy}")
        key = (model, dimension)
        if key in points:
            raise ValueError(
                f"Duplicate {model} at {dimension} dimensions: {points[key].report} and {path}"
            )
        # Model prompts and table names may differ; the evaluation design must match.
        settings = (
            config["dataset"],
            config["k"],
            config["embed_images"],
            config["embed_class_labels"],
            config["embed_descriptions"],
            config.get("n_imgs_per_class"),
            report["metrics"]["evaluated"],
        )
        if settings[-1] <= 0:
            raise ValueError(f"No evaluated images in {path}")
        if comparison is not None and settings != comparison:
            raise ValueError(
                f"Dataset, reference settings, k, or evaluated count differs in {path}"
            )
        comparison = settings
        points[key] = Point(model, dimension, float(accuracy), path)
    if not points:
        raise ValueError("No reports found for the four selected models")
    missing = [
        f"{label}: {[d for d in DIMENSIONS if d <= width and (model, d) not in points]}"
        for model, (label, width) in MODELS.items()
        if any((model, d) not in points for d in DIMENSIONS if d <= width)
    ]
    if missing:
        message = "Missing dimensions — " + "; ".join(missing)
        if not allow_partial:
            raise ValueError(message + ". Wait for the sweeps, or use --allow-partial.")
        warnings.warn(message, stacklevel=2)
    return sorted(
        points.values(), key=lambda p: (list(MODELS).index(p.model), p.dimension)
    )


def plot(points: list[Point], output: Path, *, partial: bool = False) -> None:
    """Save a headless Matplotlib figure and a CSV with source-report provenance."""
    figure = Figure(figsize=(12, 6), layout="constrained")
    axes = figure.subplots()
    for (model, (label, _)), color, marker in zip(
        MODELS.items(),
        ("#0072B2", "#D55E00", "#009E73", "#CC79A7"),
        ("o", "s", "^", "D"),
        strict=True,
    ):
        series = [point for point in points if point.model == model]
        if series:
            axes.plot(
                [point.dimension for point in series],
                [100 * point.accuracy for point in series],
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
            )
    axes.set_xlabel("Embedding dimension")
    axes.set_ylabel("Majority-vote validation accuracy (%)")
    axes.set_title(
        "Accuracy vs. embedding dimension" + (" (partial results)" if partial else "")
    )
    axes.set_xticks(DIMENSIONS)
    axes.tick_params(axis="x", labelrotation=45)
    axes.set_ylim(65, 85)
    axes.grid(alpha=0.25)
    axes.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model_id", "embedding_dimension", "majority_vote_accuracy", "report"]
        )
        writer.writerows(
            (p.model, p.dimension, p.accuracy, str(p.report)) for p in points
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path, help="Sweep directories or report.json files"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("assets/plots/matroyshka_accuracy.png")
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Plot available points and warn about missing ones",
    )
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        parser.error("Output must end in .png, .svg, or .pdf")
    try:
        points = read_points(args.inputs, allow_partial=args.allow_partial)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    expected = sum(sum(d <= width for d in DIMENSIONS) for _, width in MODELS.values())
    plot(points, args.output, partial=len(points) < expected)
    print(
        f"Saved {args.output} and {args.output.with_suffix('.csv')} ({len(points)} points)"
    )


if __name__ == "__main__":
    main()
