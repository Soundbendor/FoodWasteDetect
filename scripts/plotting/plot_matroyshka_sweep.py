"""Plot majority-vote validation accuracy against embedding dimension.

Reads the results.csv that scripts/collect_results.py writes for one sweep
submission. Runs marked missing are left out and the title says so.
"""

import argparse
import csv
import math
from pathlib import Path

from matplotlib.ticker import MultipleLocator

if __package__:
    from . import style
else:  # Direct script invocation
    import style


def read_rows(path: Path) -> list[dict]:
    """Return every expected run, with typed dimension and accuracy for completed ones."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["dimension"] = int(row["dimension"])
        if row["status"] == "done":
            row["accuracy"] = float(row["majority_vote_accuracy"])
    if not any(row["status"] == "done" for row in rows):
        raise ValueError(f"No completed runs in {path}")
    return rows


@style.dissertation_style()
def plot(rows: list[dict], output: Path) -> None:
    """Save one curve per model, with gaps at missing scheduled dimensions."""
    figure, axes = style.accuracy_figure()
    models = {row["model_id"] for row in rows}
    for model in sorted(models, key=lambda model: style.model_style(model)["label"]):
        series = sorted(
            (row["dimension"], 100 * row["accuracy"] if row["status"] == "done" else math.nan)
            for row in rows
            if row["model_id"] == model
        )
        dimensions, accuracies = zip(*series, strict=True)
        if any(math.isfinite(value) for value in accuracies):
            axes.plot(dimensions, accuracies, **style.model_style(model))

    completed = [100 * row["accuracy"] for row in rows if row["status"] == "done"]
    lower = max(0, 5 * math.floor((min(completed) - 2) / 5))
    upper = min(100, 5 * math.ceil((max(completed) + 2) / 5))
    axes.set_ylim(lower, upper)
    # Keep wide ranges readable with ticks at multiples of five percentage points.
    axes.yaxis.set_major_locator(MultipleLocator(5 * math.ceil((upper - lower) / 30)))
    partial = any(row["status"] != "done" for row in rows)
    axes.set_xlabel("Embedding dimension")
    axes.set_title("Accuracy vs. embedding dimension" + (" (partial results)" if partial else ""))
    axes.set_xticks(sorted({row["dimension"] for row in rows}))
    axes.tick_params(axis="x", labelrotation=60)
    style.add_model_legend(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results.csv from collect_results.py")
    parser.add_argument(
        "--output", type=Path, default=Path("assets/plots/matroyshka_accuracy.png")
    )
    args = parser.parse_args()
    if args.output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        parser.error("Output must end in .png, .svg, or .pdf")
    try:
        rows = read_rows(args.results)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    plot(rows, args.output)
    done = sum(row["status"] == "done" for row in rows)
    print(f"Saved {args.output} ({done} of {len(rows)} points)")


if __name__ == "__main__":
    main()
