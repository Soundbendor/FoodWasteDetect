"""Plot majority-vote validation accuracy against embedding dimension.

Reads the results.csv that scripts/collect_results.py writes for one sweep
submission. Runs marked missing are left out and the title says so.
"""

import argparse
import csv
from pathlib import Path

from matplotlib.figure import Figure

LABELS = {
    "Qwen/Qwen3-VL-Embedding-8B": "Qwen3-VL-Embedding-8B",
    "jinaai/jina-embeddings-v4": "Jina Embeddings v4",
    "google/embeddinggemma-2": "EmbeddingGemma2",
    "google/ge-v2": "Gemini Embedding 2",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


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


def plot(rows: list[dict], output: Path) -> None:
    """Save one curve per model; colors follow manifest order, including missing models."""
    figure = Figure(figsize=(12, 6), layout="constrained")
    axes = figure.subplots()
    models = list(dict.fromkeys(row["model_id"] for row in rows))
    for index, model in enumerate(models):
        series = sorted(
            (row["dimension"], 100 * row["accuracy"])
            for row in rows
            if row["model_id"] == model and row["status"] == "done"
        )
        if series:
            axes.plot(
                *zip(*series, strict=True),
                label=LABELS.get(model, model),
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[index % len(MARKERS)],
                linewidth=2,
            )
    partial = any(row["status"] != "done" for row in rows)
    axes.set_xlabel("Embedding dimension")
    axes.set_ylabel("Majority-vote validation accuracy (%)")
    axes.set_title("Accuracy vs. embedding dimension" + (" (partial results)" if partial else ""))
    axes.set_xticks(sorted({row["dimension"] for row in rows}))
    axes.tick_params(axis="x", labelrotation=45)
    axes.set_ylim(65, 85)
    axes.grid(alpha=0.25)
    axes.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


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
