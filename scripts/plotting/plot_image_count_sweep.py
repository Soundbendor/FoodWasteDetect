"""Plot FoodX251 majority-vote accuracy from a manifest of full-width reports."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

if __package__:
    from . import style
else:  # Direct script invocation
    import style

CAPS = (1, 5, 10, 20, 50, 100, None)
COUNTS = dict(zip(CAPS, (251, 1255, 2510, 5020, 12534, 25034, 118475), strict=True))
MODELS = {
    "google/embeddinggemma-2": 768,
    "google/ge-v2": 3072,
    "google/siglip2-giant-opt-patch16-384": 1536,
}


def read_grid(manifest, annotation_dir):
    with (annotation_dir / "class_list.txt").open() as handle:
        labels = dict(line.strip().split(maxsplit=1) for line in handle if line.strip())
    with (annotation_dir / "val_info.csv").open() as handle:
        truth = {
            f"foodx251:val:{name}": labels[label].capitalize().replace("_", " ")
            for name, label in csv.reader(handle)
        }
    if len(truth) != 11994:
        raise ValueError("Expected 11,994 unique FoodX251 validation IDs")
    return read_reports(manifest, truth)


def read_reports(manifest, truth):
    points = {}
    grid = {(model, cap) for model in MODELS for cap in CAPS}
    for entry in json.loads(manifest.read_text()):
        model, cap = entry["model"], entry["cap"]
        key = (model, cap)
        if key not in grid or key in points:
            raise ValueError(f"Unexpected or duplicate point {key}")
        path = (manifest.parent / entry["report"]).resolve()
        report = json.loads(path.read_text())
        config = report["config"]
        embedding = config["embedding"]
        width = MODELS[model]
        if not (
            config["dataset"]["name"].lower() == "foodx251"
            and config["k"] == 10
            and config["n_imgs_per_class"] == cap
            and embedding["model_id"] == model
            and config["embed_images"]
            and not config["embed_class_labels"]
            and not config["embed_descriptions"]
            and embedding.get("truncate_dim") in (None, width)
            and config.get("vector_dimension") in (None, width)
            and (
                model != "google/ge-v2" or embedding["vertex"]["classification_prompt"]
            )
            and report["reference_source_counts"] == {"training_image": COUNTS[cap]}
        ):
            raise ValueError(f"Incompatible configuration or reference count in {path}")
        predictions = report["predictions"]
        actual = {
            p["source_id"]: p["class"]
            for p in predictions
            if p["source_split"] == "val"
        }
        if (
            actual != truth
            or len(predictions) != len(truth)
            or report["metrics"]["evaluated"] != len(truth)
        ):
            raise ValueError(f"Incompatible validation predictions in {path}")
        accuracy = report["metrics"]["majority_vote_accuracy"]
        correct = sum(p["prediction"] == p["class"] for p in predictions)
        if not math.isclose(accuracy, correct / len(truth), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Accuracy disagrees with predictions in {path}")
        points[key] = (accuracy, path)
    if grid != points.keys():
        raise ValueError(f"Missing points: {grid - points.keys()}")
    return points


@style.dissertation_style()
def render(points, output, validation_count):
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = style.accuracy_figure()
    with (output / "accuracy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "model",
                "model_id",
                "cap",
                "reference_count",
                "validation_count",
                "majority_vote_accuracy",
                "report",
            )
        )
        for model in MODELS:
            series_style = style.model_style(model)
            values = []
            for cap in CAPS:
                accuracy, path = points[model, cap]
                values.append(accuracy * 100)
                writer.writerow(
                    (
                        series_style["label"],
                        model,
                        cap or "All",
                        COUNTS[cap],
                        validation_count,
                        accuracy,
                        os.path.relpath(path, output),
                    )
                )
            ax.plot(range(len(CAPS)), values, **series_style)
    ax.set(
        xticks=range(len(CAPS)),
        xticklabels=[str(cap or "All") for cap in CAPS],
        xlabel="Training images per class (cap)",
        ylim=(0, 100),
        title="FoodX251 recognition by training images per class",
    )
    style.add_model_legend(fig)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"accuracy.{extension}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--annotation-dir", type=Path, default=Path("data/FoodX-251/annot")
    )
    args = parser.parse_args()
    render(read_grid(args.manifest, args.annotation_dir), args.output_dir, 11994)


if __name__ == "__main__":
    main()
