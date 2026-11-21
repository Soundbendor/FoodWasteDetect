"""Merge one group's per-run result records into results.csv.

GROUP_DIR holds manifest.json, listing the expected runs, and results/<run>.json,
written by each completed evaluation. Runs without a record are marked missing.
"""

import argparse
import csv
import json
from pathlib import Path

# Settings every run must share, unless the manifest declares one as a run axis.
SHARED = ("k", "embed_images", "embed_class_labels", "embed_descriptions", "n_imgs_per_class")


def collect(group: Path) -> list[dict]:
    """Return one row per expected run: manifest fields, status, metrics, report."""
    rows, shared = [], None
    for run in json.loads((group / "manifest.json").read_text())["runs"]:
        path = group / "results" / f"{run['run']}.json"
        if not path.exists():
            rows.append({**run, "status": "missing"})
            continue
        record = json.loads(path.read_text())
        config, metrics = record["config"], record["metrics"]
        settings = {key: config[key] for key in SHARED if key not in run}
        settings["dataset"] = config["dataset"]["name"]
        settings["evaluated"] = metrics["evaluated"]
        if shared is not None and settings != shared:
            raise ValueError(f"{path} differs from earlier runs: {settings} != {shared}")
        shared = settings
        rows.append({**run, "status": "done", **metrics, "report": record["report"]})
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", type=Path, help="Submission directory with manifest.json")
    parser.add_argument(
        "--allow-partial", action="store_true", help="Exit zero when some runs are missing"
    )
    args = parser.parse_args(argv)
    try:
        rows = collect(args.group)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    missing = [row["run"] for row in rows if row["status"] == "missing"]
    if len(missing) == len(rows):
        parser.exit(1, f"No results in {args.group / 'results'}\n")
    output = args.group / "results.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output} ({len(rows) - len(missing)} of {len(rows)} runs)")
    if missing:
        print("Missing: " + ", ".join(missing))
        if not args.allow_partial:
            parser.exit(1, "Wait for the remaining runs, or use --allow-partial.\n")


if __name__ == "__main__":
    main()
