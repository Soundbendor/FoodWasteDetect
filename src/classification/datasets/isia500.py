"""ISIA Food-500 loader and split-zip download."""

import shutil
import subprocess
from pathlib import Path

import httpx
import pandas as pd

from classification.datasets.base import FoodImageDataset, spaced
from classification.datasets.util import extract, fetch_archive

BASE_URL = "http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/"
# Location assumed next to the image parts; unverified.
METADATA_URL = BASE_URL + "metadata_ISIAFood_500.zip"
PARTS = (
    *(f"ISIA_Food500.z{i:02d}" for i in range(1, 11)),
    "ISIA_Food500.zip",
)
META_DIR = "metadata_ISIAFood_500"
IMAGE_DIR = "images"
CLASS_FILE = "ISIAFood500_classLabel.txt"


class ISIAFood500ImageDataset(FoodImageDataset):
    """ISIA Food-500: official ``train_finetune`` references, ``test_finetune`` evaluation.

    ``META_DIR/ISIAFood500_classLabel.txt`` holds ``name<TAB>id`` lines. The split lists
    hold ``<class>/<file> <id>`` lines relative to ``IMAGE_DIR``; labels come from the ID.
    Names only have underscores replaced, keeping case, parentheses, and non-ASCII text.
    """

    name = "isia500"

    def _read_classes(self) -> list[str]:
        table = pd.read_csv(
            self.root / META_DIR / CLASS_FILE,
            sep="\t",
            names=["name", "id"],
            dtype={"name": str, "id": "int64"},
            keep_default_na=False,
        )
        labels = table["name"].map(spaced)
        if table.empty or table["id"].duplicated().any() or labels.duplicated().any():
            raise ValueError(f"{CLASS_FILE} must list unique IDs and canonical labels")
        self.class_map = dict(zip(table["id"], labels, strict=True))
        return list(labels)

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        lines = (self.root / META_DIR / f"{source}_finetune.txt").read_text().splitlines()
        rows = [line.rsplit(maxsplit=1) for line in lines if line.strip()]
        ids = [int(class_id) for _, class_id in rows]
        if not set(ids) <= self.class_map.keys():
            raise ValueError(f"{source}_finetune.txt names a class ID missing from {CLASS_FILE}")
        table = pd.DataFrame(
            {"fname": [path for path, _ in rows], "class": [self.class_map[i] for i in ids]}
        )
        return source, self.root / IMAGE_DIR, table

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch the metadata zip and eleven split-zip parts, then unpack them under ``root``.

        Each part must match its ``Content-Length``. ``7z x`` unpacks the images into
        ``root``; the metadata moves into ``META_DIR`` last, so the class-label file marks
        a complete download. Archives are deleted afterwards.
        """
        root = Path(root).expanduser().resolve()
        if (root / META_DIR / CLASS_FILE).exists():
            return
        sevenzip = shutil.which("7z")
        if sevenzip is None:
            raise RuntimeError(
                "7z is not on PATH; run within the Pixi environment, which provides p7zip"
            )
        archives = [
            fetch_archive(url, root, client, require_length=True)
            for url in (METADATA_URL, *(BASE_URL + part for part in PARTS))
        ]
        metadata, *parts = archives
        subprocess.run([sevenzip, "x", "-y", f"-o{root}", str(parts[-1])], check=True)
        staging = metadata.with_name(metadata.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        extract(metadata, staging)
        (label_file,) = staging.rglob(CLASS_FILE)
        destination = root / META_DIR
        destination.mkdir(exist_ok=True)
        for entry in sorted(label_file.parent.iterdir(), key=lambda p: p.name == CLASS_FILE):
            entry.replace(destination / entry.name)
        shutil.rmtree(staging)
        for archive in archives:
            archive.unlink()
