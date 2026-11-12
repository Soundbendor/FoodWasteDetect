"""VegFru loader and Hugging Face download."""

from pathlib import Path
from typing import ClassVar

import httpx
import pandas as pd

from classification.datasets.archives import fetch_archive, install_archive
from classification.datasets.base import FoodImageDataset, canonical


class VegFruImageDataset(FoodImageDataset):
    """VegFru (200 vegetable + 92 fruit classes): official train references, test evaluation.

    ``classid_classname.csv`` holds the taxonomy. ``LIST_DIR/vegfru_{train,test}.txt``
    hold ``<relative path> <label id>`` lines; labels come from the path's class
    directory, and the label ID is not read. The official val list is not read.
    """

    name = "vegfru"
    LIST_DIR: ClassVar[str] = "vegfru_list"

    def _read_classes(self) -> list[str]:
        table = pd.read_csv(
            self.root / "classid_classname.csv",
            dtype={"class_id": "int64", "class_name": str},
            keep_default_na=False,
        )
        labels = table["class_name"].map(canonical)
        if table.empty or table["class_id"].duplicated().any() or labels.duplicated().any():
            raise ValueError("classid_classname.csv must list unique IDs and canonical labels")
        return list(labels)

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        lines = (self.root / self.LIST_DIR / f"vegfru_{source}.txt").read_text().splitlines()
        paths = [line.rsplit(maxsplit=1)[0] for line in lines if line.strip()]
        table = pd.DataFrame(
            {"fname": paths, "class": [canonical(Path(p).parent.name) for p in paths]}
        )
        if not set(table["class"]) <= set(self.classes):
            raise ValueError(
                f"vegfru_{source}.txt names a class missing from classid_classname.csv"
            )
        return source, self.root, table

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch ``vegfru.tar.gz`` from Hugging Face and move its ``vegfru/`` contents into ``root``.

        The nested ``archive.zip`` is unpacked before moving; ``classid_classname.csv``
        moves last and marks a complete download.
        """
        root = Path(root).expanduser().resolve()
        if (root / "classid_classname.csv").exists():
            return
        url = "https://huggingface.co/datasets/xedwin23x/VegFru/resolve/main/vegfru.tar.gz"
        install_archive(
            fetch_archive(url, root, client),
            root,
            "vegfru",
            "classid_classname.csv",
            nested="archive.zip",
        )
