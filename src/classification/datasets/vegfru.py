"""VegFru loader and Hugging Face download."""

from pathlib import Path

import httpx
import pandas as pd

from classification.datasets.base import FoodImageDataset, canonical
from classification.datasets.util import fetch_archive, install_archive

LIST_DIR = "vegfru_list"


def _prepare_images(root: Path) -> None:
    """Normalize the outer archive's complete image tree to official list paths."""
    destinations: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for line in (root / LIST_DIR / f"vegfru_{split}.txt").read_text().splitlines():
            if not line.strip():
                continue
            relative, _ = line.rsplit(maxsplit=1)
            parts = Path(relative).parts
            if (
                len(parts) != 3
                or parts[0] not in {"veg200_images", "fru92_images"}
                or any(p in {".", ".."} for p in parts)
            ):
                raise ValueError(f"Invalid VegFru image path: {relative}")
            prefix, label, filename = parts
            previous = destinations.setdefault(label, prefix)
            if previous != prefix:
                raise ValueError(f"VegFru class occurs in both image groups: {label}")
            source = root / "images" / label / filename
            if not source.is_file():
                raise FileNotFoundError(f"Missing VegFru image: {source}")
    if not destinations:
        raise ValueError("VegFru split lists contain no images")
    # Validate the entire mapping before moving anything. Preserve official paths
    # so the loader's path-derived source identities remain unchanged.
    for label, prefix in destinations.items():
        target = root / prefix / label
        if target.exists():
            raise ValueError(f"Conflicting VegFru class directory: {target}")
    for label, prefix in destinations.items():
        target = root / prefix / label
        target.parent.mkdir(exist_ok=True)
        (root / "images" / label).replace(target)
    (root / "images").rmdir()
    # This redundant ZIP has a reproducible CRC error in the published archive.
    (root / "archive.zip").unlink(missing_ok=True)


class VegFruImageDataset(FoodImageDataset):
    """VegFru (200 vegetable + 92 fruit classes): official train references, test evaluation.

    ``classid_classname.csv`` holds the taxonomy. ``LIST_DIR/vegfru_{train,test}.txt``
    hold ``<relative path> <label id>`` lines; labels come from the path's class
    directory, and the label ID is not read. The official val list is not read.
    """

    name = "vegfru"

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
        lines = (self.root / LIST_DIR / f"vegfru_{source}.txt").read_text().splitlines()
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

        Normalize ``images/`` to the official split paths without opening the
        redundant, corrupt nested ZIP. The class CSV moves last as completion marker.
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
            prepare=_prepare_images,
        )
