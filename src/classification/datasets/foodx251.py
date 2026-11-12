"""FoodX-251 loader and public S3 download."""

from pathlib import Path

import httpx
import pandas as pd

from classification.datasets.archives import extract, fetch_archive
from classification.datasets.base import FoodImageDataset, canonical


def _read_class_map(path: Path) -> dict[int, str]:
    table = pd.read_csv(
        path,
        sep=r"\s+",
        names=["id", "name"],
        index_col="id",
        dtype={"id": "int64", "name": str},
        keep_default_na=False,
    )
    if table.index.has_duplicates:
        raise ValueError("Duplicate class ID")
    labels = table["name"].map(canonical)
    if labels.empty or labels.duplicated().any():
        raise ValueError("Taxonomy must contain unique canonical labels")
    return labels.to_dict()


class FoodX251ImageDataset(FoodImageDataset):
    """FoodX251 train references and labeled validation; the test split is unlabeled."""

    name = "foodx251"

    def _read_classes(self) -> list[str]:
        self.class_map = _read_class_map(self.root / "annot/class_list.txt")
        return list(self.class_map.values())

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "val"}[split]
        table = pd.read_csv(
            self.root / f"annot/{source}_info.csv",
            names=["fname", "class_id"],
            dtype={"fname": str, "class_id": "int64"},
            keep_default_na=False,
        )
        table["class"] = table.pop("class_id").map(self.class_map.__getitem__)
        return source, self.root / f"{source}/{source}_set", table

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch the public S3 tars; ``annot`` unpacks last and marks completion."""
        root = Path(root).expanduser().resolve()
        if (root / "annot/class_list.txt").exists():
            return
        for part in ("train", "val", "annot"):
            archive = fetch_archive(f"https://food-x.s3.amazonaws.com/{part}.tar", root, client)
            extract(archive, root / part)
            archive.unlink()
