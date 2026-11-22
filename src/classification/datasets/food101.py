"""Food-101 loader and ETHZ download."""

from pathlib import Path

import httpx
import pandas as pd

from classification.datasets.base import FoodImageDataset, canonical
from classification.datasets.util import fetch_archive, install_archive


class Food101ImageDataset(FoodImageDataset):
    """Food-101 official train references and labeled test split."""

    name = "food101"

    def _read_classes(self) -> list[str]:
        labels = [canonical(n) for n in (self.root / "meta/classes.txt").read_text().split()]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Taxonomy must contain unique canonical labels")
        return labels

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        stems = (self.root / f"meta/{source}.txt").read_text().split()
        table = pd.DataFrame(
            {
                "fname": [f"{s}.jpg" for s in stems],
                "class": [canonical(s.split("/")[0]) for s in stems],
            }
        )
        if not set(table["class"]) <= set(self.classes):
            raise ValueError(f"{source}.txt names a class missing from classes.txt")
        return source, self.root / "images", table

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch the ETHZ tarball and move its ``food-101/`` contents into ``root``.

        ``meta/`` moves last, so ``meta/classes.txt`` marks a complete download.
        """
        root = Path(root).expanduser().resolve()
        if (root / "meta/classes.txt").exists():
            return
        archive = fetch_archive("https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", root, client)
        install_archive(archive, root, "food-101", "meta")
