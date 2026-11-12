"""UEC-Food256 loader with a seeded split and Hugging Face download."""

import random
from collections import Counter
from functools import cached_property
from pathlib import Path

import httpx
import pandas as pd

from classification.datasets.base import FoodImageDataset, canonical
from classification.datasets.util import fetch_archive, install_archive

SEED = 0
VALIDATION_RATIO = 0.2


class UECFood256ImageDataset(FoodImageDataset):
    """UEC-Food256 with a seeded per-class train/validation split; no official split exists.

    Each ``<class id>/`` directory under ``root`` holds that class's ``.jpg`` files.
    Files are sorted, shuffled with a per-class seed, and the first ``VALIDATION_RATIO``
    (rounded) are held out. Images listed under several class directories show several
    foods and are dropped from both splits. Class IDs sharing a canonical name
    (``40`` and ``67`` are both ``omelet``) map to one label.
    """

    name = "uecfood256"

    def _read_classes(self) -> list[str]:
        table = pd.read_csv(
            self.root / "category.txt",
            sep="\t",
            dtype={"id": "int64", "name": str},
            keep_default_na=False,
        )
        if table.empty or table["id"].duplicated().any():
            raise ValueError("category.txt must list unique class IDs")
        self.class_map = dict(zip(table["id"], table["name"].map(canonical), strict=True))
        return list(dict.fromkeys(self.class_map.values()))

    @cached_property
    def _partition(self) -> dict[str, pd.DataFrame]:
        listed = {
            i: sorted(p.name for p in (self.root / str(i)).glob("*.jpg")) for i in self.class_map
        }
        counts = Counter(name for names in listed.values() for name in names)
        rows: dict[str, list[tuple[str, str]]] = {"train": [], "validation": []}
        for class_id, names in listed.items():
            names = [n for n in names if counts[n] == 1]
            random.Random(f"{SEED}:{class_id}").shuffle(names)
            held_out = round(len(names) * VALIDATION_RATIO)
            label = self.class_map[class_id]
            rows["validation"] += [(f"{class_id}/{n}", label) for n in names[:held_out]]
            rows["train"] += [(f"{class_id}/{n}", label) for n in names[held_out:]]
        return {s: pd.DataFrame(r, columns=["fname", "class"]) for s, r in rows.items()}

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        return split, self.root, self._partition[split]

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch ``dataset256.zip`` from Hugging Face and move its ``UECFOOD256/`` contents into ``root``.

        ``category.txt`` moves last and marks a complete download.
        """
        root = Path(root).expanduser().resolve()
        if (root / "category.txt").exists():
            return
        url = "https://huggingface.co/datasets/kng-research/uecfood256/resolve/main/dataset256.zip"
        install_archive(fetch_archive(url, root, client), root, "UECFOOD256", "category.txt")
