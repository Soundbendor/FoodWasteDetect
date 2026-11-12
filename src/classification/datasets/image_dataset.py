"""Local food image datasets as typed Hugging Face image datasets."""

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, NamedSplit, Value

from classification.experiments.config import DatasetConfig

FEATURES = Features(
    {
        "fname": Value("string"),
        "class": Value("string"),
        "source_id": Value("string"),
        "source_split": Value("string"),
        "image": Image(decode=False),
    }
)


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
    labels = table["name"].str.capitalize().str.replace("_", " ", regex=False)
    if labels.empty or labels.duplicated().any():
        raise ValueError("Taxonomy must contain unique canonical labels")
    return labels.to_dict()


class FoodImageDataset(ABC):
    """Load local metadata for a labeled image dataset, deferring image decoding.

    Subclasses supply the class list and, per split, the source split name, image
    directory, and a ``fname``/``class`` table of relative paths and labels.
    """

    name: ClassVar[str]

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.classes = self._read_classes()
        self.splits = DatasetDict(
            {s: self._build_split(s, *self._read_split(s)) for s in ("train", "validation")}
        )

    @abstractmethod
    def _read_classes(self) -> list[str]:
        """Return unique canonical labels in taxonomy order."""

    @abstractmethod
    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        """Return (source_split, image_dir, table[fname, class]) for train/validation."""

    @classmethod
    def download(cls, root: str | Path) -> None:
        """Fetch and unpack the dataset under ``root``."""
        raise NotImplementedError(f"{cls.name} has no public download host")

    def _build_split(
        self, split: str, source_split: str, image_dir: Path, table: pd.DataFrame
    ) -> Dataset:
        table = table[["fname", "class"]].copy()
        base = image_dir.resolve()
        paths = [(base / filename).resolve() for filename in table["fname"]]
        table["source_id"] = [
            f"{self.name}:{source_split}:{path.relative_to(base)}" for path in paths
        ]
        if table["source_id"].duplicated().any():
            raise ValueError("Duplicate source identities")
        table["fname"] = pd.Series([str(path) for path in paths], dtype="string")
        table["source_split"] = source_split
        table["image"] = table["fname"]
        return Dataset.from_pandas(
            table,
            features=FEATURES,
            preserve_index=False,
            split=NamedSplit(split),
        )

    def get_class_list(self) -> list[str]:
        """Return canonical labels in taxonomy-file order."""
        return list(self.classes)

    def train_set(self) -> Dataset:
        """Return training references with undecoded image paths."""
        return self.splits["train"]

    def validation_set(self) -> Dataset:
        """Return the labeled evaluation split; ``source_split`` keeps its own name."""
        return self.splits["validation"]

    @staticmethod
    def select_images(
        dataset: Dataset,
        *,
        n_imgs_per_class: int | None = None,
        excluded_source_ids: set[str] | None = None,
    ) -> Dataset:
        """Select in annotation order without decoding or changing the input dataset.

        Apply the class cap before exclusions so resumption preserves the selection.
        """
        if n_imgs_per_class is not None:
            counts: Counter[str] = Counter()
            indices: list[int] = []
            for index, label in enumerate(dataset["class"]):
                if counts[label] < n_imgs_per_class:
                    indices.append(index)
                    counts[label] += 1
            dataset = dataset.select(indices)
        if excluded_source_ids:
            dataset = dataset.select(
                [
                    i
                    for i, source_id in enumerate(dataset["source_id"])
                    if source_id not in excluded_source_ids
                ]
            )
        return dataset

    @staticmethod
    def iter_dataset_batches(
        dataset: Dataset, batch_size: int, *, decode: bool = True
    ) -> Iterator[dict[str, list]]:
        """Yield ordered column batches, including the final partial batch.

        With ``decode=False``, omit the image column and leave image files unread.
        Otherwise, decode images during iteration and propagate loading errors.
        """
        if not decode:
            dataset = dataset.remove_columns("image")
        else:
            dataset = dataset.cast_column("image", Image())
        return dataset.iter(batch_size=batch_size, drop_last_batch=False)


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


DATASETS: dict[str, type[FoodImageDataset]] = {c.name: c for c in (FoodX251ImageDataset,)}


def load_dataset(config: DatasetConfig) -> FoodImageDataset:
    """Load metadata for the configured dataset from ``config.root``."""
    return DATASETS[config.name](config.root)
