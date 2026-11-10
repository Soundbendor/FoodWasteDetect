"""Local FoodX251 annotations and Hugging Face image datasets."""

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, NamedSplit, Value

from classification.experiments.config import DatasetConfig

SPLIT_NAMES = {"train": "train", "val": "validation"}


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


class FoodX251ImageDataset:
    """Load local FoodX251 metadata, deferring image decoding to batch iteration."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.train_path = self.root / "train/train_set"
        self.val_path = self.root / "val/val_set"
        # The experiment evaluates validation data, not the unlabeled test split.
        self.test_path = self.val_path
        self.class_map = _read_class_map(self.root / "annot/class_list.txt")
        self.splits = DatasetDict(
            {
                "train": self._load_split("train", self.train_path),
                "validation": self._load_split("val", self.val_path),
            }
        )

    def _load_split(self, split: str, directory: Path) -> Dataset:
        features = Features(
            {
                "fname": Value("string"),
                "class": Value("string"),
                "source_id": Value("string"),
                "source_split": Value("string"),
                "image": Image(decode=False),
            }
        )
        table = pd.read_csv(
            self.root / f"annot/{split}_info.csv",
            names=["fname", "class_id"],
            dtype={"fname": str, "class_id": "int64"},
            keep_default_na=False,
        )
        # convert numerical class IDs to label names.
        table["class"] = table.pop("class_id").map(self.class_map.__getitem__)
        base = directory.resolve()
        paths = [(base / filename).resolve() for filename in table["fname"]]
        table["source_id"] = [f"foodx251:{split}:{path.relative_to(base)}" for path in paths]
        if table["source_id"].duplicated().any():
            raise ValueError("Duplicate source identities")
        table["fname"] = pd.Series([str(path) for path in paths], dtype="string")
        table["source_split"] = split
        table["image"] = table["fname"]
        return Dataset.from_pandas(
            table,
            features=features,
            preserve_index=False,
            split=NamedSplit(SPLIT_NAMES[split]),
        )

    def get_class_list(self) -> list[str]:
        """Return canonical labels in taxonomy-file order."""
        return list(self.class_map.values())

    def train_set(self) -> Dataset:
        """Return training references with undecoded image paths."""
        return self.splits["train"]

    def validation_set(self) -> Dataset:
        """Return the official labeled validation split used for evaluation."""
        return self.splits["validation"]

    def test_set(self) -> Dataset:
        """Compatibility alias for the labeled validation split."""
        return self.validation_set()

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

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        n_imgs_per_class: int | None = None,
        excluded_source_ids: set[str] | None = None,
        decode: bool = True,
    ) -> Iterator[dict[str, list]]:
        """Select and batch ``train``, ``val``, or ``validation`` in annotation order.

        Apply class caps before exclusions, then decode images only if requested.
        Metadata-only batches omit the image column; both modes retain partial batches.
        """
        dataset = self.select_images(
            self.splits[SPLIT_NAMES.get(split, split)],
            n_imgs_per_class=n_imgs_per_class,
            excluded_source_ids=excluded_source_ids,
        )
        return self.iter_dataset_batches(dataset, batch_size, decode=decode)


def load_dataset(config: DatasetConfig) -> FoodX251ImageDataset:
    """Load metadata from ``config.root``; the config admits only ``foodx251``."""
    return FoodX251ImageDataset(config.root)
