"""Local food classification metadata and shared Hugging Face image batches."""

import json
import math
import random
import unicodedata
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, NamedSplit, Value

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


class FoodImageDataset(ABC):
    """Local metadata only at construction; decoding happens when batches are read."""

    namespace: str
    train_path: Path
    val_path: Path
    test_path: Path
    class_map: dict[int, str]
    splits: DatasetDict

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._load_metadata()

    @abstractmethod
    def _load_metadata(self) -> None:
        """Populate paths, taxonomy, and logical train/validation slots."""

    def _build_split(
        self,
        rows: Iterable[tuple[str, int]],
        split: str,
        directory: Path,
    ) -> Dataset:
        features = Features(
            {
                "fname": Value("string"),
                "class": Value("string"),
                "source_id": Value("string"),
                "source_split": Value("string"),
                "image": Image(decode=False),
            }
        )
        table = pd.DataFrame(rows, columns=["fname", "class_id"])
        # convert numerical class IDs to label names.
        table["class"] = table.pop("class_id").map(self.class_map.__getitem__)
        base = directory.resolve()
        paths = [(base / filename).resolve() for filename in table["fname"]]
        table["source_id"] = [
            f"{self.namespace}:{split}:{path.relative_to(base)}" for path in paths
        ]
        if table["source_id"].duplicated().any():
            raise ValueError("Duplicate source identities")
        table["fname"] = pd.Series([str(path) for path in paths], dtype="string")
        table["source_split"] = split
        table["image"] = table["fname"]
        return Dataset.from_pandas(
            table,
            features=features,
            preserve_index=False,
            split=NamedSplit("train" if split == "train" else "validation"),
        )

    def _check_overlap(self) -> None:
        if set(self.train_set()["fname"]) & set(self.test_set()["fname"]):
            raise ValueError("Reference and evaluation metadata overlap")

    def get_class_list(self) -> list[str]:
        return list(self.class_map.values())

    def train_set(self) -> Dataset:
        return self.splits["train"]

    def test_set(self) -> Dataset:
        """Return evaluation images; source_split records their native partition."""
        return self.splits["validation"]

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        n_imgs_per_class: int | None = None,
        excluded_source_ids: set[str] | None = None,
        decode: bool = True,
    ) -> Iterator[dict[str, list]]:
        """Yield annotation-order batches, optionally metadata only without image reads.

        Apply the per-class cap before identity exclusion and decoding. Decoded
        images belong to the caller and are released through garbage collection.
        """
        dataset = self.splits[SPLIT_NAMES.get(split, split)]
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
        if not decode:
            dataset = dataset.remove_columns("image")
        else:
            dataset = dataset.cast_column("image", Image())
        return dataset.iter(batch_size=batch_size, drop_last_batch=False)


class FoodX251ImageDataset(FoodImageDataset):
    """Official training references and labeled validation images."""

    namespace = "foodx251"

    def _load_metadata(self) -> None:
        self.train_path = self.root / "train/train_set"
        self.val_path = self.root / "val/val_set"
        self.test_path = self.val_path
        self.class_map = _read_class_map(self.root / "annot/class_list.txt")
        self.splits = DatasetDict(
            {
                "train": self._load_split("train", self.train_path),
                "validation": self._load_split("val", self.val_path),
            }
        )

    def _load_split(self, split: str, directory: Path) -> Dataset:
        table = pd.read_csv(
            self.root / f"annot/{split}_info.csv",
            names=["fname", "class_id"],
            dtype={"fname": str, "class_id": "int64"},
            keep_default_na=False,
        )
        return self._build_split(table.itertuples(index=False, name=None), split, directory)


class Food2KImageDataset(FoodX251ImageDataset):
    """User-provided FoodX-style annotations, not an assumed publisher layout."""

    namespace = "food2k"

    def _load_metadata(self) -> None:
        super()._load_metadata()
        if any(not label.strip() for label in self.class_map.values()):
            raise ValueError("Empty class label")


def _canonical_map(pairs: Iterable[tuple[int, str]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for identifier, name in pairs:
        if not name.strip():
            raise ValueError("Empty class label")
        if identifier in result:
            raise ValueError("Duplicate class ID")
        result[identifier] = name.capitalize().replace("_", " ")
    if not result or len(set(result.values())) != len(result):
        raise ValueError("Taxonomy must contain unique canonical labels")
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON class key")
    return result


class Food101ImageDataset(FoodImageDataset):
    """Official Food-101 training and test metadata, in annotation order."""

    namespace = "food101"

    def _load_metadata(self) -> None:
        self.train_path = self.val_path = self.test_path = self.root / "images"
        metadata = {}
        for split in ("train", "test"):
            with (self.root / f"meta/{split}.json").open(encoding="utf-8") as stream:
                data = json.load(stream, object_pairs_hook=_unique_json_object)
            if not isinstance(data, dict) or not all(
                isinstance(paths, list) and all(isinstance(path, str) for path in paths)
                for paths in data.values()
            ):
                raise ValueError("Food-101 metadata must map classes to lists of paths")
            metadata[split] = data
        raw_ids = {name: i for i, name in enumerate(sorted(metadata["train"]))}
        self.class_map = _canonical_map((i, name) for name, i in raw_ids.items())
        self.splits = DatasetDict(
            {
                slot: self._build_split(
                    (
                        (f"{path}.jpg", raw_ids[name])
                        for name, paths in metadata[split].items()
                        for path in paths
                    ),
                    split,
                    self.train_path,
                )
                for slot, split in (("train", "train"), ("validation", "test"))
            }
        )
        self._check_overlap()


class ISIAFood500ImageDataset(FoodImageDataset):
    """Official finetune training/test lists; official validation is unused."""

    namespace = "isia_food500"

    def _load_metadata(self) -> None:
        self.train_path = self.val_path = self.test_path = self.root / "ISIA_Food500/images"
        metadata = self.root / "metadata_ISIAFood_500"
        pairs = []
        for line in (
            (metadata / "ISIAFood500_classLabel.txt").read_text(encoding="utf-8").splitlines()
        ):
            name, identifier = line.rsplit("\t", 1)
            pairs.append((int(identifier), name))
        self.class_map = _canonical_map(pairs)
        self.splits = DatasetDict(
            {
                slot: self._build_split(
                    self._read_rows(metadata / f"{split}_finetune.txt"),
                    split,
                    self.train_path,
                )
                for slot, split in (("train", "train"), ("validation", "test"))
            }
        )
        self._check_overlap()

    @staticmethod
    def _read_rows(path: Path) -> Iterator[tuple[str, int]]:
        for line in path.read_text(encoding="utf-8").splitlines():
            filename, identifier = line.rsplit(maxsplit=1)
            yield filename, int(identifier)


def read_mm_rows(path: Path) -> list[tuple[str, str, str]]:
    """Read URL, safe basename, normalized label without fetching images."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    for url, label in zip(table["image_url"], table["dish_name"], strict=True):
        label = " ".join(unicodedata.normalize("NFKC", label).split()).casefold()
        if not label:
            raise ValueError("Empty class label")
        parsed = urlsplit(url)
        name = unquote(parsed.path.rsplit("/", 1)[-1])
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or name in {"", ".", ".."}
            or any(c in name for c in "/\\\x00")
        ):
            raise ValueError(f"Invalid image URL or basename: {url!r}")
        if url in seen_urls:
            raise ValueError("Duplicate source identities")
        if name in seen_names:
            raise ValueError(f"Ambiguous URL-basename collision: {name}")
        seen_urls.add(url)
        seen_names.add(name)
        rows.append((url, name, label))
    if not rows:
        raise ValueError("MM-Food metadata is empty")
    return rows


def _validate_mm_options(holdout_fraction: object, split_seed: object) -> None:
    if (
        isinstance(holdout_fraction, bool)
        or not isinstance(holdout_fraction, (int, float))
        or not math.isfinite(holdout_fraction)
        or not 0 < holdout_fraction < 1
    ):
        raise ValueError("holdout_fraction must be finite and between 0 and 1")
    if type(split_seed) is not int:
        raise ValueError("split_seed must be an integer")


class MMFood100KImageDataset(FoodImageDataset):
    """Dish-name stratified holdout; singleton classes remain training-only."""

    namespace = "mm_food100k"

    def __init__(
        self,
        root: str | Path,
        *,
        holdout_fraction: float = 0.2,
        split_seed: int = 42,
    ) -> None:
        _validate_mm_options(holdout_fraction, split_seed)
        self.holdout_fraction = holdout_fraction
        self.split_seed = split_seed
        super().__init__(root)

    def _load_metadata(self) -> None:
        self.train_path = self.val_path = self.test_path = self.root / "images"
        rows = read_mm_rows(self.root / "MM-Food-100K.csv")
        groups: dict[str, list[str]] = defaultdict(list)
        for url, _, label in rows:
            groups[label].append(url)
        self.class_map = dict(enumerate(sorted(groups)))
        identifiers = {label: i for i, label in self.class_map.items()}
        holdout = set()
        for label, urls in groups.items():
            if len(urls) < 2:
                continue
            ordered = sorted(urls)
            random.Random(f"{self.split_seed}:{label}").shuffle(ordered)
            count = min(len(urls) - 1, max(1, math.floor(len(urls) * self.holdout_fraction)))
            holdout.update(ordered[:count])
        self.splits = DatasetDict(
            {
                slot: self._build_split(
                    (
                        (name, identifiers[label])
                        for url, name, label in rows
                        if (url in holdout) == is_holdout
                    ),
                    split,
                    self.train_path,
                )
                for slot, split, is_holdout in (
                    ("train", "train", False),
                    ("validation", "holdout", True),
                )
            }
        )
        self._check_overlap()


DATASETS: dict[str, type[FoodImageDataset]] = {
    "foodx251": FoodX251ImageDataset,
    "food101": Food101ImageDataset,
    "mm_food100k": MMFood100KImageDataset,
    "isia_food500": ISIAFood500ImageDataset,
    "food2k": Food2KImageDataset,
}


def validate_dataset_config(config: Mapping) -> None:
    """Validate options without filesystem access, model loading, or network calls."""
    if not isinstance(config, Mapping):
        raise ValueError("dataset must be a mapping")
    name = config.get("name")
    if not isinstance(name, str) or name.lower() not in DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}; valid choices: {', '.join(DATASETS)}")
    if not isinstance(config.get("root"), (str, Path)) or not str(config["root"]).strip():
        raise ValueError("dataset.root must be a nonempty path")
    options = {"holdout_fraction", "split_seed"} if name.lower() == "mm_food100k" else set()
    unknown = set(config) - {"name", "root"} - options
    if unknown:
        raise ValueError(f"Unsupported dataset options: {sorted(unknown)}")
    if options:
        _validate_mm_options(config.get("holdout_fraction", 0.2), config.get("split_seed", 42))


def load_dataset(config: dict) -> FoodImageDataset:
    validate_dataset_config(config)
    options = {key: value for key, value in config.items() if key not in {"name", "root"}}
    return DATASETS[config["name"].lower()](config["root"], **options)
