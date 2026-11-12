"""Local food image datasets as typed Hugging Face image datasets."""

import contextlib
import random
import shutil
import subprocess
import tarfile
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlparse

import httpx
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


def _canonical(name: str) -> str:
    """Capitalize a taxonomy name and replace underscores with spaces."""
    return name.capitalize().replace("_", " ")


def _spaced(name: str) -> str:
    """Replace underscores with spaces, keeping case, parentheses, and non-ASCII text."""
    return name.replace("_", " ")


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
    labels = table["name"].map(_canonical)
    if labels.empty or labels.duplicated().any():
        raise ValueError("Taxonomy must contain unique canonical labels")
    return labels.to_dict()


def _fetch_archive(
    url: str, root: Path, client: httpx.Client | None = None, *, require_length: bool = False
) -> Path:
    """Stream ``url`` to ``root/.downloads/<name>`` through a ``.part`` file.

    An archive completed by an earlier call is returned without a request. An unencoded
    body whose size differs from ``Content-Length`` is discarded with ``ValueError``;
    ``require_length`` also rejects responses without that header.
    """
    target = root / ".downloads" / Path(urlparse(url).path).name
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with contextlib.ExitStack() as stack:
        if client is None:
            client = stack.enter_context(httpx.Client(follow_redirects=True, timeout=60))
        response = stack.enter_context(client.stream("GET", url))
        response.raise_for_status()
        expected = response.headers.get("Content-Length")
        if expected is None and require_length:
            raise ValueError(f"{url} sent no Content-Length")
        received = 0
        with partial.open("wb") as stream:
            for chunk in response.iter_bytes(1 << 20):
                received += stream.write(chunk)
        # Content-Length counts encoded bytes, so only identity bodies are comparable.
        encoded = response.headers.get("Content-Encoding", "identity") != "identity"
        if expected is not None and not encoded and received != int(expected):
            partial.unlink()
            raise ValueError(f"{url}: received {received} bytes, Content-Length {expected}")
    partial.replace(target)
    return target


def _extract(archive: Path, destination: Path) -> None:
    """Unpack a tar (``data`` filter) or zip archive into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(destination, filter="data")


def _install_archive(
    archive: Path, root: Path, inner: str, last: str, nested: str | None = None
) -> None:
    """Extract ``archive`` to a staging directory and move its ``inner`` entries into ``root``.

    A ``nested`` archive inside ``inner`` is unpacked in place and deleted before moving.
    The ``last`` entry moves after all others, so its presence marks a complete install.
    Existing entries of the same name are replaced; the staging directory and archive
    are deleted afterwards.
    """
    staging = archive.with_name(archive.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    _extract(archive, staging)
    if nested is not None:
        _extract(staging / inner / nested, staging / inner)
        (staging / inner / nested).unlink()
    for entry in sorted((staging / inner).iterdir(), key=lambda p: p.name == last):
        target = root / entry.name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        entry.replace(target)
    shutil.rmtree(staging)
    archive.unlink()


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
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch and unpack the dataset under ``root``; ``client`` overrides HTTP transport."""
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

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch the public S3 tars; ``annot`` unpacks last and marks completion."""
        root = Path(root).expanduser().resolve()
        if (root / "annot/class_list.txt").exists():
            return
        for part in ("train", "val", "annot"):
            archive = _fetch_archive(f"https://food-x.s3.amazonaws.com/{part}.tar", root, client)
            _extract(archive, root / part)
            archive.unlink()


class Food101ImageDataset(FoodImageDataset):
    """Food-101 official train references and labeled test split."""

    name = "food101"

    def _read_classes(self) -> list[str]:
        labels = [_canonical(n) for n in (self.root / "meta/classes.txt").read_text().split()]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Taxonomy must contain unique canonical labels")
        return labels

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        stems = (self.root / f"meta/{source}.txt").read_text().split()
        table = pd.DataFrame(
            {
                "fname": [f"{s}.jpg" for s in stems],
                "class": [_canonical(s.split("/")[0]) for s in stems],
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
        archive = _fetch_archive("https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", root, client)
        _install_archive(archive, root, "food-101", "meta")


class UECFood256ImageDataset(FoodImageDataset):
    """UEC-Food256 with a seeded per-class train/validation split; no official split exists.

    Each ``<class id>/`` directory under ``root`` holds that class's ``.jpg`` files.
    Files are sorted, shuffled with a per-class seed, and the first ``VALIDATION_RATIO``
    (rounded) are held out. Images listed under several class directories show several
    foods and are dropped from both splits. Class IDs sharing a canonical name
    (``40`` and ``67`` are both ``omelet``) map to one label.
    """

    name = "uecfood256"
    SEED: ClassVar[int] = 0
    VALIDATION_RATIO: ClassVar[float] = 0.2

    def _read_classes(self) -> list[str]:
        table = pd.read_csv(
            self.root / "category.txt",
            sep="\t",
            dtype={"id": "int64", "name": str},
            keep_default_na=False,
        )
        if table.empty or table["id"].duplicated().any():
            raise ValueError("category.txt must list unique class IDs")
        self.class_map = dict(zip(table["id"], table["name"].map(_canonical), strict=True))
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
            random.Random(f"{self.SEED}:{class_id}").shuffle(names)
            held_out = round(len(names) * self.VALIDATION_RATIO)
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
        _install_archive(_fetch_archive(url, root, client), root, "UECFOOD256", "category.txt")


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
        labels = table["class_name"].map(_canonical)
        if table.empty or table["class_id"].duplicated().any() or labels.duplicated().any():
            raise ValueError("classid_classname.csv must list unique IDs and canonical labels")
        return list(labels)

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        lines = (self.root / self.LIST_DIR / f"vegfru_{source}.txt").read_text().splitlines()
        paths = [line.rsplit(maxsplit=1)[0] for line in lines if line.strip()]
        table = pd.DataFrame(
            {"fname": paths, "class": [_canonical(Path(p).parent.name) for p in paths]}
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
        _install_archive(
            _fetch_archive(url, root, client),
            root,
            "vegfru",
            "classid_classname.csv",
            nested="archive.zip",
        )


class ISIAFood500ImageDataset(FoodImageDataset):
    """ISIA Food-500: official ``train_finetune`` references, ``test_finetune`` evaluation.

    ``META_DIR/ISIAFood500_classLabel.txt`` holds ``name<TAB>id`` lines. The split lists
    hold ``<class>/<file> <id>`` lines relative to ``IMAGE_DIR``; labels come from the ID.
    Names only have underscores replaced, keeping case, parentheses, and non-ASCII text.
    """

    name = "isia500"
    BASE_URL: ClassVar[str] = "http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/"
    # Location assumed next to the image parts; unverified.
    METADATA_URL: ClassVar[str] = BASE_URL + "metadata_ISIAFood_500.zip"
    PARTS: ClassVar[tuple[str, ...]] = (
        *(f"ISIA_Food500.z{i:02d}" for i in range(1, 11)),
        "ISIA_Food500.zip",
    )
    META_DIR: ClassVar[str] = "metadata_ISIAFood_500"
    IMAGE_DIR: ClassVar[str] = "images"
    CLASS_FILE: ClassVar[str] = "ISIAFood500_classLabel.txt"

    def _read_classes(self) -> list[str]:
        table = pd.read_csv(
            self.root / self.META_DIR / self.CLASS_FILE,
            sep="\t",
            names=["name", "id"],
            dtype={"name": str, "id": "int64"},
            keep_default_na=False,
        )
        labels = table["name"].map(_spaced)
        if table.empty or table["id"].duplicated().any() or labels.duplicated().any():
            raise ValueError(f"{self.CLASS_FILE} must list unique IDs and canonical labels")
        self.class_map = dict(zip(table["id"], labels, strict=True))
        return list(labels)

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        lines = (self.root / self.META_DIR / f"{source}_finetune.txt").read_text().splitlines()
        rows = [line.rsplit(maxsplit=1) for line in lines if line.strip()]
        ids = [int(class_id) for _, class_id in rows]
        if not set(ids) <= self.class_map.keys():
            raise ValueError(
                f"{source}_finetune.txt names a class ID missing from {self.CLASS_FILE}"
            )
        table = pd.DataFrame(
            {"fname": [path for path, _ in rows], "class": [self.class_map[i] for i in ids]}
        )
        return source, self.root / self.IMAGE_DIR, table

    @classmethod
    def download(cls, root: str | Path, client: httpx.Client | None = None) -> None:
        """Fetch the metadata zip and eleven split-zip parts, then unpack them under ``root``.

        Each part must match its ``Content-Length``. ``7z x`` unpacks the images into
        ``root``; the metadata moves into ``META_DIR`` last, so the class-label file marks
        a complete download. Archives are deleted afterwards.
        """
        root = Path(root).expanduser().resolve()
        if (root / cls.META_DIR / cls.CLASS_FILE).exists():
            return
        sevenzip = shutil.which("7z")
        if sevenzip is None:
            raise RuntimeError(
                "7z is not on PATH; run within the Pixi environment, which provides p7zip"
            )
        archives = [
            _fetch_archive(url, root, client, require_length=True)
            for url in (cls.METADATA_URL, *(cls.BASE_URL + part for part in cls.PARTS))
        ]
        metadata, *parts = archives
        subprocess.run([sevenzip, "x", "-y", f"-o{root}", str(parts[-1])], check=True)
        staging = metadata.with_name(metadata.name + ".staging")
        shutil.rmtree(staging, ignore_errors=True)
        _extract(metadata, staging)
        (label_file,) = staging.rglob(cls.CLASS_FILE)
        destination = root / cls.META_DIR
        destination.mkdir(exist_ok=True)
        for entry in sorted(label_file.parent.iterdir(), key=lambda p: p.name == cls.CLASS_FILE):
            entry.replace(destination / entry.name)
        shutil.rmtree(staging)
        for archive in archives:
            archive.unlink()


DATASETS: dict[str, type[FoodImageDataset]] = {
    c.name: c
    for c in (
        FoodX251ImageDataset,
        Food101ImageDataset,
        UECFood256ImageDataset,
        VegFruImageDataset,
        ISIAFood500ImageDataset,
    )
}


def load_dataset(config: DatasetConfig) -> FoodImageDataset:
    """Load metadata for the configured dataset from ``config.root``."""
    return DATASETS[config.name](config.root)
