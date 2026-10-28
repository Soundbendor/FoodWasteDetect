"""Explicit dataset preparation. Parsing and experiment execution never download."""

import shutil
import stat
import subprocess
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from .image_dataset import FoodImageDataset, load_dataset, read_mm_rows, validate_dataset_config

FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
MM_CSV_URL = (
    "https://huggingface.co/datasets/Humanbased-AI/MM-Food-100K/resolve/main/MM-Food-100K.csv"
)
ISIA_URL = "http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/"
ISIA_VOLUMES = [f"ISIA_Food500.z{i:02}" for i in range(1, 11)] + ["ISIA_Food500.zip"]


def _contained(directory: Path, name: str) -> Path:
    """Reject archive paths and existing symlinks that could leave the destination."""
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe archive/destination path: {name!r}")
    target = directory / path
    if not target.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Path escapes destination: {name!r}")
    return target


def _download(client: httpx.Client, url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = _contained(destination.parent, destination.name + ".part")
    # One writer per root. Incomplete transfers restart; completed files are retained.
    with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        count = 0
        with partial.open("wb") as stream:
            for chunk in response.iter_bytes():
                stream.write(chunk)
                count += len(chunk)
        length = response.headers.get("content-length")
        if (
            length is not None
            and not response.headers.get("content-encoding")
            and count != int(length)
        ):
            raise OSError(f"Incomplete download: {url}")
    partial.replace(destination)


def _extract_tar(archive: Path, staging: Path) -> None:
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for member in members:
            _contained(staging, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        source.extractall(staging, members=members, filter="data")


def _extract_zip(archive: Path, staging: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            _contained(staging, member.filename)
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported archive entry: {member.filename}")
        source.extractall(staging)


def _extract_multipart(archive: Path, volumes: list[Path], staging: Path) -> None:
    for volume in volumes:
        if not volume.is_file():
            raise FileNotFoundError(f"Missing archive volume: {volume}")
    executable = shutil.which("7zz")
    if executable is None:
        raise RuntimeError("ISIA preparation requires 7zz; install the Pixi environment")
    listing = subprocess.run(
        [executable, "l", "-slt", "-ba", "-sccUTF-8", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=3600,
    ).stdout
    paths = []
    for line in listing.splitlines():
        key, separator, value = line.partition(" = ")
        if key == "Path" and separator:
            paths.append(_contained(staging, value))
        if key in {"Symbolic Link", "Hard Link"} or (
            key in {"Attributes", "Mode"} and any(part.startswith("l") for part in value.split())
        ):
            raise ValueError("Archive links are not supported")
    if not paths:
        raise ValueError("Archive has no listed paths")
    subprocess.run(
        [executable, "x", "-y", "-bd", f"-o{staging}", str(archive)],
        check=True,
        capture_output=True,
        timeout=3600,
    )
    _check_tree(staging)


def _check_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"Unsupported extracted entry: {path}")
        _contained(directory, str(path.relative_to(directory)))


def _publish(staging: Path, root: Path) -> None:
    _check_tree(staging)
    files = [path for path in staging.rglob("*") if path.is_file()]
    # Validate every destination before publishing any content.
    destinations = [_contained(root, str(path.relative_to(staging))) for path in files]
    for source, destination in zip(files, destinations, strict=True):
        if not destination.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)


def _require_images(dataset: FoodImageDataset) -> None:
    for split in dataset.splits.values():
        for name in split["fname"]:
            if not Path(name).is_file():
                raise FileNotFoundError(f"Missing annotated image: {name}")


def _complete(config: dict) -> bool:
    try:
        _require_images(load_dataset(config))
    except FileNotFoundError:
        return False
    return True


def prepare_dataset(config: dict) -> None:
    """Provision a configured dataset, without models, tracking, or service startup.

    Supports one preparation writer per root. No hashes or image decoding are used.
    Existing completed files are retained; failures propagate with downloads reusable.
    """
    validate_dataset_config(config)
    name = config["name"].lower()
    if name in {"foodx251", "food2k"}:
        raise ValueError(
            f"{name} requires local provisioning: annot/class_list.txt, annot/train_info.csv, "
            "annot/val_info.csv, train/train_set/, and val/val_set/"
        )
    if _complete(config):
        return
    root = Path(config["root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    downloads = _contained(root, ".downloads")
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=15)) as client:
        if name == "mm_food100k":
            csv_path = _contained(root, "MM-Food-100K.csv")
            _download(client, MM_CSV_URL, csv_path)
            rows = read_mm_rows(csv_path)
            targets = [_contained(root, f"images/{filename}") for _, filename, _ in rows]
            with ThreadPoolExecutor(max_workers=4) as pool:
                # map yields in submission order and propagates errors; no record is dropped.
                try:
                    for _ in pool.map(
                        lambda pair: _download(client, pair[0], pair[1]),
                        zip((url for url, _, _ in rows), targets, strict=True),
                    ):
                        pass
                except BaseException:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
        else:
            if name == "food101":
                archive = _contained(downloads, "food-101.tar.gz")
                _download(client, FOOD101_URL, archive)
            else:
                for filename in [*ISIA_VOLUMES, "metadata_ISIAFood_500.zip"]:
                    _download(client, ISIA_URL + filename, _contained(downloads, filename))
            with TemporaryDirectory(prefix=".extract-", dir=root) as temporary:
                staging = Path(temporary)
                if name == "food101":
                    _extract_tar(downloads / "food-101.tar.gz", staging)
                    content = staging / "food-101"
                else:
                    _extract_zip(downloads / "metadata_ISIAFood_500.zip", staging)
                    _extract_multipart(
                        downloads / "ISIA_Food500.zip",
                        [downloads / filename for filename in ISIA_VOLUMES],
                        staging,
                    )
                    content = staging
                _require_images(load_dataset({**config, "root": str(content)}))
                _publish(content, root)
    _require_images(load_dataset(config))
