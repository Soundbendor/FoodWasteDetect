"""CPU checks for dataset archive download and extraction with a mock HTTP transport."""

import io
import tarfile
import zipfile
from unittest.mock import patch

import httpx
import pytest
import yaml

from classification.cli import main
from classification.datasets.image_dataset import (
    FoodImageDataset,
    FoodX251ImageDataset,
    _extract,
    _fetch_archive,
)

FOODX251_FILES = {
    "annot": {"class_list.txt": "0 apple_pie\n1 pear\n", "train_info.csv": "a.png,0\nb.png,1\n", "val_info.csv": "c.png,1\n"},
    "train": {"train_set/a.png": "x", "train_set/b.png": "x"},
    "val": {"val_set/c.png": "x"},
}


def _tar(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _client(archives: dict[str, bytes], requests: list[str]) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        body = archives.get(request.url.path.lstrip("/"))
        return httpx.Response(200, content=body) if body is not None else httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_foodx251_download_loads_and_is_idempotent(tmp_path):
    requests: list[str] = []
    client = _client({f"{part}.tar": _tar(files) for part, files in FOODX251_FILES.items()}, requests)
    FoodX251ImageDataset.download(tmp_path, client=client)
    assert requests == [f"https://food-x.s3.amazonaws.com/{p}.tar" for p in ("train", "val", "annot")]
    assert not list((tmp_path / ".downloads").iterdir())

    dataset = FoodX251ImageDataset(tmp_path)
    assert dataset.get_class_list() == ["Apple pie", "Pear"]
    assert dataset.train_set()["source_id"] == ["foodx251:train:a.png", "foodx251:train:b.png"]
    assert dataset.validation_set()["class"] == ["Pear"]

    FoodX251ImageDataset.download(tmp_path, client=client)
    assert len(requests) == 3


def test_fetch_failure_leaves_no_archive(tmp_path):
    with pytest.raises(httpx.HTTPStatusError):
        _fetch_archive("https://host/missing.tar", tmp_path, _client({}, []))
    assert not (tmp_path / ".downloads/missing.tar").exists()


def test_completed_archive_is_reused(tmp_path):
    requests: list[str] = []
    client = _client({"a.tar": b"data"}, requests)
    first = _fetch_archive("https://host/a.tar", tmp_path, client)
    assert _fetch_archive("https://host/a.tar", tmp_path, client) == first
    assert first.read_bytes() == b"data" and len(requests) == 1


def test_extract_zip_and_reject_tar_escape(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dir/x.txt", "x")
    _extract(archive, tmp_path / "out")
    assert (tmp_path / "out/dir/x.txt").read_text() == "x"

    escape = tmp_path / "bad.tar"
    escape.write_bytes(_tar({"../escaped.txt": "x"}))
    with pytest.raises(tarfile.FilterError):
        _extract(escape, tmp_path / "bad")
    assert not (tmp_path / "escaped.txt").exists()


def test_cli_downloads_configured_dataset(tmp_path):
    class Fake(FoodImageDataset):
        name = "foodx251"
        roots: list[str] = []

        @classmethod
        def download(cls, root, client=None):
            cls.roots.append(root)

    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"dataset": {"name": "foodx251", "root": "data/x"}, "unrelated": None}))
    with (
        patch.dict("classification.datasets.image_dataset.DATASETS", {"foodx251": Fake}),
        patch("classification.tracking.wandb.init") as initialize,
    ):
        main(["download_dataset", "--config_file", str(config)])
    assert Fake.roots == ["data/x"]
    initialize.assert_not_called()
