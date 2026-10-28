"""Preparation with mocked HTTP, staged archives, and a tiny real split ZIP."""

import io
import json
import random
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from internvl import dataset_download as download


def tar_bytes(files):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def zip_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return stream.getvalue()


@pytest.fixture
def http(monkeypatch):
    client_class = httpx.Client
    calls = []

    def install(handler):
        def handle(request):
            calls.append(str(request.url))
            return handler(request)

        monkeypatch.setattr(
            download.httpx,
            "Client",
            lambda **kwargs: client_class(
                **kwargs,
                transport=httpx.MockTransport(handle),
            ),
        )
        return calls

    return install


def food101_archive():
    return tar_bytes(
        {
            "food-101/meta/train.json": json.dumps({"apple": ["apple/a"]}).encode(),
            "food-101/meta/test.json": json.dumps({"apple": ["apple/b"]}).encode(),
            "food-101/images/apple/a.jpg": b"not decoded during preparation",
            "food-101/images/apple/b.jpg": b"not decoded during preparation either",
        }
    )


def test_food101_download_reuse_and_missing_image_repair(tmp_path, http):
    calls = http(lambda request: httpx.Response(200, content=food101_archive()))
    config = {"name": "food101", "root": str(tmp_path / "data")}
    root = Path(config["root"])
    download.prepare_dataset(config)
    assert calls == [download.FOOD101_URL]
    assert (
        root / "images/apple/a.jpg"
    ).read_bytes() == b"not decoded during preparation"
    assert not (root / "food-101").exists()
    download.prepare_dataset(config)
    assert len(calls) == 1
    (root / "images/apple/a.jpg").unlink()
    download.prepare_dataset(config)
    assert (root / "images/apple/a.jpg").is_file()
    assert len(calls) == 1
    assert not list(root.glob(".extract-*"))


def test_mm_preserves_formats_and_reuses_completed_files(tmp_path, http):
    csv = b"image_url,dish_name\nhttps://fixture.test/a.webp,Pear\nhttps://fixture.test/b.png,pear\n"
    content = {
        download.MM_CSV_URL: csv,
        "https://fixture.test/a.webp": b"webp",
        "https://fixture.test/b.png": b"png",
    }
    calls = http(lambda request: httpx.Response(200, content=content[str(request.url)]))
    config = {"name": "mm_food100k", "root": str(tmp_path)}
    (tmp_path / "images").mkdir()
    (tmp_path / "images/a.webp").write_bytes(b"local file")
    download.prepare_dataset(config)
    assert set(calls) == {download.MM_CSV_URL, "https://fixture.test/b.png"}
    assert (tmp_path / "images/a.webp").read_bytes() == b"local file"
    assert (tmp_path / "images/b.png").read_bytes() == b"png"
    download.prepare_dataset(config)
    assert len(calls) == 2


class InterruptedStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"partial"
        raise httpx.ReadError("interrupted transfer")


def test_interrupted_download_restarts_and_retains_completed(tmp_path):
    target = tmp_path / "image.jpg"
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    stream=InterruptedStream(),
                )
            )
        ) as client,
        pytest.raises(httpx.ReadError),
    ):
        download._download(client, "https://fixture.test/image.jpg", target)
    assert not target.exists()
    assert target.with_suffix(".jpg.part").read_bytes() == b"partial"
    handler = Mock(return_value=httpx.Response(200, content=b"complete"))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download._download(client, "https://fixture.test/image.jpg", target)
        download._download(client, "https://fixture.test/image.jpg", target)
    assert handler.call_count == 1
    assert target.read_bytes() == b"complete"
    assert not target.with_suffix(".jpg.part").exists()


@pytest.mark.parametrize("status", [404, 500])
def test_http_failures_propagate(tmp_path, status):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(status))
        ) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        download._download(client, "https://fixture.test/a", tmp_path / "a")
    assert not (tmp_path / "a").exists()


def test_short_download_is_not_published(tmp_path):
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    content=b"short",
                    headers={"Content-Length": "100"},
                )
            )
        ) as client,
        pytest.raises(OSError, match="Incomplete download"),
    ):
        download._download(client, "https://fixture.test/a", tmp_path / "a")
    assert not (tmp_path / "a").exists()


def test_partial_symlink_cannot_overwrite_another_directory(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    protected = tmp_path / "protected"
    protected.write_bytes(b"keep")
    (images / "a.part").symlink_to(protected)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client, pytest.raises(ValueError, match="escapes"):
        download._download(client, "https://fixture.test/a", images / "a")
    assert protected.read_bytes() == b"keep"


def test_mm_image_failure_keeps_completed_files_and_propagates(tmp_path, http):
    csv = b"image_url,dish_name\nhttps://fixture.test/a.png,Pear\nhttps://fixture.test/b.png,pear\n"
    (tmp_path / "MM-Food-100K.csv").write_bytes(csv)
    (tmp_path / "images").mkdir()
    (tmp_path / "images/a.png").write_bytes(b"completed")
    calls = http(lambda request: httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        download.prepare_dataset({"name": "mm_food100k", "root": str(tmp_path)})
    assert calls == ["https://fixture.test/b.png"]
    assert (tmp_path / "images/a.png").read_bytes() == b"completed"
    assert not (tmp_path / "images/b.png").exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "dir/../../outside", "dir\\outside"]
)
def test_archive_path_containment(tmp_path, kind, name):
    target = tmp_path / "archive"
    target.write_bytes(
        tar_bytes({name: b"x"}) if kind == "tar" else zip_bytes({name: b"x"})
    )
    with pytest.raises(ValueError, match="Unsafe"):
        (download._extract_tar if kind == "tar" else download._extract_zip)(
            target, tmp_path / "stage"
        )
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("kind", ["tar", "zip"])
def test_archive_links_rejected(tmp_path, kind):
    archive = tmp_path / "archive"
    if kind == "tar":
        with tarfile.open(archive, "w") as stream:
            member = tarfile.TarInfo("link")
            member.type, member.linkname = tarfile.SYMTYPE, "../outside"
            stream.addfile(member)
    else:
        with zipfile.ZipFile(archive, "w") as stream:
            member = zipfile.ZipInfo("link")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            stream.writestr(member, "../outside")
    with pytest.raises(ValueError, match="Unsupported"):
        (download._extract_tar if kind == "tar" else download._extract_zip)(
            archive, tmp_path / "stage"
        )


def test_failed_or_incomplete_extraction_keeps_archive_without_publication(
    tmp_path, http
):
    for data in (
        b"bad tar",
        tar_bytes(
            {
                "food-101/meta/train.json": b'{"apple": ["a"]}',
                "food-101/meta/test.json": b'{"apple": ["b"]}',
            }
        ),
    ):
        root = tmp_path / str(len(data))
        calls = http(lambda request, data=data: httpx.Response(200, content=data))
        with pytest.raises((tarfile.ReadError, FileNotFoundError)):
            download.prepare_dataset({"name": "food101", "root": str(root)})
        assert (root / ".downloads/food-101.tar.gz").is_file()
        assert not (root / "meta").exists()
        assert not list(root.glob(".extract-*"))
        assert calls


def test_isia_preparation_and_extraction_failure(tmp_path, http, monkeypatch):
    metadata = zip_bytes(
        {
            "metadata_ISIAFood_500/ISIAFood500_classLabel.txt": b"pear\t1\n",
            "metadata_ISIAFood_500/train_finetune.txt": b"pear/a.png 1\n",
            "metadata_ISIAFood_500/test_finetune.txt": b"pear/b.png 1\n",
        }
    )
    calls = http(
        lambda request: httpx.Response(
            200,
            content=(
                metadata
                if str(request.url).endswith("metadata_ISIAFood_500.zip")
                else b"volume"
            ),
        )
    )
    monkeypatch.setattr(
        download, "_extract_multipart", Mock(side_effect=RuntimeError("extract failed"))
    )
    config = {"name": "isia_food500", "root": str(tmp_path)}
    with pytest.raises(RuntimeError, match="extract failed"):
        download.prepare_dataset(config)
    assert len(calls) == 12
    assert not (tmp_path / "metadata_ISIAFood_500").exists()

    def extract(archive, volumes, staging):
        assert len(volumes) == 11 and all(volume.is_file() for volume in volumes)
        images = staging / "ISIA_Food500/images/pear"
        images.mkdir(parents=True)
        for name in ("a.png", "b.png"):
            (images / name).write_bytes(b"image")

    monkeypatch.setattr(download, "_extract_multipart", extract)
    download.prepare_dataset(config)
    assert len(calls) == 12
    assert (tmp_path / "ISIA_Food500/images/pear/a.png").is_file()
    download.prepare_dataset(config)
    assert len(calls) == 12


def test_missing_multipart_volume_precedes_extraction(tmp_path, monkeypatch):
    run = Mock(side_effect=AssertionError("subprocess"))
    monkeypatch.setattr(download.subprocess, "run", run)
    with pytest.raises(FileNotFoundError, match="Missing archive volume"):
        download._extract_multipart(
            tmp_path / "a.zip", [tmp_path / "a.z01"], tmp_path / "stage"
        )


def test_packaged_7zz_extracts_real_multipart_zip(tmp_path):
    sevenzip, zip_command = shutil.which("7zz"), shutil.which("zip")
    if not sevenzip or not zip_command:
        pytest.skip(
            "requires packaged 7zz and Info-ZIP to construct a split ZIP fixture"
        )
    payload = random.Random(42).randbytes(150_000)
    (tmp_path / "食物.bin").write_bytes(payload)
    subprocess.run(
        [zip_command, "-0", "-s", "64k", "archive.zip", "食物.bin"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    volumes = sorted(tmp_path.glob("archive.z*"))
    assert len(volumes) >= 3
    staging = tmp_path / "stage"
    download._extract_multipart(tmp_path / "archive.zip", volumes, staging)
    assert (staging / "食物.bin").read_bytes() == payload


@pytest.mark.parametrize(
    "listing", ["Path = ../outside\n", "Path = link\nSymbolic Link = target\n"]
)
def test_7zz_unsafe_listing_prevents_extraction(tmp_path, monkeypatch, listing):
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(download.shutil, "which", Mock(return_value="7zz"))
    run = Mock(return_value=Mock(stdout=listing))
    monkeypatch.setattr(download.subprocess, "run", run)
    with pytest.raises(ValueError):
        download._extract_multipart(archive, [archive], tmp_path / "stage")
    assert run.call_count == 1


@pytest.mark.parametrize("name", ["foodx251", "food2k"])
def test_local_provisioning_is_actionable_without_network(tmp_path, monkeypatch, name):
    monkeypatch.setattr(
        download.httpx, "Client", Mock(side_effect=AssertionError("network"))
    )
    with pytest.raises(
        ValueError, match="requires local provisioning.*annot/class_list.txt"
    ):
        download.prepare_dataset({"name": name, "root": str(tmp_path)})
