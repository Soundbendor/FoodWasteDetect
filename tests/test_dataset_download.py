"""CPU checks for dataset archive download and extraction with a mock HTTP transport."""

import io
import tarfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import yaml

from classification.cli import main
from classification.datasets.util import extract, fetch_archive
from classification.datasets.base import FoodImageDataset
from classification.datasets.food101 import Food101ImageDataset
from classification.datasets.foodx251 import FoodX251ImageDataset
from classification.datasets import isia500
from classification.datasets.isia500 import ISIAFood500ImageDataset
from classification.datasets.uecfood256 import UECFood256ImageDataset
from classification.datasets.vegfru import VegFruImageDataset

FOODX251_FILES = {
    "annot": {"class_list.txt": "0 apple_pie\n1 pear\n", "train_info.csv": "a.png,0\nb.png,1\n", "val_info.csv": "c.png,1\n"},
    "train": {"train_set/a.png": "x", "train_set/b.png": "x"},
    "val": {"val_set/c.png": "x"},
}


FOOD101_FILES = {
    "meta/classes.txt": "apple_pie\nbeef_tartare\n",
    "meta/train.txt": "apple_pie/1\nbeef_tartare/2\n",
    "meta/test.txt": "beef_tartare/3\n",
    "images/apple_pie/1.jpg": "x",
    "images/beef_tartare/2.jpg": "x",
    "images/beef_tartare/3.jpg": "x",
}


# Mirrors dataset256.zip: UECFOOD256/{README.txt,category.txt,<id>/{bb_info.txt,*.jpg}}.
# IDs 2 and 3 share a canonical name, as 40 and 67 ("omelet") do upstream; 99.jpg is
# listed under classes 1 and 3.
UECFOOD256_FILES = {
    "README.txt": "readme",
    "category.txt": "id\tname\n1\trice\n2\tomelet\n3\tOmelet\n",
    **{f"1/{n}.jpg": "x" for n in range(10, 20)},
    "1/99.jpg": "x",
    "1/bb_info.txt": "img x1 y1 x2 y2\n10 0 0 1 1\n",
    "2/20.jpg": "x",
    "3/30.jpg": "x",
    "3/99.jpg": "x",
}


# Mirrors vegfru.tar.gz: vegfru/{classid_classname.csv,veg200_images/,archive.zip}, where
# archive.zip holds fru92_images/ (verified from the archive head). The list directory
# and its "<path> <label id>" lines follow the published layout and are unverified.
VEGFRU_CSV = "class_id,class_name\n0,Chinese_artichoke\n1,shepherd's_purse\n2,apple\n"
VEGFRU_LISTS = {
    "vegfru_list/vegfru_train.txt": (
        "veg200_images/Chinese_artichoke/v_01_01_0001.jpg 0\n"
        "veg200_images/shepherd's_purse/v_02_01_0001.jpg 1\n"
        "fru92_images/apple/f_01_01_0001.jpg 2\n"
    ),
    "vegfru_list/vegfru_val.txt": "not a list\n",
    "vegfru_list/vegfru_test.txt": "fru92_images/apple/f_01_01_0002.jpg 2\n",
}
VEGFRU_VEG = {
    "veg200_images/Chinese_artichoke/v_01_01_0001.jpg": "x",
    "veg200_images/shepherd's_purse/v_02_01_0001.jpg": "x",
}
VEGFRU_FRU = {"fru92_images/apple/f_01_01_0001.jpg": "x", "fru92_images/apple/f_01_01_0002.jpg": "x"}


# Mirrors metadata_ISIAFood_500.zip: tab-separated name/id labels and "<class>/<file> <id>"
# split lists. Names keep case, parentheses, and non-ASCII text.
ISIA_META = {
    "ISIAFood500_classLabel.txt": "Apple_pie\t1\nCrème_brûlée_(Burnt_cream)\t2\n",
    "train_finetune.txt": "Apple_pie/Apple_pie_0001.jpg 1\nCrème_brûlée_(Burnt_cream)/Crème_brûlée_(Burnt_cream)_0001.jpg 2\n",
    "test_finetune.txt": "Crème_brûlée_(Burnt_cream)/Crème_brûlée_(Burnt_cream)_0002.jpg 2\n",
}
ISIA_IMAGES = [
    "Apple_pie/Apple_pie_0001.jpg",
    "Crème_brûlée_(Burnt_cream)/Crème_brûlée_(Burnt_cream)_0001.jpg",
    "Crème_brûlée_(Burnt_cream)/Crème_brûlée_(Burnt_cream)_0002.jpg",
]
ISIA_URL = "http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/"


def _tar(files: dict[str, str], mode: str = "w") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode=mode) as bundle:
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


def test_food101_loads_miniature_tree(tmp_path):
    for name, text in FOOD101_FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    dataset = Food101ImageDataset(tmp_path)
    assert dataset.get_class_list() == ["Apple pie", "Beef tartare"]
    train = dataset.train_set()
    assert train["source_id"] == ["food101:train:apple_pie/1.jpg", "food101:train:beef_tartare/2.jpg"]
    assert train["class"] == ["Apple pie", "Beef tartare"]
    assert train["fname"][0] == str(tmp_path.resolve() / "images/apple_pie/1.jpg")
    validation = dataset.validation_set()
    assert validation["source_id"] == ["food101:test:beef_tartare/3.jpg"]
    assert validation["source_split"] == ["test"]

    (tmp_path / "meta/test.txt").write_text("pear/4\n")
    with pytest.raises(ValueError, match="missing from classes"):
        Food101ImageDataset(tmp_path)


def test_food101_download_flattens_and_is_idempotent(tmp_path):
    requests: list[str] = []
    body = _tar({f"food-101/{name}": text for name, text in FOOD101_FILES.items()}, mode="w:gz")
    client = _client({"cvl/food-101.tar.gz": body}, requests)
    Food101ImageDataset.download(tmp_path, client=client)
    assert requests == ["https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"]
    assert not list((tmp_path / ".downloads").iterdir())
    assert not (tmp_path / "food-101").exists()

    dataset = Food101ImageDataset(tmp_path)
    assert len(dataset.train_set()) == 2 and dataset.validation_set()["class"] == ["Beef tartare"]

    Food101ImageDataset.download(tmp_path, client=client)
    assert len(requests) == 1


def test_uecfood256_seeded_split_drops_multi_listed_images(tmp_path):
    for name, text in UECFOOD256_FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    dataset = UECFood256ImageDataset(tmp_path)
    assert dataset.get_class_list() == ["Rice", "Omelet"]
    train, validation = dataset.train_set(), dataset.validation_set()
    rice = lambda split: {i for i, c in zip(split["source_id"], split["class"], strict=True) if c == "Rice"}
    assert len(rice(validation)) == 2 and len(rice(train)) == 8
    assert {i.split(":")[2] for i in rice(train) | rice(validation)} == {f"1/{n}.jpg" for n in range(10, 20)}
    assert train["source_id"][-2:] == ["uecfood256:train:2/20.jpg", "uecfood256:train:3/30.jpg"]
    assert set(validation["source_split"]) == {"validation"}
    assert train["fname"][-1] == str(tmp_path.resolve() / "3/30.jpg")
    assert UECFood256ImageDataset(tmp_path).validation_set()["source_id"] == validation["source_id"]

    (tmp_path / "category.txt").write_text("id\tname\n1\trice\n1\tbread\n")
    with pytest.raises(ValueError, match="unique class IDs"):
        UECFood256ImageDataset(tmp_path)


def test_uecfood256_download_flattens_and_is_idempotent(tmp_path):
    requests: list[str] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, text in UECFOOD256_FILES.items():
            bundle.writestr(f"UECFOOD256/{name}", text)
    client = _client({"datasets/kng-research/uecfood256/resolve/main/dataset256.zip": buffer.getvalue()}, requests)
    UECFood256ImageDataset.download(tmp_path, client=client)
    assert requests == ["https://huggingface.co/datasets/kng-research/uecfood256/resolve/main/dataset256.zip"]
    assert not list((tmp_path / ".downloads").iterdir())
    assert not (tmp_path / "UECFOOD256").exists()

    dataset = UECFood256ImageDataset(tmp_path)
    assert len(dataset.train_set()) + len(dataset.validation_set()) == 12

    UECFood256ImageDataset.download(tmp_path, client=client)
    assert len(requests) == 1


def test_vegfru_loads_official_train_and_test_lists(tmp_path):
    files = {"classid_classname.csv": VEGFRU_CSV, **VEGFRU_LISTS, **VEGFRU_VEG, **VEGFRU_FRU}
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    dataset = VegFruImageDataset(tmp_path)
    assert dataset.get_class_list() == ["Chinese artichoke", "Shepherd's purse", "Apple"]
    train = dataset.train_set()
    assert train["class"] == ["Chinese artichoke", "Shepherd's purse", "Apple"]
    assert train["source_id"][2] == "vegfru:train:fru92_images/apple/f_01_01_0001.jpg"
    assert train["fname"][0] == str(tmp_path.resolve() / "veg200_images/Chinese_artichoke/v_01_01_0001.jpg")
    validation = dataset.validation_set()
    assert validation["source_id"] == ["vegfru:test:fru92_images/apple/f_01_01_0002.jpg"]
    assert validation["source_split"] == ["test"]

    (tmp_path / "vegfru_list/vegfru_test.txt").write_text("fru92_images/pear/f_02_01_0001.jpg 3\n")
    with pytest.raises(ValueError, match="missing from classid"):
        VegFruImageDataset(tmp_path)


def test_vegfru_download_unpacks_nested_zip_and_is_idempotent(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, text in {**VEGFRU_FRU, **VEGFRU_LISTS}.items():
            bundle.writestr(name, text)
    members = {f"vegfru/{n}": t for n, t in {"classid_classname.csv": VEGFRU_CSV, **VEGFRU_VEG}.items()}
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as bundle:
        for name, data in [*((n, t.encode()) for n, t in members.items()), ("vegfru/archive.zip", buffer.getvalue())]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))
    requests: list[str] = []
    client = _client({"datasets/xedwin23x/VegFru/resolve/main/vegfru.tar.gz": tar_buffer.getvalue()}, requests)
    VegFruImageDataset.download(tmp_path, client=client)
    assert requests == ["https://huggingface.co/datasets/xedwin23x/VegFru/resolve/main/vegfru.tar.gz"]
    assert not list((tmp_path / ".downloads").iterdir())
    assert not (tmp_path / "vegfru").exists() and not (tmp_path / "archive.zip").exists()

    dataset = VegFruImageDataset(tmp_path)
    assert len(dataset.train_set()) == 3 and dataset.validation_set()["class"] == ["Apple"]

    VegFruImageDataset.download(tmp_path, client=client)
    assert len(requests) == 1


def test_isia500_loads_tab_separated_labels_and_finetune_lists(tmp_path):
    for name, text in ISIA_META.items():
        (tmp_path / "metadata_ISIAFood_500").mkdir(exist_ok=True)
        (tmp_path / "metadata_ISIAFood_500" / name).write_text(text)
    dataset = ISIAFood500ImageDataset(tmp_path)
    assert dataset.get_class_list() == ["Apple pie", "Crème brûlée (Burnt cream)"]
    train = dataset.train_set()
    assert train["class"] == ["Apple pie", "Crème brûlée (Burnt cream)"]
    assert train["source_id"][0] == "isia500:train:Apple_pie/Apple_pie_0001.jpg"
    assert train["fname"][1] == str(tmp_path.resolve() / "images" / ISIA_IMAGES[1])
    validation = dataset.validation_set()
    assert validation["source_id"] == [f"isia500:test:{ISIA_IMAGES[2]}"]
    assert validation["source_split"] == ["test"]

    (tmp_path / "metadata_ISIAFood_500/test_finetune.txt").write_text("Pear/Pear_0001.jpg 3\n")
    with pytest.raises(ValueError, match="missing from ISIAFood500"):
        ISIAFood500ImageDataset(tmp_path)


def _isia_client(requests: list[str], **overrides: httpx.Response) -> httpx.Client:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, text in ISIA_META.items():
            bundle.writestr(f"metadata_ISIAFood_500/{name}", text)
    bodies = {"metadata_ISIAFood_500.zip": buffer.getvalue()}
    bodies |= {part: part.encode() for part in isia500.PARTS}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        name = request.url.path.rsplit("/", 1)[1]
        return overrides.get(name) or httpx.Response(200, content=bodies[name])

    return httpx.Client(transport=httpx.MockTransport(handle))


def _fake_7z(command, check):
    output = Path(command[3].removeprefix("-o"))
    for name in ISIA_IMAGES:
        (output / "images" / name).parent.mkdir(parents=True, exist_ok=True)
        (output / "images" / name).write_text("x")


def test_isia500_download_fetches_parts_runs_7z_and_is_idempotent(tmp_path):
    requests: list[str] = []
    client = _isia_client(requests)
    with (
        patch("classification.datasets.isia500.shutil.which", return_value="/bin/7z"),
        patch("classification.datasets.isia500.subprocess.run", side_effect=_fake_7z) as run,
    ):
        ISIAFood500ImageDataset.download(tmp_path, client=client)
        ISIAFood500ImageDataset.download(tmp_path, client=client)
    parts = [f"ISIA_Food500.z{i:02d}" for i in range(1, 11)] + ["ISIA_Food500.zip"]
    assert requests == [ISIA_URL + n for n in ["metadata_ISIAFood_500.zip", *parts]]
    root = tmp_path.resolve()
    run.assert_called_once_with(
        ["/bin/7z", "x", "-y", f"-o{root}", str(root / ".downloads/ISIA_Food500.zip")], check=True
    )
    assert not list((tmp_path / ".downloads").iterdir())

    dataset = ISIAFood500ImageDataset(tmp_path)
    assert len(dataset.train_set()) == 2 and dataset.validation_set()["fname"] == [
        str(root / "images" / ISIA_IMAGES[2])
    ]


def test_isia500_download_rejects_truncated_part(tmp_path):
    requests: list[str] = []
    short = httpx.Response(200, headers={"Content-Length": "999"}, content=b"short")
    client = _isia_client(requests, **{"ISIA_Food500.z03": short})
    with (
        patch("classification.datasets.isia500.shutil.which", return_value="/bin/7z"),
        patch("classification.datasets.isia500.subprocess.run") as run,
        pytest.raises(ValueError, match="Content-Length 999"),
    ):
        ISIAFood500ImageDataset.download(tmp_path, client=client)
    run.assert_not_called()
    downloads = {p.name for p in (tmp_path / ".downloads").iterdir()}
    assert downloads == {"metadata_ISIAFood_500.zip", "ISIA_Food500.z01", "ISIA_Food500.z02"}


def test_isia500_download_requires_7z(tmp_path):
    requests: list[str] = []
    with (
        patch("classification.datasets.isia500.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="Pixi environment"),
    ):
        ISIAFood500ImageDataset.download(tmp_path, client=_isia_client(requests))
    assert requests == []


def test_fetch_failure_leaves_no_archive(tmp_path):
    with pytest.raises(httpx.HTTPStatusError):
        fetch_archive("https://host/missing.tar", tmp_path, _client({}, []))
    assert not (tmp_path / ".downloads/missing.tar").exists()


def test_completed_archive_is_reused(tmp_path):
    requests: list[str] = []
    client = _client({"a.tar": b"data"}, requests)
    first = fetch_archive("https://host/a.tar", tmp_path, client)
    assert fetch_archive("https://host/a.tar", tmp_path, client) == first
    assert first.read_bytes() == b"data" and len(requests) == 1


def test_extract_zip_and_reject_tar_escape(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("dir/x.txt", "x")
    extract(archive, tmp_path / "out")
    assert (tmp_path / "out/dir/x.txt").read_text() == "x"

    escape = tmp_path / "bad.tar"
    escape.write_bytes(_tar({"../escaped.txt": "x"}))
    with pytest.raises(tarfile.FilterError):
        extract(escape, tmp_path / "bad")
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
        patch.dict("classification.util.DATASETS", {"foodx251": Fake}),
        patch("classification.tracking.wandb.init") as initialize,
    ):
        main(["download_dataset", "--config_file", str(config)])
    assert Fake.roots == ["data/x"]
    initialize.assert_not_called()
