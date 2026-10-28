"""Local dataset protocols using tiny images; no publisher data or models."""

import csv
import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from datasets import Image as HFImage
from PIL import Image

from internvl.image_dataset import (
    DATASETS,
    FoodImageDataset,
    load_dataset,
    validate_dataset_config,
)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_dataset(root, name):
    """Four references and two queries; MM derives an additional singleton."""
    if name in {"foodx251", "food2k"}:
        write(root / "annot/class_list.txt", "7 apple_PIE\n2 pear\n")
        write(root / "annot/train_info.csv", "a.png,7\nb.png,2\nc.png,7\nd.png,2\n")
        write(root / "annot/val_info.csv", "a.png,7\nb.png,2\n")
    elif name == "food101":
        write(
            root / "meta/train.json",
            json.dumps(
                {
                    "pear": ["pear/b", "pear/d"],
                    "apple_PIE": ["apple_PIE/a", "apple_PIE/c"],
                }
            ),
        )
        write(
            root / "meta/test.json",
            json.dumps({"pear": ["pear/q"], "apple_PIE": ["apple_PIE/q"]}),
        )
    elif name == "isia_food500":
        write(
            root / "metadata_ISIAFood_500/ISIAFood500_classLabel.txt",
            "苹果_派\t7\npear\t2\n",
        )
        write(
            root / "metadata_ISIAFood_500/train_finetune.txt",
            "苹果_派/a 1.png 7\npear/b.png 2\n苹果_派/c.png 7\npear/d.png 2\n",
        )
        write(
            root / "metadata_ISIAFood_500/test_finetune.txt",
            "苹果_派/q.png 7\npear/q.png 2\n",
        )
    else:
        root.mkdir(parents=True, exist_ok=True)
        with (root / "MM-Food-100K.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_url", "dish_name"])
            for i, label in enumerate(
                [
                    " Ａpple  PIE ",
                    "pear",
                    "apple pie",
                    "PEAR",
                    "Apple\tpie",
                    "pear",
                    "unique",
                ]
            ):
                writer.writerow([f"https://fixture.test/{i}.png", label])
    config = {"name": name, "root": str(root)}
    dataset = load_dataset(config)
    for split in dataset.splits.values():
        for row in split:
            path = Path(row["fname"])
            path.parent.mkdir(parents=True, exist_ok=True)
            color = (0, 240, 0) if row["class"].casefold() == "pear" else (240, 0, 0)
            Image.new("RGB", (5, 3), color).save(path)
    return config


@pytest.fixture(params=list(DATASETS))
def dataset_config(tmp_path, request):
    return make_dataset(tmp_path / "raw", request.param)


def test_shared_contract(dataset_config, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(Image, "open", Mock(side_effect=AssertionError("eager decode")))
        dataset = load_dataset(dataset_config)
        assert isinstance(dataset, FoodImageDataset)
        assert set(dataset.splits) == {"train", "validation"}
        assert dataset.test_path == dataset.val_path
        assert dataset.train_set().features["image"] == HFImage(decode=False)
        full = next(iter(dataset.iter_batches("train", 100, decode=False)))
        capped = next(
            iter(dataset.iter_batches("train", 100, n_imgs_per_class=1, decode=False))
        )
        ids = capped["source_id"]
        excluded = next(
            iter(
                dataset.iter_batches(
                    "train",
                    100,
                    n_imgs_per_class=1,
                    excluded_source_ids={ids[0]},
                    decode=False,
                )
            )
        )
        assert excluded["source_id"] == ids[1:]  # no replacement after exclusion
        assert "image" not in full
    expected_split = {
        "food101": "test",
        "isia_food500": "test",
        "mm_food100k": "holdout",
    }.get(dataset_config["name"], "val")
    assert set(dataset.test_set()["source_split"]) == {expected_split}
    for size in (1, 3, 20):
        batches = list(dataset.iter_batches("train", size))
        assert [sid for batch in batches for sid in batch["source_id"]] == full[
            "source_id"
        ]
        assert [len(batch["image"]) for batch in batches] == [
            min(size, len(full["fname"]) - start)
            for start in range(0, len(full["fname"]), size)
        ]
        for batch in batches:
            for image in batch["image"]:
                assert image.size == (5, 3)
                image.close()
    assert all(
        sid.startswith(dataset_config["name"] + ":train:") for sid in full["source_id"]
    )
    assert dataset.train_set().features["image"] == HFImage(decode=False)


def test_mapping_paths_and_order(tmp_path):
    food101 = load_dataset(make_dataset(tmp_path / "food101", "food101"))
    assert food101.class_map == {0: "Apple pie", 1: "Pear"}
    assert list(food101.train_set()["source_id"]) == [
        "food101:train:pear/b.jpg",
        "food101:train:pear/d.jpg",
        "food101:train:apple_PIE/a.jpg",
        "food101:train:apple_PIE/c.jpg",
    ]
    isia = load_dataset(make_dataset(tmp_path / "isia", "isia_food500"))
    assert isia.class_map == {7: "苹果 派", 2: "Pear"}
    assert isia.train_set()[0]["source_id"] == "isia_food500:train:苹果_派/a 1.png"


def test_missing_and_broken_images_fail_only_on_decode(dataset_config):
    dataset = load_dataset(dataset_config)
    path = Path(dataset.test_set()[0]["fname"])
    path.unlink()
    reloaded = load_dataset(dataset_config)
    assert list(reloaded.iter_batches("val", 2, decode=False))
    with pytest.raises(FileNotFoundError):
        next(reloaded.iter_batches("val", 2))
    path.write_text("invalid image")
    with pytest.raises(OSError):
        next(reloaded.iter_batches("val", 2))


def test_mm_determinism_normalization_and_singletons(tmp_path):
    config = make_dataset(tmp_path, "mm_food100k")
    dataset = load_dataset(config)
    assert dataset.get_class_list() == ["apple pie", "pear", "unique"]
    assert Counter(dataset.train_set()["class"]) == {
        "apple pie": 2,
        "pear": 2,
        "unique": 1,
    }
    assert Counter(dataset.test_set()["class"]) == {"apple pie": 1, "pear": 1}
    expected = set(dataset.test_set()["fname"])
    assert not expected & set(dataset.train_set()["fname"])
    assert set(load_dataset(config).test_set()["fname"]) == expected
    path = tmp_path / "MM-Food-100K.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows([rows[0], *reversed(rows[1:])])
    reversed_dataset = load_dataset(config)
    assert set(reversed_dataset.test_set()["fname"]) == expected
    assert list(reversed_dataset.train_set()["fname"]) == list(
        reversed(dataset.train_set()["fname"])
    )
    assert len(load_dataset({**config, "holdout_fraction": 0.99}).train_set()) == 3
    assert len(load_dataset({**config, "holdout_fraction": 0.01}).test_set()) == 2


@pytest.mark.parametrize(
    "body,match",
    [
        ("https://x/a.png, \n", "Empty"),
        ("https://x/a.png,A\nhttps://x/a.png,A\n", "Duplicate"),
        ("https://x/a.png,A\nhttps://y/a.png,A\n", "collision"),
        ("https://x/%2e%2e,A\n", "Invalid"),
        ("https://x/a%2fb.png,A\n", "Invalid"),
    ],
)
def test_invalid_mm_metadata(tmp_path, body, match):
    write(tmp_path / "MM-Food-100K.csv", "image_url,dish_name\n" + body)
    with pytest.raises(ValueError, match=match):
        load_dataset({"name": "mm_food100k", "root": str(tmp_path)})


@pytest.mark.parametrize("name", ["food101", "isia_food500", "food2k"])
@pytest.mark.parametrize(
    "problem", ["duplicate", "escape", "unknown", "malformed", "empty_label"]
)
def test_invalid_annotations(tmp_path, name, problem):
    config = make_dataset(tmp_path, name)
    if name == "food101":
        value = (
            {"pear": ["pear/b", "pear/b"]}
            if problem == "duplicate"
            else {
                "escape": {"pear": ["../outside"]},
                "unknown": {"unknown": ["a"]},
                "malformed": {"pear": "not a list"},
                "empty_label": {"": ["a"]},
            }[problem]
        )
        split = "train" if problem == "empty_label" else "test"
        write(tmp_path / f"meta/{split}.json", json.dumps(value))
    elif name == "isia_food500":
        if problem == "empty_label":
            write(
                tmp_path / "metadata_ISIAFood_500/ISIAFood500_classLabel.txt", "\t7\n"
            )
        else:
            write(
                tmp_path / "metadata_ISIAFood_500/test_finetune.txt",
                {
                    "duplicate": "pear/q.png 2\npear/q.png 2\n",
                    "escape": "../outside 2\n",
                    "unknown": "a.png 99\n",
                    "malformed": "a.png nope\n",
                }[problem],
            )
    else:
        if problem == "empty_label":
            write(tmp_path / "annot/class_list.txt", "7 \n")
        else:
            write(
                tmp_path / "annot/val_info.csv",
                {
                    "duplicate": "a.png,7\na.png,7\n",
                    "escape": "../outside,7\n",
                    "unknown": "a.png,99\n",
                    "malformed": "a.png,nope\n",
                }[problem],
            )
    with pytest.raises((ValueError, KeyError)):
        load_dataset(config)


@pytest.mark.parametrize("name", ["food101", "isia_food500"])
def test_reference_evaluation_overlap(tmp_path, name):
    config = make_dataset(tmp_path, name)
    if name == "food101":
        write(tmp_path / "meta/test.json", json.dumps({"pear": ["pear/b"]}))
    else:
        write(tmp_path / "metadata_ISIAFood_500/test_finetune.txt", "pear/b.png 2\n")
    with pytest.raises(ValueError, match="overlap"):
        load_dataset(config)


def test_symlink_escape_rejected(dataset_config, tmp_path):
    dataset = load_dataset(dataset_config)
    path = Path(dataset.train_set()[0]["fname"])
    path.unlink()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (1, 1)).save(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        load_dataset(dataset_config)


@pytest.mark.parametrize("name", ["food101", "mm_food100k", "isia_food500", "food2k"])
def test_example_configs_keep_taxonomies_separate(name):
    config = yaml.safe_load(Path(f"cfg/classification_{name}.yml").read_text())
    validate_dataset_config(config["dataset"])
    assert config["dataset"]["name"] == name
    assert config["embedding"]["model_id"] == "jinaai/jina-clip-v2"
    assert config["lancedb"]["table"] == f"{name}_whole_image"
    assert config["output_dir"] == f"assets/classification/{name}"
    assert config["descriptions_file"] is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("holdout_fraction", True),
        ("holdout_fraction", 0),
        ("holdout_fraction", 1),
        ("holdout_fraction", float("nan")),
        ("holdout_fraction", "0.2"),
        ("split_seed", True),
        ("split_seed", 1.5),
        ("split_seed", "42"),
    ],
)
def test_config_validation_is_local_free(monkeypatch, option, value):
    monkeypatch.setattr(
        Path, "resolve", Mock(side_effect=AssertionError("filesystem access"))
    )
    with pytest.raises(ValueError, match=option):
        validate_dataset_config(
            {"name": "mm_food100k", "root": "missing", option: value}
        )
    validate_dataset_config({"name": "food101", "root": "missing"})


def test_unsupported_dataset_and_options():
    with pytest.raises(
        ValueError, match="foodx251, food101, mm_food100k, isia_food500, food2k"
    ):
        validate_dataset_config({"name": "unknown", "root": "."})
    with pytest.raises(ValueError, match="Unsupported dataset options"):
        validate_dataset_config({"name": "food101", "root": ".", "split_seed": 42})


@pytest.mark.parametrize("vertex", [False, True])
def test_local_and_vertex_flows(dataset_config, tmp_path, monkeypatch, vertex):
    from internvl.experiment import Experiment
    from internvl.vertex_utils import VertexExperiment

    config = yaml.safe_load(
        Path("cfg/ge_v2.yml" if vertex else "cfg/classification.yml").read_text()
    )
    config.update(
        dataset=dataset_config,
        embed_images=True,
        embed_class_labels=False,
        embed_descriptions=False,
        n_imgs_per_class=1,
        k=1,
        output_dir=str(tmp_path / "output"),
    )
    config["lancedb"]["uri"] = str(tmp_path / "db")
    config["embedding"]["batch_size"] = 2
    calls = []

    def encode(images, *, on_embedding=None):
        calls.extend(images)
        vectors = np.asarray(
            [[1, 0] if image.getpixel((0, 0))[0] > 100 else [0, 1] for image in images],
            dtype=np.float32,
        )
        if on_embedding:
            for i, vector in enumerate(vectors):
                on_embedding(i, vector)
        return vectors

    encoder = Mock(dimension=2, encode_images=Mock(side_effect=encode))
    experiment_class = VertexExperiment if vertex else Experiment
    if vertex:
        config["embedding"]["vertex"]["project"] = "fixture"
    monkeypatch.setattr(experiment_class, "_create_embedder", lambda self: encoder)
    experiment = experiment_class(config)
    experiment.build_vector_database()
    references = len(experiment.dataset.get_class_list())
    assert len(calls) == references
    report = experiment.evaluate()
    assert report["metrics"]["evaluated"] == len(experiment.dataset.test_set())
    assert report["metrics"]["majority_vote_accuracy"] == 1
    assert {row["source_split"] for row in report["predictions"]} == set(
        experiment.dataset.test_set()["source_split"]
    )
    if vertex:
        count = len(calls)
        restarted = experiment_class(config)
        restarted.build_vector_database()
        restarted.evaluate()
        assert len(calls) == count  # reference exclusion and complete validation cache


def test_download_cli_overrides_and_no_tracking(tmp_path, monkeypatch):
    from internvl import cli

    path = tmp_path / "config.yml"
    write(
        path,
        "dataset:\n  name: mm_food100k\n  root: missing\n  holdout_fraction: 0.2\n  split_seed: 42\n",
    )
    prepare = Mock()
    monkeypatch.setattr(cli, "prepare_dataset", prepare)
    monkeypatch.setattr(cli.wandb, "init", Mock(side_effect=AssertionError("tracking")))
    monkeypatch.setattr(
        cli, "Experiment", Mock(side_effect=AssertionError("experiment"))
    )
    cli.main(
        [
            "dataset",
            "download",
            "--config_file",
            str(path),
            "--set",
            "dataset.split_seed=1",
            "--set",
            "dataset.split_seed=5",
        ]
    )
    assert prepare.call_args.args[0]["split_seed"] == 5
    for command in (["dataset", "--help"], ["dataset", "download", "--help"]):
        with pytest.raises(SystemExit) as error:
            cli.main(command)
        assert error.value.code == 0
    with pytest.raises(SystemExit):
        cli.main(
            [
                "dataset",
                "download",
                "--config_file",
                str(path),
                "--set",
                "dataset.unknown=2",
            ]
        )
    assert prepare.call_count == 1


def test_invalid_dataset_precedes_tracking(tmp_path, monkeypatch):
    from internvl import cli

    config = yaml.safe_load(Path("cfg/classification.yml").read_text())
    config["dataset"]["name"] = "unsupported"
    monkeypatch.setattr(cli.wandb, "init", Mock(side_effect=AssertionError("tracking")))
    with pytest.raises(ValueError, match="Unsupported dataset"):
        cli._run_experiment(config, "evaluate")
