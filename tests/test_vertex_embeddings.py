"""Vertex contracts using the real SDK and an in-memory HTTP transport."""

import base64
import json
import threading
from collections import Counter
from io import BytesIO
from pathlib import Path
from time import sleep
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pytest
from config_helpers import config_data, updated
from classification.experiments.config import ExperimentConfig
import yaml
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import errors
from PIL import Image

from classification.encoders.vertex import VertexEmbeddingModel


@pytest.fixture
def transport(monkeypatch):
    real_client = genai.Client
    requests, clients, options = [], [], []

    def install(handler):
        def create(**kwargs):
            options.append(kwargs)
            http_options = kwargs["http_options"].model_copy(deep=True)
            # Exercise SDK retry selection/counts without wall-clock backoff.
            http_options.retry_options.initial_delay = 0
            http_options.retry_options.jitter = 0
            http_options.retry_options.max_delay = 0

            def handle(request):
                requests.append(request)
                return handler(request)

            http_options.client_args = {"transport": httpx.MockTransport(handle)}
            credentials = AnonymousCredentials()
            credentials.token = "synthetic-test-token"
            client = real_client(
                **dict(kwargs, http_options=http_options),
                credentials=credentials,
            )
            clients.append(client._api_client._httpx_client)
            return client

        monkeypatch.setattr("classification.encoders.vertex.genai.Client", create)
        return requests, clients, options

    return install


def response(value=1, dimension=128):
    return httpx.Response(200, json={"embedding": {"values": [value] * dimension}})


def model(**kwargs):
    return VertexEmbeddingModel(
        **dict(project="fixture-project", truncate_dim=128, **kwargs)
    )


@pytest.mark.parametrize("prompt", [False, True])
def test_text_order_concurrency_and_request_contract(transport, prompt):
    lock = threading.Lock()
    active = peak = 0
    barrier = threading.Barrier(2)

    def handle(request):
        nonlocal active, peak
        body = json.loads(request.content)
        text = body["content"]["parts"][0]["text"]
        number = int(text.rsplit(" ", 1)[-1])
        with lock:
            active += 1
            peak = max(peak, active)
        if number < 2:
            barrier.wait(timeout=5)
        sleep(0.02 if number == 0 else 0.001)
        with lock:
            active -= 1
        return response(number + 1)

    requests, clients, options = transport(handle)
    encoder = model(max_concurrency=2, classification_prompt=prompt)
    vectors = encoder.encode_text(str(i) for i in range(5))
    np.testing.assert_array_equal(vectors[:, 0], [1, 2, 3, 4, 5])
    assert vectors.dtype == np.float32
    assert peak == 2
    assert len(clients) == 1 and clients[0].is_closed
    for request in requests:
        assert "/v1/projects/fixture-project/locations/global/" in str(request.url)
        assert str(request.url).endswith("/models/gemini-embedding-2:embedContent")
        body = json.loads(request.content)
        assert body["embedContentConfig"]["outputDimensionality"] == 128
        assert "taskType" not in body
        assert len(body["content"]["parts"]) == 1
        assert (
            body["content"]["parts"][0]["text"].startswith(
                "task: classification | query: "
            )
            == prompt
        )
    opts = options[0]["http_options"]
    assert opts.timeout == 60000
    assert opts.retry_options.attempts == 5
    assert (
        opts.retry_options.initial_delay,
        opts.retry_options.exp_base,
        opts.retry_options.jitter,
        opts.retry_options.max_delay,
    ) == (1, 2, 1, 60)


@pytest.mark.parametrize("prompt", [False, True])
def test_images_are_rgb_png_without_metadata(transport, tmp_path, prompt):
    import base64

    requests, clients, _ = transport(lambda request: response())
    image = Image.new("RGBA", (7, 9), (100, 20, 30, 50))
    filename = tmp_path / "private-class-name.png"
    image.save(filename)
    encoder = model(classification_prompt=prompt)
    assert encoder.encode_images([image, filename]).shape == (2, 128)
    assert image.getpixel((0, 0)) == (100, 20, 30, 50)
    for request in requests:
        assert b"private-class-name" not in request.content
        parts = json.loads(request.content)["content"]["parts"]
        assert len(parts) == (2 if prompt else 1)
        if prompt:
            assert parts[0] == {"text": "task: classification | query: "}
        data = parts[-1]["inlineData"]
        assert data["mime_type"] == "image/png"
        with Image.open(BytesIO(base64.b64decode(data["data"]))) as decoded:
            assert decoded.mode == "RGB" and decoded.size == (7, 9)
            assert decoded.getpixel((0, 0)) == (100, 20, 30)
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("dimension", [None, 128, 3072])
def test_dimensions_and_empty_inputs(transport, dimension):
    selected = dimension or 3072
    requests, clients, _ = transport(lambda request: response(dimension=selected))
    encoder = VertexEmbeddingModel(project="fixture-project", truncate_dim=dimension)
    assert encoder.encode_text([]).shape == (0, selected)
    assert encoder.encode_images([]).shape == (0, selected)
    assert not clients
    assert encoder.encode_text(["class"]).shape == (1, selected)
    assert (
        json.loads(requests[0].content)["embedContentConfig"]["outputDimensionality"]
        == selected
    )


@pytest.mark.parametrize(
    "setting,value",
    [
        ("project", None),
        ("project", ""),
        ("project", " "),
        ("location", 1),
        ("location", ""),
        ("max_concurrency", True),
        ("max_concurrency", 0),
        ("max_attempts", 0),
        ("max_attempts", 1.5),
        ("request_timeout", True),
        ("request_timeout", 0),
        ("request_timeout", float("inf")),
        ("request_timeout", float("nan")),
        ("request_timeout", "60"),
        ("classification_prompt", 1),
        ("truncate_dim", True),
        ("truncate_dim", 127),
        ("truncate_dim", 3073),
        ("truncate_dim", 128.0),
    ],
)
def test_invalid_settings_before_authentication(setting, value):
    with patch("classification.encoders.vertex.genai.Client") as client:
        with pytest.raises(ValueError):
            VertexEmbeddingModel(**dict({"project": "fixture"}, **{setting: value}))
        client.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"embeddings": []},
        {"embeddings": [{"values": [1] * 128}] * 2},
        {"embeddings": [{"values": [1] * 127}]},
        {"embeddings": [{"values": [0] * 128}]},
        {"embeddings": [{"values": None}]},
        {"embeddings": [{"values": ["NaN"] * 128}]},
        {"embeddings": [{"values": ["Infinity"] * 128}]},
    ],
)
def test_invalid_response_closes_client(transport, payload):
    if "embeddings" in payload:
        payload = {
            "predictions": [{"embeddings": item} for item in payload["embeddings"]]
        }
    requests, clients, _ = transport(lambda request: httpx.Response(200, json=payload))
    with pytest.raises(ValueError):
        model().encode_text(["class"])
    assert len(requests) == 1 and clients[0].is_closed


@pytest.mark.parametrize(
    "failure", [408, 429, 500, 502, 503, 504, "connect", "timeout"]
)
@pytest.mark.parametrize("recover", [False, True])
def test_sdk_retries(transport, failure, recover):
    attempts = 0

    def handle(request):
        nonlocal attempts
        attempts += 1
        if recover and attempts == 3:
            return response()
        if failure == "connect":
            raise httpx.ConnectError("synthetic", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic", request=request)
        return httpx.Response(
            failure, json={"error": {"code": failure, "message": "synthetic"}}
        )

    _, clients, _ = transport(handle)
    encoder = model(max_attempts=3)
    if recover:
        assert encoder.encode_text(["class"]).shape == (1, 128)
    else:
        with pytest.raises((errors.APIError, httpx.ConnectError, httpx.ReadTimeout)):
            encoder.encode_text(["class"])
    assert attempts == 3 and clients[0].is_closed


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_nonretryable_errors(transport, status):
    requests, clients, _ = transport(
        lambda request: httpx.Response(
            status, json={"error": {"code": status, "message": "synthetic"}}
        )
    )
    with pytest.raises(errors.APIError):
        model().encode_text(["class"])
    assert len(requests) == 1 and clients[0].is_closed


@pytest.mark.parametrize("failure", [RuntimeError("failed"), KeyboardInterrupt()])
def test_failure_cancels_queue_and_waits_for_workers(transport, monkeypatch, failure):
    _, clients, _ = transport(lambda request: response())
    encoder = model(max_concurrency=2)
    barrier = threading.Barrier(2)
    completed = threading.Event()
    calls = []

    def request(client, content):
        text = content.parts[0].text
        calls.append(text)
        if text == "0":
            barrier.wait(timeout=5)
            raise failure
        if text == "1":
            barrier.wait(timeout=5)
        sleep(0.03)
        completed.set()
        return np.ones(128, dtype=np.float32)

    monkeypatch.setattr(encoder, "_request", request)
    with pytest.raises(type(failure)):
        encoder.encode_text(map(str, range(100)))
    assert completed.is_set() and len(calls) < 100
    assert clients[0].is_closed


def test_routing_and_local_settings():
    from classification.experiments.classification import Experiment

    experiment = object.__new__(Experiment)
    experiment._embedder = None
    experiment.batch_size = 2
    experiment.config = ExperimentConfig.model_validate(config_data(**{
        "embedding": {
            "model_id": "google/ge-v2",
            "device": "cpu",
            "vertex": {"project": "fixture-project"},
        }
    }))
    with patch("classification.experiments.classification.EmbeddingModel") as local:
        assert isinstance(experiment.embedder, VertexEmbeddingModel)
        assert experiment.embedder is experiment.embedder
        local.assert_not_called()
    from classification.experiments.classification import Experiment

    local_config = experiment.config
    experiment = object.__new__(Experiment)
    experiment.config = local_config
    experiment.batch_size = 2
    experiment._embedder = None
    experiment.config = updated(experiment.config, "embedding.model_id", "openai/clip-vit-base-patch32")
    with patch("classification.experiments.classification.EmbeddingModel") as local:
        assert experiment.embedder is local.return_value
        local.assert_called_once_with(
            model_id="openai/clip-vit-base-patch32", device="cpu", truncate_dim=None, batch_size=2
        )


def test_yaml_cli_overrides_and_defaults():
    from classification.cli import main

    path = Path("cfg/ge_v2.yml")
    original = path.read_text()
    config = yaml.safe_load(original)
    baseline = yaml.safe_load(Path("cfg/classification.yml").read_text())
    for key in baseline.keys() - {"embedding", "lancedb", "output_dir", "cache_validation_embeddings"}:
        assert config[key] == baseline[key]
    assert config["lancedb"]["table"] == "foodx251_ge_v2"
    assert config["output_dir"] == "assets/classification/ge_v2"
    with (
        patch("classification.cli.wandb.init"),
        patch("classification.cli.Tracker"),
        patch("classification.cli.Experiment") as experiment,
    ):
        main(
            [
                "experiment",
                "run",
                "--config_file",
                str(path),
                "--set",
                "embedding.vertex.project=fixture-project",
                "--set",
                "embedding.vertex.max_concurrency=2",
                "--set",
                "embedding.vertex.classification_prompt=true",
                "--set",
                "embedding.truncate_dim=128",
                "--set",
                "embed_images=true",
                "--set",
                "cache_validation_embeddings=false",
                "--set",
                "lancedb.validation_table=custom_validation",
            ]
        )
        effective = experiment.call_args.args[0]
        assert effective.embedding.vertex.project == "fixture-project"
        assert effective.embedding.vertex.max_concurrency == 2
        assert effective.embedding.vertex.classification_prompt is True
        assert effective.embedding.truncate_dim == 128
        assert effective.embed_images is True
        assert effective.cache_validation_embeddings is False
        assert effective.lancedb.validation_table == "custom_validation"
    assert path.read_text() == original


def test_experiment_references_cache_and_failed_evaluation(transport, tmp_path):
    from classification.experiments.classification import Experiment

    root = tmp_path / "dataset"
    (root / "annot").mkdir(parents=True)
    (root / "annot/class_list.txt").write_text("0 apple\n")
    for split in ("train", "val"):
        folder = root / split / f"{split}_set"
        folder.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (3, 4), "red").save(folder / f"{i}.png")
        (root / f"annot/{split}_info.csv").write_text("0.png,0\n1.png,0\n2.png,0\n")
    descriptions = tmp_path / "descriptions.json"
    descriptions.write_text(
        json.dumps(
            {"descriptions": {"Apple": ["red fruit", "round fruit", "sweet fruit"]}}
        )
    )
    config = yaml.safe_load(Path("cfg/ge_v2.yml").read_text())
    config.update(
        dataset={"name": "foodx251", "root": str(root)},
        output_dir=str(tmp_path / "output"),
        embed_images=True,
        embed_class_labels=True,
        descriptions_file=str(descriptions),
    )
    config["embedding"].update(batch_size=2, truncate_dim=128)
    config["embedding"]["vertex"]["project"] = "fixture-project"
    config["lancedb"]["uri"] = str(tmp_path / "lancedb")
    requests, clients, _ = transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(config))
    summary = experiment.build_vector_database().model_dump(mode="json")
    assert summary["source_counts"] == {
        "class_name": 1,
        "generated_description": 3,
        "training_image": 3,
    }
    assert len(requests) == 7
    assert all(client.is_closed for client in clients)
    with patch("classification.encoders.vertex.genai.Client") as client:
        cached = Experiment(ExperimentConfig.model_validate(config))
        assert cached.build_vector_database().model_dump(mode="json")["source_counts"] == summary["source_counts"]
        assert cached._embedder is None
        client.assert_not_called()
    report = experiment.evaluate().model_dump(mode="json")
    assert report["metrics"]["evaluated"] == 3
    assert len(report["predictions"]) == 3
    assert len(requests) == 10
    previous = set(experiment.output.glob("evaluation_*/report.json"))
    failures = 0

    def fail_second_batch(request):
        nonlocal failures
        failures += 1
        if failures == 3:
            return httpx.Response(
                400, json={"error": {"code": 400, "message": "synthetic"}}
            )
        return response()

    experiment.config = updated(experiment.config, "cache_validation_embeddings", False)
    transport(fail_second_batch)
    with pytest.raises(errors.APIError):
        experiment.evaluate()
    assert failures == 3
    assert set(experiment.output.glob("evaluation_*/report.json")) == previous
    assert all(client.is_closed for client in clients)


@pytest.fixture
def resume_config(tmp_path):
    root = tmp_path / "dataset"
    (root / "annot").mkdir(parents=True)
    (root / "annot/class_list.txt").write_text("0 apple\n")
    for split in ("train", "val"):
        folder = root / split / f"{split}_set"
        folder.mkdir(parents=True)
        for i in range(13):
            Image.new("RGB", (2, 2), (i + 1, 0, 0)).save(folder / f"{i}.png")
        (root / f"annot/{split}_info.csv").write_text(
            "".join(f"{i}.png,0\n" for i in range(13))
        )
    descriptions = tmp_path / "descriptions.json"
    descriptions.write_text(json.dumps({"descriptions": {"Apple": ["red", "round"]}}))
    config = yaml.safe_load(Path("cfg/ge_v2.yml").read_text())
    config.update(dataset={"name": "foodx251", "root": str(root)},
                  output_dir=str(tmp_path / "output"), embed_images=True,
                  embed_class_labels=True, descriptions_file=str(descriptions))
    config["embedding"].update(batch_size=20, truncate_dim=128)
    config["embedding"]["vertex"].update(project="fixture", max_attempts=1)
    config["lancedb"]["uri"] = str(tmp_path / "db")
    return config


def test_index_builder_direct_build_and_complete_resume(transport, resume_config):
    from classification.experiments.classification import Experiment

    requests, _, _ = transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(resume_config))
    builder = experiment._index_builder()
    index, counts = builder.build_reference_index()
    assert counts == {"class_name": 1, "generated_description": 2, "training_image": 13}
    cache = builder.build_validation_table(resume_config["lancedb"]["validation_table"])
    assert len(cache.get_record_ids()) == 13
    before = len(requests)
    resumed = Experiment(ExperimentConfig.model_validate(resume_config))
    resumed_builder = resumed._index_builder()
    resumed_index, resumed_counts = resumed_builder.build_reference_index()
    resumed_builder.build_validation_table(resume_config["lancedb"]["validation_table"])
    assert resumed_index.name == index.name
    assert resumed_counts == counts
    assert len(requests) == before
    assert resumed._embedder is None


def test_resume_all_sources_and_complete_validation(transport, resume_config, monkeypatch):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.schemas import PointRecord
    from classification.experiments.classification import Experiment

    config = resume_config
    config["n_imgs_per_class"] = 12
    index = LanceDBIndex.create(config["lancedb"]["uri"], config["lancedb"]["table"], 128)
    initial = [PointRecord("Apple", "class_name", "Apple", "Apple"),
               PointRecord("Apple", "generated_description", "Apple:1", "round")]
    initial += [PointRecord("Apple", "training_image", f"foodx251:train:{i}.png", "old")
                for i in range(11)]
    index.append(initial, np.ones((13, 128), dtype=np.float32))
    old_ids = [r.id for r in initial]
    requests, _, _ = transport(lambda request: response())
    tracker = Mock()
    experiment = Experiment(ExperimentConfig.model_validate(config), tracker=tracker)
    summary = experiment.build_vector_database().model_dump(mode="json")
    tracker.progress.assert_not_called()
    assert len(requests) == 2  # one description and the twelfth capped image
    assert summary["source_counts"] == {
        "class_name": 1, "generated_description": 2, "training_image": 12}
    index = LanceDBIndex.open(index.uri, index.name)
    assert len(index.get_record_ids()) == 15
    assert set(old_ids) <= {r["id"] for r in index.table.search().limit(None).to_list()}
    report = experiment.evaluate().model_dump(mode="json")
    assert len(requests) == 15
    assert [r["source_id"] for r in report["predictions"]] == [
        f"foodx251:val:{i}.png" for i in range(13)]
    assert {kind for kind, _ in index.get_record_ids()} == {
        "class_name", "generated_description", "training_image"}
    (experiment.output / "index.json").unlink()
    # Cached runs must work even when all image files are missing.
    for path in Path(config["dataset"]["root"]).glob("*/*_set/*.png"):
        path.unlink()
    config["embedding"]["vertex"]["project"] = None
    with patch("classification.encoders.vertex.genai.Client", side_effect=AssertionError("client")):
        cached = Experiment(ExperimentConfig.model_validate(config))
        assert cached.build_vector_database().model_dump(mode="json")["source_counts"] == summary["source_counts"]
        config["k"] = 1
        assert cached.evaluate().model_dump(mode="json")["metrics"]["evaluated"] == 13
        assert cached._embedder is None


@pytest.mark.parametrize("stage", ["reference", "validation"])
@pytest.mark.parametrize("batch_size, committed", [(20, 0), (2, 2)])
def test_failed_batch_restart_only_unsaved(transport, resume_config, stage, batch_size, committed):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.experiments.classification import Experiment

    config = resume_config
    config["embedding"]["batch_size"] = batch_size
    transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(config))
    if stage == "validation":
        experiment.build_vector_database()
    calls = 0

    def fail(request):
        nonlocal calls
        calls += 1
        if calls == (3 if committed else 2):
            return httpx.Response(400, json={"error": {"code": 400, "message": "failed"}})
        return response()

    transport(fail)
    action = experiment.evaluate if stage == "validation" else experiment.build_vector_database
    with pytest.raises(errors.APIError):
        action()
    assert not list(experiment.output.glob("evaluation_*/report.json"))
    table = config["lancedb"]["validation_table" if stage == "validation" else "table"]
    index = LanceDBIndex.open(config["lancedb"]["uri"], table)
    saved = set(index.get_record_ids())
    assert len(saved) == committed
    expected = {}
    if stage == "reference":
        expected.update({("class_name", "Apple"): "Apple",
                         ("generated_description", "Apple:0"): "red",
                         ("generated_description", "Apple:1"): "round"})
    split = "val" if stage == "validation" else "train"
    kind = "validation_image" if stage == "validation" else "training_image"
    expected.update({(kind, f"foodx251:{split}:{i}.png"): i + 1 for i in range(13)})
    requests, _, _ = transport(lambda request: response())
    requests.clear()
    restarted = Experiment(ExperimentConfig.model_validate(config))
    if stage == "validation":
        restarted.evaluate()
        total = 13
    else:
        restarted.build_vector_database()
        total = 16
    submitted = []
    for request in requests:
        part = json.loads(request.content)["content"]["parts"][0]
        if "text" in part:
            submitted.append(part["text"])
        else:
            with Image.open(BytesIO(base64.urlsafe_b64decode(part["inlineData"]["data"]))) as image:
                submitted.append(image.getpixel((0, 0))[0])
    assert Counter(submitted) == Counter(value for key, value in expected.items() if key not in saved)
    index = LanceDBIndex.open(index.uri, index.name)
    assert len(index.get_record_ids()) == total


@pytest.mark.parametrize("value", [False, True])
def test_cache_toggle_boolean(transport, resume_config, value):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.experiments.classification import Experiment

    resume_config["cache_validation_embeddings"] = value
    transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(resume_config))
    experiment.build_vector_database()
    experiment.evaluate()
    uri, name = resume_config["lancedb"]["uri"], resume_config["lancedb"]["validation_table"]
    if value:
        assert len(LanceDBIndex.open(uri, name).get_record_ids()) == 13
    else:
        assert not LanceDBIndex.exists(uri, name)


def test_default_cache_disabled_and_local_cache_supported(transport, resume_config):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.experiments.classification import Experiment

    resume_config.pop("cache_validation_embeddings")
    transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(resume_config))
    experiment.build_vector_database()
    experiment.evaluate()
    assert not LanceDBIndex.exists(resume_config["lancedb"]["uri"],
                                   resume_config["lancedb"]["validation_table"])
    from classification.experiments.classification import Experiment

    resume_config["embedding"]["model_id"] = "openai/clip-vit-base-patch32"
    resume_config["cache_validation_embeddings"] = True
    ExperimentConfig.model_validate(resume_config)
    resume_config["lancedb"]["table"] = "local_fresh"
    local = Experiment(ExperimentConfig.model_validate(resume_config))
    local._embedder = Mock(dimension=128)
    local._embedder.encode_text.side_effect = lambda inputs: np.ones((len(inputs), 128))
    local._embedder.encode_images.side_effect = lambda images: np.ones((len(images), 128))
    append = LanceDBIndex.append
    batch_sizes = []

    def save(index, records, vectors):
        batch_sizes.append(len(records))
        append(index, records, vectors)

    with patch.object(LanceDBIndex, "append", save):
        local.build_vector_database()
        assert local.evaluate().model_dump(mode="json")["metrics"]["evaluated"] == 13
    assert batch_sizes == [3, 13, 13]


def test_cached_report_failure_keeps_vectors(transport, resume_config):
    from classification.experiments.classification import Experiment

    requests, _, _ = transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(resume_config))
    experiment.build_vector_database()
    with (
        patch("classification.experiments.classification.write_json", side_effect=OSError("report")),
        pytest.raises(OSError, match="report"),
    ):
        experiment.evaluate()
    assert not list(experiment.output.glob("evaluation_*/report.json"))
    before = len(requests)
    experiment.evaluate()
    assert len(requests) == before


def test_progress_and_storage_failure_resume(transport, resume_config, progress_bars):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.experiments.classification import Experiment

    transport(lambda request: response())
    tracker = Mock()
    experiment = Experiment(ExperimentConfig.model_validate(resume_config), tracker=tracker)
    original = LanceDBIndex.append
    calls = 0

    def append(index, records, vectors):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk")
        original(index, records, vectors)

    with patch.object(LanceDBIndex, "append", append), pytest.raises(OSError, match="disk"):
        experiment.build_vector_database()
    assert calls == 2
    assert not (experiment.output / "index.json").exists()
    tracker.reset_mock()
    experiment.build_vector_database()
    tracker.progress.assert_not_called()
    assert [(bar.desc, bar.n) for bar in progress_bars[-2:]] == [
        ("Text references", 0), ("Image references", 13),
    ]
    experiment.evaluate()
    progress_bars.clear()
    experiment.evaluate()
    assert [(bar.desc, bar.n) for bar in progress_bars] == [
        ("Validation embeddings", 0), ("Evaluation", 13),
    ]
    tracker.progress.assert_not_called()


def test_cached_predictions_pair_vectors_with_metadata(transport, resume_config, progress_bars):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.schemas import PointRecord
    from classification.experiments.classification import Experiment

    config = resume_config
    root = Path(config["dataset"]["root"])
    (root / "annot/class_list.txt").write_text("0 apple\n1 banana\n")
    (root / "annot/val_info.csv").write_text(
        "".join(f"{i}.png,{i % 2}\n" for i in range(13))
    )
    config["k"] = 1
    reference = LanceDBIndex.create(config["lancedb"]["uri"], config["lancedb"]["table"], 128)
    vectors = np.eye(2, 128, dtype=np.float32)
    reference.append([PointRecord(label, "class_name", label, label)
                      for label in ("Apple", "Banana")], vectors)
    cache = LanceDBIndex.create(config["lancedb"]["uri"], config["lancedb"]["validation_table"], 128)
    # Seed a partial cache in reverse order; new responses also finish out of order.
    for i in reversed(range(0, 13, 2)):
        cache.append([PointRecord("Apple", "validation_image", f"foodx251:val:{i}.png", "old")],
                     vectors[[0]])

    def handle(request):
        part = json.loads(request.content)["content"]["parts"][0]
        with Image.open(BytesIO(base64.urlsafe_b64decode(part["inlineData"]["data"]))) as image:
            i = image.getpixel((0, 0))[0] - 1
        sleep((13 - i) * 0.001)
        return httpx.Response(200, json={"embedding": {"values": vectors[i % 2].tolist()}})

    requests, _, _ = transport(handle)
    config["embedding"]["batch_size"] = 4
    tracker = Mock()
    report = Experiment(ExperimentConfig.model_validate(config), tracker=tracker).evaluate().model_dump(mode="json")
    tracker.progress.assert_not_called()
    assert [(bar.desc, bar.n) for bar in progress_bars] == [
        ("Validation embeddings", 6), ("Evaluation", 13),
    ]
    assert len(requests) == 6
    assert [row["prediction"] for row in report["predictions"]] == [
        "Apple" if i % 2 == 0 else "Banana" for i in range(13)]
    assert report["metrics"]["majority_vote_accuracy"] == 1
    for path in root.glob("val/val_set/*.png"):
        path.unlink()
    with patch("classification.encoders.vertex.genai.Client", side_effect=AssertionError("client")):
        cached = Experiment(ExperimentConfig.model_validate(config)).evaluate().model_dump(mode="json")
        assert cached["predictions"] == report["predictions"]
        assert cached["metrics"] == report["metrics"]
    for key in ("f1_macro", "f1_micro", "top5_hit_rate", "retrieval_precision_at_k",
                "mrr_at_k", "ndcg_at_k", "map_at_r"):
        assert report["metrics"][key] == 1
    tracker.evaluation.assert_called_once()
    assert tracker.evaluation.call_args.args[0].metrics == report["metrics"]


def test_reference_snapshot_counts_all_stored_rows_once(transport, resume_config):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.schemas import PointRecord
    from classification.experiments.classification import Experiment

    transport(lambda request: response())
    tracker = Mock()
    experiment = Experiment(ExperimentConfig.model_validate(resume_config), tracker=tracker)
    index = LanceDBIndex.create(resume_config["lancedb"]["uri"], resume_config["lancedb"]["table"], 128)
    # Counts include duplicate and unrequested stored rows; filtering is a set.
    index.append([PointRecord("Apple", "class_name", identity, identity)
                  for identity in ("Apple", "Apple", "old")], np.ones((3, 128)))
    original = LanceDBIndex.get_record_ids
    scans = []

    def get_record_ids(index):
        scans.append(index.name)
        return original(index)

    with patch.object(LanceDBIndex, "get_record_ids", get_record_ids):
        summary = experiment.build_vector_database().model_dump(mode="json")
    assert scans == [index.name]
    assert summary["source_counts"] == {
        "class_name": 3, "generated_description": 2, "training_image": 13,
    }
    tracker.progress.assert_not_called()


def test_reference_inputs_fail_before_table_creation(transport, resume_config):
    from classification.experiments.classification import Experiment
    from classification.retrieval.lancedb import LanceDBIndex

    transport(lambda request: response())
    Path(resume_config["descriptions_file"]).unlink()
    experiment = Experiment(ExperimentConfig.model_validate(resume_config))
    with pytest.raises(FileNotFoundError):
        experiment.build_vector_database()
    assert not LanceDBIndex.exists(resume_config["lancedb"]["uri"], resume_config["lancedb"]["table"])
    assert experiment._embedder is None


def test_validation_cache_completes_before_classification(transport, resume_config):
    from classification.retrieval.lancedb import LanceDBIndex
    from classification.experiments.classification import Experiment

    transport(lambda request: response())
    experiment = Experiment(ExperimentConfig.model_validate(resume_config))
    experiment.build_vector_database()
    original = LanceDBIndex.query

    def query(index, vector, k):
        cache = LanceDBIndex.open(index.uri, resume_config["lancedb"]["validation_table"])
        assert len(cache.get_record_ids()) == 13
        return original(index, vector, k)

    with patch.object(LanceDBIndex, "query", query):
        experiment.evaluate()


@pytest.mark.parametrize("stage,method", [
    ("describe", "generate_descriptions"), ("index", "build_vector_database"),
    ("evaluate", "evaluate"), ("run", "run"),
])
@pytest.mark.parametrize("model,selected", [
    ("google/ge-v2", "Experiment"), ("openai/clip-vit-base-patch32", "Experiment"),
])
def test_cli_selects_class_before_tracking(resume_config, stage, method, model, selected):
    from classification.cli import _run_experiment

    resume_config["embedding"]["model_id"] = model
    events = []
    with (
        patch("classification.cli.Experiment") as chosen,
        patch("classification.cli.wandb.init") as init,
        patch("classification.cli.Tracker"),
    ):
        init.side_effect = lambda **kwargs: events.append("init") or MockContext()
        chosen.side_effect = lambda *args, **kwargs: events.append("construct") or chosen.return_value
        _run_experiment(ExperimentConfig.model_validate(resume_config), stage)
        assert events == ["init", "construct"]
        assert isinstance(chosen.call_args.args[0], ExperimentConfig)
        getattr(chosen.return_value, method).assert_called_once_with()


class MockContext:
    def __enter__(self):
        return Mock()

    def __exit__(self, *args):
        return False


def test_run_dispatches_shared_stages():
    from classification.experiments.classification import Experiment

    experiment = object.__new__(Experiment)
    calls = Mock()
    with (
        patch.object(Experiment, "generate_descriptions", calls.describe),
        patch.object(Experiment, "build_vector_database", calls.build),
        patch.object(Experiment, "evaluate", calls.evaluate),
    ):
        assert experiment.run() is calls.evaluate.return_value
    assert [entry[0] for entry in calls.mock_calls] == ["describe", "build", "evaluate"]


def test_experiment_import_does_not_construct_backends():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
from unittest.mock import patch
import google.genai
import lancedb
import ollama
import sentence_transformers

with (
    patch("google.genai.Client") as vertex,
    patch("lancedb.db.LanceDBConnection") as storage,
    patch("ollama.Client") as ollama_client,
    patch("sentence_transformers.SentenceTransformer") as local,
):
    import classification.experiments.classification
    for constructor in (vertex, storage, ollama_client, local):
        constructor.assert_not_called()
"""], check=True, timeout=60)
