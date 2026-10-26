"""Vertex contracts using the real SDK and an in-memory HTTP transport."""

import json
import threading
from io import BytesIO
from pathlib import Path
from time import sleep
from unittest.mock import patch

import httpx
import numpy as np
import pytest
import yaml
from google import genai
from google.auth.credentials import AnonymousCredentials
from google.genai import errors
from PIL import Image

from internvl.vertex_embeddings import VertexEmbeddingModel


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

        monkeypatch.setattr("internvl.vertex_embeddings.genai.Client", create)
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
    with patch("internvl.vertex_embeddings.genai.Client") as client:
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
    from internvl.experiment import Experiment

    experiment = object.__new__(Experiment)
    experiment._embedder = None
    experiment.batch_size = 2
    experiment.config = {
        "embedding": {
            "model_id": "google/ge-v2",
            "device": "cpu",
            "vertex": {"project": "fixture-project"},
        }
    }
    with patch("internvl.experiment.EmbeddingModel") as local:
        assert isinstance(experiment.embedder, VertexEmbeddingModel)
        assert experiment.embedder is experiment.embedder
        local.assert_not_called()
    experiment._embedder = None
    experiment.config["embedding"]["model_id"] = "openai/clip-vit-base-patch32"
    with patch("internvl.experiment.EmbeddingModel") as local:
        assert experiment.embedder is local.return_value
        local.assert_called_once_with(
            model_id="openai/clip-vit-base-patch32", device="cpu", batch_size=2
        )


def test_yaml_cli_overrides_and_defaults():
    from internvl.cli import main

    path = Path("cfg/ge_v2.yml")
    original = path.read_text()
    config = yaml.safe_load(original)
    baseline = yaml.safe_load(Path("cfg/classification.yml").read_text())
    for key in baseline.keys() - {"embedding", "lancedb", "output_dir"}:
        assert config[key] == baseline[key]
    assert config["lancedb"]["table"] == "foodx251_ge_v2"
    assert config["output_dir"] == "assets/classification/ge_v2"
    with (
        patch("internvl.cli.wandb.init"),
        patch("internvl.cli.Tracker"),
        patch("internvl.cli.Experiment") as experiment,
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
            ]
        )
        effective = experiment.call_args.args[0]
        assert effective["embedding"]["vertex"]["project"] == "fixture-project"
        assert effective["embedding"]["vertex"]["max_concurrency"] == 2
        assert effective["embedding"]["vertex"]["classification_prompt"] is True
        assert effective["embedding"]["truncate_dim"] == 128
        assert effective["embed_images"] is True
    assert path.read_text() == original


def test_experiment_references_cache_and_failed_evaluation(transport, tmp_path):
    from internvl.experiment import Experiment

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
    experiment = Experiment(config)
    summary = experiment.build_vector_database()
    assert summary["source_counts"] == {
        "class_name": 1,
        "generated_description": 3,
        "training_image": 3,
    }
    assert len(requests) == 7
    assert all(client.is_closed for client in clients)
    with patch("internvl.vertex_embeddings.genai.Client") as client:
        cached = Experiment(config)
        assert cached.build_vector_database() is None
        assert cached._embedder is None
        client.assert_not_called()
    report = experiment.evaluate()
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

    transport(fail_second_batch)
    with pytest.raises(errors.APIError):
        experiment.evaluate()
    assert failures == 3
    assert set(experiment.output.glob("evaluation_*/report.json")) == previous
    assert all(client.is_closed for client in clients)
