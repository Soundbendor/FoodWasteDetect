"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, ExitStack
import json
import logging
from pathlib import Path
from typing import Any
import uuid

from PIL.Image import Image
from qdrant_client import QdrantClient

from .embeddings import EmbeddingModel
from .generate_descriptors import DescriberLLM
from .image_dataset import FoodX251ImageDataset, load_dataset
from .records import ReferenceRecord
from .retrieval import Neighbor, QdrantIndex, classify, metrics

logger = logging.getLogger(__name__)


def _write_json(path: Path, value: Any) -> None:
    """Write UTF-8 JSON through a sibling temporary file, then replace the target.

    The parent directory must exist. Serialization and filesystem errors
    propagate to the caller; NaN and infinity are rejected.
    """
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_config(config: dict) -> dict:
    """Require boolean reference switches with at least one source enabled.

    Return the original configuration unchanged. Numeric settings are checked
    by the stages that use them, and backend settings by their constructors.
    """
    sources = ("embed_images", "embed_class_labels", "embed_descriptions")
    for key in sources:
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not any(config[key] for key in sources):
        raise ValueError("Select at least one reference source")
    return config


def _positive_integer(value: Any, name: str) -> int:
    """Return a positive integer or raise ValueError naming the setting.

    Booleans are rejected even though Python treats them as integers.
    """
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _batch_rows(batch: dict[str, list]) -> list[dict]:
    """Convert batch metadata columns into ordered rows, omitting image objects.

    Require every metadata column to have one entry per decoded image so labels,
    source identities, and filenames stay aligned during embedding and retrieval.
    """
    metadata = {key: values for key, values in batch.items() if key != "image"}
    if any(len(values) != len(batch["image"]) for values in metadata.values()):
        raise ValueError("Batch metadata/image count mismatch")
    return [
        dict(zip(metadata, values, strict=True))
        for values in zip(*metadata.values(), strict=True)
    ]


@contextmanager
def _closing_images(images: Sequence[Image]) -> Iterator[None]:
    """Close all supplied batch images when the context exits, including on errors."""
    with ExitStack() as stack:
        for image in images:
            stack.callback(image.close)
        yield


def _prediction(record: dict, neighbors: Sequence[Neighbor]) -> dict:
    """Combine image metadata with the vote result and retrieved neighbors.

    Each neighbor contributes one vote; ties use best similarity, then class
    name. Empty retrieval abstains. The vote fraction uses the actual neighbor
    count and is not a calibrated probability.
    """
    label, fraction = classify(neighbors)
    return dict(
        record,
        value=record["fname"],
        prediction=label,
        vote_fraction=fraction,
        nearest=neighbors[0].label if neighbors else None,
        neighbors=[vars(neighbor) for neighbor in neighbors],
    )


def _write_evaluation(
    run_dir: Path, report: dict, counts: dict[str, int], classes: list[str]
) -> None:
    """Add metrics to the report and write report, predictions, and metrics JSON.

    Coverage uses the evaluation manifest size; accuracy uses recorded
    predictions. This also writes partial results after processing failures.
    Files are replaced individually, and write errors propagate to the caller.
    """
    report["metrics"] = metrics(
        report["predictions"],
        len(report.get("evaluation_manifest", [])),
        counts["available"],
        counts["embedded"],
        classes,
    )
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "predictions.json", report["predictions"])
    _write_json(run_dir / "metrics.json", report["metrics"])


class Experiment:
    """Execute experiment for open-vocabulary constrastive food image classification.

    Uses Ollama (default: gemma3:27b) to generate description sentences for each class,
    generating an embedding vector for training class labels, images, and descriptor captions
    with CLIP. Populates a Qdrant vector database with these embedding vectors. Uses cosine
    similarity search to classify samples at inference time.
    """

    def __init__(
        self,
        config: dict,
        *,
        dataset: FoodX251ImageDataset | None = None,
        embedder_factory: Callable[..., EmbeddingModel] | None = None,
        client: QdrantClient | None = None,
        describer_factory: Callable[[dict], DescriberLLM] | None = None,
    ):
        """Store configuration and optional dependencies without starting a stage.

        Validate reference switches and reject output paths inside the configured
        raw-data directory. Dataset loading, directory creation, and backend
        construction are deferred until needed. Factories allow test doubles.
        """
        self.config = validate_config(config)
        self._dataset = dataset
        self._classes = None
        self.output = Path(self.config["output_dir"]).expanduser().resolve()
        # Never write derived artifacts into the raw dataset tree.
        raw = self.config.get("dataset", {}).get("root")
        if raw and self.output.is_relative_to(Path(raw).expanduser().resolve()):
            raise ValueError("Output directory must be separate from raw data")
        self.embedder_factory = embedder_factory
        self.describer_factory = describer_factory
        self.client = client
        self._embedder = None

    @property
    def dataset(self) -> FoodX251ImageDataset:
        """Return the supplied dataset or load and cache its local split metadata."""
        if self._dataset is None:
            self._dataset = load_dataset(self.config["dataset"])
        return self._dataset

    @property
    def classes(self) -> list[str]:
        """Return and cache the dataset's canonical class names in dataset order."""
        if self._classes is None:
            self._classes = self.dataset.get_class_list()
        return self._classes

    @property
    def batch_size(self) -> int:
        """Return the configured positive embedding batch size, defaulting to 32."""
        return _positive_integer(
            self.config["embedding"].get("batch_size", 32), "batch_size"
        )

    @property
    def embedder(self) -> EmbeddingModel:
        """Construct the configured embedder once and reuse it across stages.

        The default Jina CLIP v2 adapter loads cached weights and produces full,
        normalized embeddings. An injected factory receives the same settings.
        """
        if self._embedder is None:
            factory = self.embedder_factory
            if factory is None:
                factory = EmbeddingModel
            self._embedder = factory(
                **dict(self.config["embedding"], batch_size=self.batch_size)
            )
        return self._embedder

    def _index_adapter(self, collection: str) -> QdrantIndex:
        """Wrap a named collection using the shared client and embedder dimension.

        Construct the Qdrant client on first use if none was supplied. This
        creates an adapter only; collection creation is a separate operation.
        """
        if self.client is None:
            self.client = QdrantClient(**self.config["qdrant"]["connection"])
        return QdrantIndex(self.client, collection, self.embedder.dimension)

    def describe(self) -> dict[str, list[str]] | None:
        """Return saved class descriptions or generate and save new descriptions.

        Return None when descriptions are disabled. Existing descriptions load
        directly without dataset loading or compatibility checks. Otherwise,
        generate the configured number per class in an owned Ollama context,
        reject blank descriptions, and save only after successful generation.
        """
        if not self.config["embed_descriptions"]:
            return None
        path = self.output / "descriptions.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["descriptions"]
        count = _positive_integer(self.config["description_count"], "description_count")
        classes = self.classes
        self.output.mkdir(parents=True, exist_ok=True)
        factory = self.describer_factory
        if factory is None:
            factory = DescriberLLM
        descriptions = {}
        with factory(self.config["ollama"]) as llm:
            for label in classes:
                descriptions[label] = [
                    llm.describe_class(label).description for _ in range(count)
                ]
                if any(not text.strip() for text in descriptions[label]):
                    raise ValueError(f"Empty generated description for {label}")
        _write_json(path, dict(descriptions=descriptions))
        return descriptions

    def _text_references(self) -> list[ReferenceRecord]:
        """Build enabled class-name and saved-description reference records.

        Preserve class order and description order, with a new point UUID per
        record. Description identities combine the class name and list position.
        Enabled descriptions must already exist; this method does not generate them.
        """
        records = []
        if self.config["embed_class_labels"]:
            records += [ReferenceRecord(c, "class_name", c, c) for c in self.classes]
        if self.config["embed_descriptions"]:
            path = self.output / "descriptions.json"
            if not path.exists():
                raise ValueError("Run describe before index")
            data = json.loads(path.read_text(encoding="utf-8"))["descriptions"]
            records += [
                ReferenceRecord(c, "generated_description", f"{c}:{i}", text)
                for c in self.classes
                for i, text in enumerate(data[c])
            ]
        return records

    def _index_text(
        self,
        index: QdrantIndex,
        records: list[ReferenceRecord],
        counts: Counter,
        batch_size: int,
    ) -> None:
        """Embed and upsert text references in batches, including the final partial batch.

        Update source counts only after each successful upsert. Embedding and
        database errors propagate without retries or skipped references.
        """
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            index.upsert(batch, self.embedder.encode_text([r.value for r in batch]))
            counts.update(r.source_kind for r in batch)

    def _index_images(
        self, index: QdrantIndex, counts: Counter, batch_size: int
    ) -> None:
        """Embed all training images and store labeled records from each decoded batch.

        Retain source IDs and filenames, count successful upserts, and close
        images after each batch. Fail on processing errors or if any class lacks
        an indexed image reference. Raw images and annotations remain unchanged.
        """
        image_classes = set()
        for batch in self.dataset.iter_batches("train", batch_size):
            with _closing_images(batch["image"]):
                records = [
                    ReferenceRecord(
                        row["class"], "training_image", row["source_id"], row["fname"]
                    )
                    for row in _batch_rows(batch)
                ]
                index.upsert(records, self.embedder.encode_images(batch["image"]))
                counts.update(r.source_kind for r in records)
                image_classes.update(r.label for r in records)
        if image_classes != set(self.classes):
            raise ValueError(
                f"Missing usable training references for {set(self.classes) - image_classes}"
            )

    def index(self) -> dict:
        """Build a fresh reference collection and return its saved summary.

        Create a uniquely named cosine Qdrant collection and insert embeddings
        for the enabled text and training-image sources. Vectors and reference
        payloads live in Qdrant; index.json records the collection name, model
        settings, source counts, configuration, and completion state. Each call
        replaces that summary while retaining previous collections.

        An empty index fails. Processing failures attempt to save an incomplete
        summary before re-raising the original error, even if reporting also fails.
        """
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "index.json"
        collection = self.config["qdrant"]["collection_prefix"] + "_" + uuid.uuid4().hex
        artifact = dict(
            state="incomplete",
            collection=collection,
            source_counts=Counter(),
            config=self.config,
        )
        try:
            _write_json(path, artifact)
            batch_size = self.batch_size
            records = self._text_references()
            artifact["model"] = self.embedder.metadata
            index = self._index_adapter(collection)
            index.create()
            self._index_text(index, records, artifact["source_counts"], batch_size)
            if self.config["embed_images"]:
                self._index_images(index, artifact["source_counts"], batch_size)
            if not artifact["source_counts"]:
                raise ValueError("Empty reference index")
            artifact["state"] = "complete"
        except BaseException as exc:
            artifact["error"] = str(exc)
            try:
                _write_json(path, artifact)
            except Exception:
                logger.exception("Could not write incomplete index report")
            raise
        _write_json(path, artifact)
        return artifact

    def evaluate(self) -> dict:
        """Classify the full labeled validation split and return its saved report.

        Read the collection from index.json directly, recompute image embeddings,
        and classify exact cosine top-k neighbors by majority vote. The dataset's
        test_set() supplies validation metadata, not the unlabeled official test
        split. Empty retrieval abstains and counts as incorrect.

        Write predictions, metrics, and a report in a new evaluation directory.
        Completion requires a nonempty split and matching selected, available,
        embedded, and predicted counts. Processing failures close batch images,
        attempt to save incomplete results, and propagate without retries or exclusions.
        """
        run_dir = self.output / ("evaluation_" + uuid.uuid4().hex)
        run_dir.mkdir(parents=True)
        report = dict(
            state="incomplete",
            config=self.config,
            retrieval={"k": self.config["k"], "exact": True, "distance": "Cosine"},
            predictions=[],
        )
        counts = {"available": 0, "embedded": 0}
        classes = []
        try:
            batch_size = self.batch_size
            k = _positive_integer(self.config["k"], "k")
            classes = self.classes
            report["evaluation_manifest"] = [
                dict(row, value=row["fname"])
                for row in self.dataset.test_set().remove_columns("image")
            ]
            artifact = json.loads(
                (self.output / "index.json").read_text(encoding="utf-8")
            )
            index = self._index_adapter(artifact["collection"])
            report["collection"] = artifact["collection"]
            report["model"] = artifact["model"]
            report["reference_source_counts"] = artifact["source_counts"]
            for batch in self.dataset.iter_batches("val", batch_size):
                with _closing_images(batch["image"]):
                    counts["available"] += len(batch["image"])
                    rows = _batch_rows(batch)
                    vectors = self.embedder.encode_images(batch["image"])
                    counts["embedded"] += len(vectors)
                    if len(vectors) != len(rows):
                        raise ValueError("Record/vector count mismatch")
                    for record, vector in zip(rows, vectors, strict=True):
                        report["predictions"].append(
                            _prediction(record, index.query(vector, k))
                        )
            selected = len(report["evaluation_manifest"])
            if not selected or not counts["embedded"]:
                raise ValueError(
                    "No evaluation images could be embedded; accuracy is undefined"
                )
            if (
                not selected
                == counts["available"]
                == counts["embedded"]
                == len(report["predictions"])
            ):
                raise ValueError(
                    "Incomplete evaluation: selected, available, embedded and prediction counts differ"
                )
            report["state"] = "complete"
        except BaseException as exc:
            report["error"] = str(exc)
            try:
                _write_evaluation(run_dir, report, counts, classes)
            except Exception:
                logger.exception("Could not write incomplete evaluation report")
            raise
        _write_evaluation(run_dir, report, counts, classes)
        return report

    def run(self) -> dict:
        """Describe, build a fresh index, and evaluate, returning the evaluation report.

        Description generation closes its owned Ollama server before subsequent
        stages construct the embedder. Any stage failure stops the sequence.
        """
        self.describe()  # Owned Ollama context exits before embedder construction.
        self.index()
        return self.evaluate()


def main(argv: list[str] | None = None) -> None:
    """Load the required YAML configuration and dispatch the selected stage.

    Accept describe, index, evaluate, or run with --config_file. When argv is
    None, argparse reads the process arguments. Configuration paths resolve
    relative to the current working directory, and stage errors propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("describe", "index", "evaluate", "run"))
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args(argv)
    import yaml

    with open(args.config_file, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    getattr(Experiment(config), args.stage)()


if __name__ == "__main__":
    main()
