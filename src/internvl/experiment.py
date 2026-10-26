"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .embeddings import EmbeddingModel
from .generate_descriptors import DescriberLLM
from .image_dataset import load_dataset
from .retrieval import LanceDBIndex, classify, metrics
from .schemas import ImageMetadata, Neighbor, PointRecord, Prediction, SourceIdentity
from .tracking import Tracker
from .vertex_embeddings import VertexEmbeddingModel


def _write_json(path: Path, value: Any) -> None:
    """Safely clean and write JSON config or results file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_config(config: dict) -> dict:
    """Validate experiment config does not force exp pipeline into invalid state.

    - Require that at least one modality (class labels, images, descriptions)
        is used in training dataset
    - Ensure that results output directory is not nested inside data directory.
    """
    sources = ("embed_images", "embed_class_labels", "embed_descriptions")
    for key in sources:
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not any(config[key] for key in sources):
        raise ValueError("Select at least one reference source")
    limit = config.get("n_imgs_per_class")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("n_imgs_per_class must be null or a positive integer")
    if config.get("embedding", {}).get("model_id") == "google/ge-v2":
        cache = config.get("cache_validation_embeddings", False)
        if type(cache) is not bool:
            raise ValueError("cache_validation_embeddings must be a boolean")
        if cache:
            name = config["lancedb"].get("validation_table")
            if not isinstance(name, str) or not name.strip() or name == config["lancedb"]["table"]:
                raise ValueError("validation_table must be nonempty and distinct from reference table")
    # Reject output paths equal to or nested inside the resolved raw dataset directory.
    raw = config.get("dataset", {}).get("root")
    if raw:
        output = Path(config["output_dir"]).expanduser().resolve()
        if output.is_relative_to(Path(raw).expanduser().resolve()):
            raise ValueError("Output directory must be separate from raw data")
    return config


def _batch_rows(batch: dict[str, list]) -> list[ImageMetadata]:
    """Convert metadata columns into ordered rows, omitting decoded images."""
    metadata = batch.copy()
    metadata.pop("image", None)
    return [
        cast(ImageMetadata, dict(zip(metadata, values, strict=True)))
        for values in zip(*metadata.values(), strict=True)
    ]


def _prediction(record: ImageMetadata, neighbors: Sequence[Neighbor]) -> Prediction:
    """Combine image metadata, the vote result, and retrieved neighbors."""
    label, fraction = classify(neighbors)
    return {
        # append image metadata into the result.
        **record,
        "prediction": label,
        "vote_fraction": fraction,
        "nearest": neighbors[0]["label"] if neighbors else None,
        "neighbors": list(neighbors),
    }


class Experiment:
    """Execute experiment for open-vocabulary constrastive food image classification.

    Uses Ollama (default: gemma3:27b) to generate description sentences for each class,
    generating an embedding vector for training class labels, images, and descriptor captions
    with CLIP. Populates a LanceDB vector database with these embedding vectors. Uses cosine
    similarity search to classify samples at inference time.
    """

    def __init__(self, config: dict, *, tracker: Tracker | None = None):
        self.tracker = tracker
        # Load dataset metadata and settings
        self.config = validate_config(config)
        self.output = Path(config["output_dir"]).expanduser().resolve()
        descriptions_file = config.get("descriptions_file")
        self.descriptions_path = (
            Path(descriptions_file).expanduser().resolve()
            if descriptions_file is not None else self.output / "descriptions.json"
        )
        self.dataset = load_dataset(config["dataset"])
        self.classes = self.dataset.get_class_list()
        self.batch_size = config["embedding"].get("batch_size", 32)

        self._embedder = None

    @property
    def embedder(self) -> EmbeddingModel | VertexEmbeddingModel:
        """Construct the embedder on first use and reuse it across stages."""
        if self._embedder is None:
            options = dict(self.config["embedding"], batch_size=self.batch_size)
            vertex = options.pop("vertex", {})
            if options.get("model_id") == "google/ge-v2":
                self._embedder = VertexEmbeddingModel(
                    truncate_dim=options.get("truncate_dim"), **vertex
                )
            else:
                self._embedder = EmbeddingModel(**options)
        return self._embedder

    def _lancedb_adapter(self) -> LanceDBIndex:
        options = self.config["lancedb"]
        return LanceDBIndex(uri=options["uri"], table=options["table"])

    @property
    def _is_vertex(self) -> bool:
        return self.config["embedding"].get("model_id") == "google/ge-v2"

    @staticmethod
    def _embedding_callback(
        index: LanceDBIndex, records: Sequence[PointRecord],
    ) -> Callable[[int, NDArray[np.float32]], None]:
        """Persist each completed response against its original input position."""
        def save(position: int, vector: NDArray[np.float32]) -> None:
            index.append([records[position]], vector[None, :])

        return save

    def _save_text_batch(self, index: LanceDBIndex, records: Sequence[PointRecord]) -> None:
        inputs = [record.value for record in records]
        encoder = self.embedder
        if isinstance(encoder, VertexEmbeddingModel):
            encoder.encode_text(inputs, on_embedding=self._embedding_callback(index, records))
        else:
            index.append(records, encoder.encode_text(inputs))

    def _save_image_batch(
        self, index: LanceDBIndex, records: Sequence[PointRecord], images: list[Image.Image],
    ) -> None:
        encoder = self.embedder
        if isinstance(encoder, VertexEmbeddingModel):
            encoder.encode_images(images, on_embedding=self._embedding_callback(index, records))
        else:
            index.append(records, encoder.encode_images(images))

    def _progress_callback(
        self, stage: str, total: int, reused: int,
    ) -> Callable[[int], None]:
        completed = reused
        if self.tracker:
            self.tracker.progress(stage, 0, total)
            if reused:
                self.tracker.progress(stage, reused, total)

        def advance(count: int) -> None:
            nonlocal completed
            completed += count
            if self.tracker:
                self.tracker.progress(stage, completed, total)

        return advance

    def _prepare_validation_cache(self) -> LanceDBIndex:
        """Complete missing validation vectors before classification begins."""
        options = self.config["lancedb"]
        cache = LanceDBIndex(uri=options["uri"], table=options["validation_table"])
        if cache.exists():
            cache.open()
        else:
            cache.create(cast(int, self.embedder.dimension))
        saved = frozenset(cache.identities())
        requested = self.dataset.test_set()["source_id"]
        reused = sum(("validation_image", identity) in saved for identity in requested)
        advance = self._progress_callback("validation_embedding", len(requested), reused)
        self._insert_image_records(cache, split="val", saved=saved, callback=advance)
        return cache

    def _evaluation_batches(
        self, cache: LanceDBIndex | None,
    ) -> Iterator[tuple[list[ImageMetadata], np.ndarray[tuple[int, int], np.dtype[np.float32]]]]:
        """Read cached vectors in metadata order, or encode decoded image batches."""
        for batch in self.dataset.iter_batches("val", self.batch_size, decode=cache is None):
            rows = _batch_rows(batch)
            vectors = (
                cache.fetch_vectors([("validation_image", row["source_id"]) for row in rows])
                if cache is not None else self.embedder.encode_images(batch["image"])
            )
            yield rows, vectors

    def generate_descriptions(self) -> dict[str, list[str]] | None:
        """Read an explicit vocabulary, or reuse/generate descriptions in the output directory.

        Returns:
            A list of descriptions in {"class_name", ["desc1", "desc2",...]} format.
        """
        if not self.config["embed_descriptions"]:
            logging.info("Description generation is disabled")
            return None
        path = self.descriptions_path
        if self.config.get("descriptions_file") is not None or path.exists():
            logging.info(f"Loading existing descriptions at {path}")
            return json.loads(path.read_text(encoding="utf-8"))["descriptions"]
        on_description = None
        if self.tracker:
            total = len(self.classes) * self.config["description_count"]
            self.tracker.progress("descriptions", 0, total)
            on_description = partial(self.tracker.progress, "descriptions")

        self.output.mkdir(parents=True, exist_ok=True)
        with DescriberLLM(self.config["ollama"]) as llm:
            descriptions = llm.generate_descriptions(
                self.classes,
                self.config["description_count"],
                on_description=on_description,
            )
        _write_json(path, dict(descriptions=descriptions))
        return descriptions

    def _get_text_inputs(self) -> list[PointRecord]:
        """Build enabled text inputs in class order from names and saved descriptions.

        Returns:
            A list of PointRecords which contain class name and class descriptor captions
                for each class in the label space.
        """
        records = []
        # Append class labels to text dataset
        if self.config["embed_class_labels"]:
            records += [PointRecord(c, "class_name", c, c) for c in self.classes]
        # Append generated class descriptor captions.
        if self.config["embed_descriptions"]:
            path = self.descriptions_path
            if self.config.get("descriptions_file") is None and not path.exists():
                raise ValueError("Run describe before index")
            data = json.loads(path.read_text(encoding="utf-8"))["descriptions"]
            records += [
                PointRecord(c, "generated_description", f"{c}:{i}", text)
                for c in self.classes
                for i, text in enumerate(data[c])
            ]
        return records

    def _insert_text_records(
        self, index: LanceDBIndex, records: list[PointRecord], *,
        callback: Callable[[int], None] | None = None
    ) -> Counter[str]:
        """Generate embedding vectors from text inputs and save vectors in LanceDB.

        Args:
            index: A LanceDBIndex object, wrapping the vector database table.
            records: A list of text records to embed and append.
            callback: Optional callback for W&B progress logging, called with the
                batch length after each successful insertion. None disables logging.

        Returns:
            A Counter documenting how many records of each modality were inserted.
        """
        counts: Counter[str] = Counter()
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            self._save_text_batch(index, batch)
            counts.update(r.source_kind for r in batch)
            if callback is not None:
                callback(len(batch))
        return counts

    def _insert_image_records(
        self, index: LanceDBIndex, *, split: Literal["train", "val"],
        saved: frozenset[SourceIdentity] = frozenset(),
        callback: Callable[[int], None] | None = None,
    ) -> Counter[str]:
        """Decode and persist only missing images, applying training caps first."""
        kind = "training_image" if split == "train" else "validation_image"
        counts: Counter[str] = Counter()
        for batch in self.dataset.iter_batches(
            split, self.batch_size,
            n_imgs_per_class=self.config.get("n_imgs_per_class") if split == "train" else None,
            excluded_source_ids={identity for source, identity in saved if source == kind},
        ):
            records = [
                PointRecord(row["class"], kind, row["source_id"], row["fname"])
                for row in _batch_rows(batch)
            ]
            self._save_image_batch(index, records, batch["image"])
            counts.update(record.source_kind for record in records)
            if callback is not None:
                callback(len(records))
        return counts

    def build_vector_database(self) -> dict | None:
        """Build references, resuming committed Vertex rows from one identity snapshot."""
        index = self._lancedb_adapter()
        exists = index.exists()
        counts: Counter[str] = Counter()
        saved: frozenset[SourceIdentity] = frozenset()
        if exists:
            if not self._is_vertex:
                logging.warning(
                    "LanceDB table %r already exists; reusing its precomputed vectors "
                    "without inserting records or checking compatibility.", index.name,
                )
                return None
            index.open()
            identities = index.identities()
            saved = frozenset(identities)
            counts.update(kind for kind, _ in identities)

        records = self._get_text_inputs()
        total = len(records)
        reused = sum((record.source_kind, record.source_identity) in saved for record in records)
        if self.config["embed_images"]:
            represented: set[str] = set()
            for batch in self.dataset.iter_batches(
                "train", self.batch_size, decode=False,
                n_imgs_per_class=self.config.get("n_imgs_per_class"),
            ):
                total += len(batch["source_id"])
                reused += sum(("training_image", identity) in saved for identity in batch["source_id"])
                represented.update(batch["class"])
            missing = set(self.classes) - represented
            if missing:
                raise ValueError(f"Missing usable training references for {missing}")
        records = [r for r in records if (r.source_kind, r.source_identity) not in saved]
        advance = self._progress_callback("reference_embedding", total, reused)
        if not exists:
            index.create(cast(int, self.embedder.dimension))
        counts.update(self._insert_text_records(index, records, callback=advance))
        if self.config["embed_images"]:
            counts.update(self._insert_image_records(
                index, split="train", saved=saved, callback=advance,
            ))
        if not counts:
            raise ValueError("No records inserted into LanceDB - check dataset and config.")
        artifact = dict(
            table=index.name, uri=str(index.uri), source_counts=counts, config=self.config,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        _write_json(self.output / "index.json", artifact)
        if self.tracker:
            self.tracker.reference_counts(dict(counts))
        return artifact

    def evaluate(self) -> dict:
        """Classify the full validation split and save one report after success."""
        if self.tracker:
            total = len(self.dataset.test_set())
            self.tracker.progress("evaluation", 0, total)
        table = self.config["lancedb"]["table"]
        index = self._lancedb_adapter()
        index.open()
        artifact = {}
        path = self.output / "index.json"
        if path.exists():
            logging.info("Loading index.json from checkpoint...")
            summary = json.loads(path.read_text(encoding="utf-8"))
            # Only attach metadata identifying this database and table.
            saved_uri = summary.get("uri")
            if (
                summary.get("table") == table
                and saved_uri is not None
                and Path(saved_uri).expanduser().resolve() == index.uri
            ):
                artifact = summary
        if self.tracker:
            self.tracker.reference_counts(artifact.get("source_counts"))
        # Classify test-time samples based on majority-vote query to vdb
        predictions: list[Prediction] = []
        cache = (
            self._prepare_validation_cache()
            if self._is_vertex and self.config.get("cache_validation_embeddings", False) else None
        )
        for rows, vectors in self._evaluation_batches(cache):
            for record, vector in zip(rows, vectors, strict=True):
                predictions.append(_prediction(record, index.query(vector, self.config["k"])))
            if self.tracker:
                self.tracker.progress("evaluation", len(predictions), total)
        selected = len(self.dataset.test_set())
        if not selected:
            raise ValueError("No evaluation images; accuracy is undefined")
        if selected != len(predictions):
            raise ValueError("Incomplete evaluation: validation and prediction counts differ")
        report = dict(
            config=self.config,
            table=table,
            reference_source_counts=artifact.get("source_counts"),
            predictions=predictions,
            metrics=metrics(predictions, self.classes),
        )
        run_dir = self.output / datetime.now(UTC).strftime("evaluation_%Y%m%dT%H%M%S_%fZ")
        run_dir.mkdir(parents=True)
        report_path = run_dir / "report.json"
        _write_json(report_path, report)
        if self.tracker:
            self.tracker.evaluation(report, report_path)
        return report

    def run(self) -> dict:
        """Describe, index, and evaluate, closing Ollama before loading the embedder."""
        self.generate_descriptions()  # Owned Ollama context exits before embedder construction.
        self.build_vector_database()
        return self.evaluate()
