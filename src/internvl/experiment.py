"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .embeddings import EmbeddingModel
from .generate_descriptors import DescriberLLM
from .image_dataset import load_dataset
from .retrieval import LanceDBIndex, classify, metrics
from .schemas import ImageMetadata, Neighbor, PointRecord, Prediction


def _write_json(path: Path, value: Any) -> None:
    """Write JSON via temporary-file replacement; reject nonfinite numbers."""
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

    def __init__(self, config: dict):
        # Load dataset metadata and settings
        self.config = validate_config(config)
        self.output = Path(config["output_dir"]).expanduser().resolve()
        self.dataset = load_dataset(config["dataset"])
        self.classes = self.dataset.get_class_list()
        self.batch_size = config["embedding"].get("batch_size", 32)

        self._embedder = None

    @property
    def embedder(self):
        """Construct the embedder on first use and reuse it across stages."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(
                **dict(self.config["embedding"], batch_size=self.batch_size)
            )
        return self._embedder

    def _lancedb_adapter(self):
        return LanceDBIndex(**self.config["lancedb"])

    def generate_descriptions(self) -> dict[str, list[str]] | None:
        """Load saved descriptions or generate them in an owned Ollama context.

        Returns:
            A list of descriptions in {"class_name", ["desc1", "desc2",...]} format.
        """
        if not self.config["embed_descriptions"]:
            return None
        path = self.output / "descriptions.json"
        if path.exists():
            logging.info(f"Loading existing descriptions at {path}")
            return json.loads(path.read_text(encoding="utf-8"))["descriptions"]
        self.output.mkdir(parents=True, exist_ok=True)
        with DescriberLLM(self.config["ollama"]) as llm:
            descriptions = llm.generate_descriptions(self.classes, self.config["description_count"])
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
            path = self.output / "descriptions.json"
            if not path.exists():
                raise ValueError("Run describe before index")
            data = json.loads(path.read_text(encoding="utf-8"))["descriptions"]
            records += [
                PointRecord(c, "generated_description", f"{c}:{i}", text)
                for c in self.classes
                for i, text in enumerate(data[c])
            ]
        return records

    def _insert_text_records(self, index, records: list[PointRecord]) -> Counter:
        """Generate embedding vectors from text inputs and save vectors in LanceDB.

        Args:
            index: A LanceDBIndex object, wrapping the vector database table.
            records: A list of text records to embed and append.

        Returns:
            A Counter documenting how many records of each modality were inserted.
        """
        counts = Counter()
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            index.append(batch, self.embedder.encode_text([r.value for r in batch]))
            counts.update(r.source_kind for r in batch)
        return counts

    def _insert_image_records(self, index) -> Counter:
        """Generate embedding vectors from images and save vectors in LanceDB.

        Args:
            index: A LanceDBIndex object, wrapping the vector database table.
        Returns:
            A Counter documenting how many records of each modality were inserted.
        """
        counts = Counter()
        image_classes = set()
        for batch in self.dataset.iter_batches(
            "train", self.batch_size, n_imgs_per_class=self.config.get("n_imgs_per_class")
        ):
            records = [
                PointRecord(row["class"], "training_image", row["source_id"], row["fname"])
                for row in _batch_rows(batch)
            ]
            index.append(records, self.embedder.encode_images(batch["image"]))
            counts.update(r.source_kind for r in records)
            image_classes.update(r.label for r in records)
        if image_classes != set(self.classes):
            raise ValueError(
                f"Missing usable training references for {set(self.classes) - image_classes}"
            )
        return counts

    def build_vector_database(self) -> dict | None:
        """Build the configured table, or warn and return if it already exists."""
        table = self.config["lancedb"]["table"]
        index = self._lancedb_adapter()
        if index.exists():
            logging.warning(
                "LanceDB table %r already exists; reusing its precomputed vectors "
                "without inserting records or checking compatibility.",
                table,
            )
            return None
        self.output.mkdir(parents=True, exist_ok=True)
        records = self._get_text_inputs()
        index.create(self.embedder.dimension)
        counts = self._insert_text_records(index, records)
        if self.config["embed_images"]:
            counts.update(self._insert_image_records(index))
        if not counts:
            raise ValueError("No records inserted into LanceDB - check dataset and config.")
        artifact = dict(
            table=table,
            uri=str(index.uri),
            source_counts=counts,
            config=self.config,
        )
        _write_json(self.output / "index.json", artifact)
        return artifact

    def evaluate(self) -> dict:
        """Classify the full validation split and save one report after success."""
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
        # Classify test-time samples based on majority-vote query to vdb
        predictions: list[Prediction] = []
        for batch in self.dataset.iter_batches("val", self.batch_size):
            rows = _batch_rows(batch)
            vectors = self.embedder.encode_images(batch["image"])
            for record, vector in zip(rows, vectors, strict=True):
                predictions.append(_prediction(record, index.query(vector, self.config["k"])))
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
        _write_json(run_dir / "report.json", report)
        return report

    def run(self) -> dict:
        """Describe, index, and evaluate, closing Ollama before loading the embedder."""
        self.generate_descriptions()  # Owned Ollama context exits before embedder construction.
        self.build_vector_database()
        return self.evaluate()
