"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from .embedder import Embedder
from .embeddings import EmbeddingModel
from .generate_descriptors import DescriberLLM
from .image_dataset import load_dataset
from .index_builder import IndexBuilder
from .retrieval import LanceDBIndex, classify, metrics, ranking_metrics
from .schemas import ImageMetadata, Neighbor, Prediction
from .tracking import Tracker

type EmbeddingBatch = tuple[dict[str, list], np.ndarray[tuple[int, int], np.dtype[np.float32]]]


def _write_json(path: Path, value: Any) -> None:
    """Safely clean and write JSON config or results file."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# TODO: Review overly defensive config checks
def validate_config(config: dict) -> dict:
    """Validate experiment config does not force exp pipeline into invalid state.

    - Require that at least one modality (class labels, images, descriptions)
        is used in training dataset
    - Ensure that results output directory is not nested inside data directory.
    """
    if type(config["k"]) is not int or config["k"] <= 0:
        raise ValueError("k must be a positive integer")
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
    if config.get("cache_validation_embeddings", False):
        name = config["lancedb"].get("validation_table")
        if not isinstance(name, str) or not name.strip() or name == config["lancedb"]["table"]:
            raise ValueError("validation_table must be nonempty and distinct from reference table")
    return config


def _extract_batch_metadata(batch: dict[str, list]) -> list[ImageMetadata]:
    """Convert metadata columns into ordered report rows, omitting images."""
    metadata = batch.copy()
    metadata.pop("image", None)
    return [
        cast(ImageMetadata, dict(zip(metadata, values, strict=True)))
        for values in zip(*metadata.values(), strict=True)
    ]


def _prediction(
    record: ImageMetadata,
    ranked: Sequence[Neighbor],
    k: int,
    relevant_count: int,
) -> Prediction:
    """Combine image metadata, the vote result, and retrieved neighbors."""
    neighbors = ranked[:k]
    label, fraction = classify(neighbors)
    return {
        # append image metadata into the result.
        **record,
        "prediction": label,
        "vote_fraction": fraction,
        "nearest": neighbors[0]["label"] if neighbors else None,
        "neighbors": list(neighbors),
        "retrieval_metrics": ranking_metrics(ranked, record["class"], k, relevant_count),
    }


def cached_batches(
    cache: LanceDBIndex,
    metadata_batches: Iterable[dict[str, list]],
) -> Iterator[EmbeddingBatch]:
    """Pair metadata with cached vectors in source identity order."""
    for batch in metadata_batches:
        yield (
            batch,
            cache.fetch_vectors(
                [("validation_image", identity) for identity in batch["source_id"]]
            ),
        )


class Experiment:
    """Execute experiment for open-vocabulary constrastive food image classification.

    Uses Ollama (default: gemma3:27b) to generate description sentences for each class,
    generating an embedding vector for training class labels, images, and descriptor captions
    with CLIP. Populates a LanceDB vector database with these embedding vectors. Uses cosine
    similarity search to classify samples at inference time.
    """

    @staticmethod
    def validate_config(config: dict) -> dict:
        return validate_config(config)

    def __init__(self, config: dict, *, tracker: Tracker | None = None):
        self.tracker = tracker
        # Load dataset metadata and settings
        self.config = self.validate_config(config)
        self.output = Path(config["output_dir"]).expanduser().resolve()
        descriptions_file = config.get("descriptions_file")
        self.descriptions_path = (
            Path(descriptions_file).expanduser().resolve()
            if descriptions_file is not None
            else self.output / "descriptions.json"
        )
        self.dataset = load_dataset(config["dataset"])
        self.classes = self.dataset.get_class_list()
        self.batch_size = config["embedding"].get("batch_size", 32)

        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        """Lazy-loading for embedding model."""
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    def _create_embedder(self) -> Embedder:
        options = self.config["embedding"]
        if options.get("model_id") == "google/ge-v2":
            from .vertex_embeddings import VertexEmbeddingModel

            return VertexEmbeddingModel(
                truncate_dim=options.get("truncate_dim"),
                **options.get("vertex", {}),
            )
        return EmbeddingModel(
            **{
                key: options[key]
                for key in ("model_id", "device", "truncate_dim")
                if key in options
            },
            batch_size=self.batch_size,
        )

    def _lancedb_adapter(self) -> LanceDBIndex:
        options = self.config["lancedb"]
        return LanceDBIndex(uri=options["uri"], table=options["table"])

    def _evaluation_batches(
        self,
    ) -> Iterator[EmbeddingBatch]:
        """Encode uncached validation images in bounded batches."""
        for batch in self.dataset.iter_batches("val", self.batch_size):
            vectors = self.embedder.encode_images(batch.pop("image"))
            yield batch, vectors

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
        self.output.mkdir(parents=True, exist_ok=True)
        with DescriberLLM(self.config["ollama"]) as llm:
            descriptions = llm.generate_descriptions(
                self.classes,
                self.config["description_count"],
            )
        _write_json(path, dict(descriptions=descriptions))
        return descriptions

    def _index_builder(self) -> IndexBuilder:
        options = self.config["lancedb"]
        return IndexBuilder(
            get_embedder=lambda: self.embedder, dataset=self.dataset,
            uri=options["uri"], table=options["table"], config=self.config,
            descriptions_path=self.descriptions_path, classes=self.classes,
            batch_size=self.batch_size,
        )

    def build_vector_database(self) -> dict:
        index, counts = self._index_builder().build_reference_index()
        return self._publish_index(index, counts)

    def _publish_index(self, index: LanceDBIndex, counts: Counter[str]) -> dict:
        """Write training summary to log file, update wandb tracking"""
        artifact = dict(
            table=index.name,
            uri=str(index.uri),
            source_counts=counts,
            config=self.config,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        _write_json(self.output / "index.json", artifact)
        if self.tracker:
            self.tracker.reference_counts(dict(counts))
        return artifact

    def evaluate(self) -> dict:
        """Complete optional validation caching, then classify the full split."""
        index, artifact = self._prepare_evaluation()
        if self.config.get("cache_validation_embeddings", False):
            cache = self._index_builder().build_validation_table(
                self.config["lancedb"]["validation_table"]
            )
            batches = cached_batches(
                cache,
                self.dataset.iter_batches("val", self.batch_size, decode=False),
            )
        else:
            batches = self._evaluation_batches()
        return self._execute_evaluation(index, artifact, batches)

    def evaluate_from_batches(
        self,
        batches: Iterable[EmbeddingBatch],
    ) -> dict:
        """Evaluate supplied, ordered metadata/vector batches through standard reporting."""
        index, artifact = self._prepare_evaluation()
        return self._execute_evaluation(index, artifact, batches)

    def _prepare_evaluation(self) -> tuple[LanceDBIndex, dict]:
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
        return index, artifact

    def _execute_evaluation(
        self,
        index: LanceDBIndex,
        artifact: dict,
        batches: Iterable[EmbeddingBatch],
    ) -> dict:
        k = self.config["k"]
        if type(k) is not int or k <= 0:
            raise ValueError("k must be a positive integer")
        counts = index.class_counts()
        total = len(self.dataset.test_set())
        predictions: list[Prediction] = []
        with tqdm(total=total, desc="Evaluation", unit="image", mininterval=5) as progress:
            for batch, vectors in batches:
                rows = _extract_batch_metadata(batch)
                for record, vector in zip(rows, vectors, strict=True):
                    relevant_count = counts[record["class"]]
                    ranked = index.query(vector, max(k, 5, relevant_count))
                    predictions.append(_prediction(record, ranked, k, relevant_count))
                progress.update(len(rows))
        selected = len(self.dataset.test_set())
        if not selected:
            raise ValueError("No evaluation images; accuracy is undefined")
        if selected != len(predictions):
            raise ValueError("Incomplete evaluation: validation and prediction counts differ")
        report = dict(
            config=self.config,
            table=index.name,
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
