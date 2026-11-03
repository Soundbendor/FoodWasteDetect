"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

from .embedder import Embedder
from .embeddings import EmbeddingModel
from .generate_descriptors import DescriberLLM
from .image_dataset import load_dataset
from .retrieval import LanceDBIndex, classify, metrics, ranking_metrics
from .schemas import ImageMetadata, Neighbor, PointRecord, Prediction
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

    def _save_text_batch(self, index: LanceDBIndex, records: list[PointRecord]) -> None:
        index.append(records, self.embedder.encode_text([record.value for record in records]))

    def _save_image_batch(
        self,
        index: LanceDBIndex,
        records: list[PointRecord],
        images: list[Image.Image],
    ) -> None:
        index.append(records, self.embedder.encode_images(images))

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
        self,
        index: LanceDBIndex,
        records: list[PointRecord],
    ) -> Counter[str]:
        """Embed and save text references, reporting successfully inserted records."""
        counts: Counter[str] = Counter()
        with tqdm(
            total=len(records), desc="Text references", unit="record", mininterval=5
        ) as progress:
            for start in range(0, len(records), self.batch_size):
                batch = records[start : start + self.batch_size]
                self._save_text_batch(index, batch)
                counts.update(r.source_kind for r in batch)
                progress.update(len(batch))
        return counts

    def _insert_image_records(
        self,
        index: LanceDBIndex,
        dataset: Dataset,
        *,
        split: Literal["train", "val"],
    ) -> Counter[str]:
        """Decode and persist the prepared image dataset in batches."""
        kind = "training_image" if split == "train" else "validation_image"
        counts: Counter[str] = Counter()
        description = "Image references" if split == "train" else "Validation embeddings"
        with tqdm(desc=description, unit="image", mininterval=5) as progress:
            for batch in self.dataset.iter_dataset_batches(dataset, self.batch_size):
                records = [
                    PointRecord(label, kind, source_id, fname)
                    for label, source_id, fname in zip(
                        batch["class"], batch["source_id"], batch["fname"], strict=True
                    )
                ]
                self._save_image_batch(index, records, batch["image"])
                counts.update(record.source_kind for record in records)
                progress.update(len(records))
        return counts

    def _prepare_reference_inputs(self) -> tuple[Dataset | None, list[PointRecord]]:
        images = None
        if self.config["embed_images"]:
            images = self.dataset.select_images(
                self.dataset.train_set(),
                n_imgs_per_class=self.config.get("n_imgs_per_class"),
            )
            missing = set(self.classes) - set(images["class"])
            if missing:
                raise ValueError(f"Missing usable training references for {missing}")
        return images, self._get_text_inputs()

    def build_vector_database(self) -> dict:
        """Generate embedding vectors from training data, storing in LanceDB table.

        From training dataset (including class labels, description captions, and images), run
        embedding model to generate vectors and store embeddings in LanceDB."""
        index = self._lancedb_adapter()
        if index.exists():
            logging.info("Loading existing table...")
            index.open()
            # Load a list of files which are already stored in the vector db
            vector_ids = index.get_record_ids()
        else:
            index.create(self.embedder.dimension)
            vector_ids = []
        images, records = self._prepare_reference_inputs()
        # convert to set for efficient filtering
        existing_ids = frozenset(vector_ids)
        new_record_refs = [
            r for r in records if (r.source_kind, r.source_identity) not in existing_ids
        ]
        if images is not None:
            excluded = {identity for kind, identity in existing_ids if kind == "training_image"}
            images = self.dataset.select_images(images, excluded_source_ids=excluded)
        counts = Counter(kind for kind, _ in vector_ids)
        counts.update(self._insert_references(index, new_record_refs, images))
        if not counts:
            raise ValueError("No records inserted into LanceDB - check dataset and config.")
        logging.info("Compressing LanceDB table...")
        index.optimize()
        return self._publish_index(index, counts)

    def _insert_references(
        self,
        index: LanceDBIndex,
        records: list[PointRecord],
        images: Dataset | None,
    ) -> Counter[str]:
        counts = self._insert_text_records(index, records)
        if images is not None:
            counts.update(self._insert_image_records(index, images, split="train"))
        return counts

    def _publish_index(self, index: LanceDBIndex, counts: Counter[str]) -> dict:
        if not counts:
            raise ValueError("No records inserted into LanceDB - check dataset and config.")
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

    def _prepare_validation_cache(self) -> LanceDBIndex:
        options = self.config["lancedb"]
        cache = LanceDBIndex(uri=options["uri"], table=options["validation_table"])
        if cache.exists():
            cache.open()
        else:
            cache.create(self.embedder.dimension)
        excluded = {
            identity for kind, identity in cache.get_record_ids() if kind == "validation_image"
        }
        images = self.dataset.select_images(self.dataset.test_set(), excluded_source_ids=excluded)
        self._insert_image_records(cache, images, split="val")
        return cache

    def evaluate(self) -> dict:
        """Complete optional validation caching, then classify the full split."""
        index, artifact = self._prepare_evaluation()
        if self.config.get("cache_validation_embeddings", False):
            cache = self._prepare_validation_cache()
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
