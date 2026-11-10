"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from classification.datasets.image_dataset import FoodX251ImageDataset, load_dataset
from classification.encoders.base import Embedder
from classification.encoders.sentence_transformers import EmbeddingModel
from classification.encoders.vertex import VertexEmbeddingModel
from classification.experiments import evaluation
from classification.experiments.artifacts import write_json
from classification.experiments.config import ExperimentConfig
from classification.experiments.descriptions import load_descriptions
from classification.experiments.index_builder import IndexBuilder
from classification.experiments.report_types import (
    EvaluationReport,
    IndexSummary,
)
from classification.llm.generate_descriptors import DescriberLLM
from classification.llm.ollama import OllamaSession
from classification.retrieval.lancedb import LanceDBIndex
from classification.retrieval.metrics import metrics
from classification.schemas import EmbeddingBatch
from classification.tracking import Tracker


class Experiment:
    """Run food image classification experiment with image vector retrieval.

    Primary entry-point for image classification experiments. Uses LanceDB to
    build vector databases from image and text embeddings. Breaks experiments
    into three stages: (optionally) generating description captions via Ollama,
    building the vector database with the configured embedding model, and
    running classification evaluation.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        tracker: Tracker | None = None,
        dataset: FoodX251ImageDataset | None = None,
    ) -> None:
        self.tracker = tracker
        # Load dataset metadata and settings
        self.config = config
        self.output = Path(config.output_dir).expanduser().resolve()
        self.index_path = self.output / "index.json"
        descriptions_file = config.descriptions_file
        self.descriptions_path = (
            Path(descriptions_file).expanduser().resolve()
            if descriptions_file is not None
            else self.output / "descriptions.json"
        )
        # Sweeps pass one loaded dataset to every dimension's experiment.
        self.dataset = dataset or load_dataset(config.dataset)
        self.classes = self.dataset.get_class_list()

        self._embedder: Embedder | None = None

    @property
    def embedder(self) -> Embedder:
        """Construct the configured encoder on first access and reuse it."""
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    def _create_embedder(self) -> Embedder:
        options = self.config.embedding
        if options.model_id == "google/ge-v2":
            return VertexEmbeddingModel(
                **options.vertex.model_dump(), truncate_dim=options.truncate_dim
            )
        return EmbeddingModel(
            model_id=options.model_id,
            device=options.device,
            truncate_dim=options.truncate_dim,
            batch_size=options.batch_size,
        )

    def _evaluation_batches(
        self,
    ) -> Iterator[EmbeddingBatch]:
        """Encode uncached validation images in bounded batches."""
        for batch in self.dataset.iter_dataset_batches(
            self.dataset.validation_set(), self.config.embedding.batch_size
        ):
            vectors = self.embedder.encode_images(batch.pop("image"))
            yield batch, vectors

    def generate_descriptions(self) -> dict[str, list[str]] | None:
        """Load saved descriptions or generate and save a new vocabulary.

        An explicit descriptions file is read-only; a missing file fails without
        generation. Otherwise, reuse the output vocabulary or generate it through
        an owned Ollama session. File and generation errors propagate.

        Returns:
            A mapping from class labels to description lists, or None when
            description references are disabled.
        """
        if not self.config.embed_descriptions:
            logging.info("Description generation is disabled")
            return None
        path = self.descriptions_path
        if self.config.descriptions_file is not None or path.exists():
            logging.info(f"Loading existing descriptions at {path}")
            return load_descriptions(path)
        settings = self.config.ollama
        if settings is None or settings.llm is None or self.config.description_count is None:
            raise ValueError("Generating descriptions requires ollama.llm and description_count")
        self.output.mkdir(parents=True, exist_ok=True)
        with OllamaSession(settings) as session:
            llm = DescriberLLM(session.client, settings.llm)
            descriptions = llm.generate_descriptions(
                self.classes,
                self.config.description_count,
            )
        write_json(path, dict(descriptions=descriptions))
        return descriptions

    def _index_builder(self) -> IndexBuilder:
        return IndexBuilder(
            self.config,
            dataset=self.dataset,
            descriptions_path=self.descriptions_path,
            # enable lazy loading for embedding model
            get_embedder=lambda: self.embedder,
        )

    def build_vector_database(self) -> IndexSummary:
        """Build or resume references, optimize the table, and publish its summary."""
        index, counts = self._index_builder().build_reference_index()
        return self._publish_index(index, counts)

    def _publish_index(self, index: LanceDBIndex, counts: Counter[str]) -> IndexSummary:
        """Write index.json before publishing reference counts to the tracker."""
        artifact = IndexSummary(
            table=index.name,
            uri=str(index.uri),
            source_counts=counts,
            config=self.config,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.index_path, artifact.model_dump(mode="json"))
        if self.tracker:
            self.tracker.reference_counts(dict(counts))
        return artifact

    def _prepare_validation_batches(self, *, cached_only: bool) -> Iterator[EmbeddingBatch]:
        if not cached_only and not self.config.cache_validation_embeddings:
            return self._evaluation_batches()
        options = self.config.lancedb
        table = options.require_validation_table()
        if cached_only:
            cache = LanceDBIndex.open(uri=options.uri, table=table)
        else:
            cache = self._index_builder().build_validation_table(table)
        metadata = self.dataset.iter_dataset_batches(
            self.dataset.validation_set(), self.config.embedding.batch_size, decode=False
        )
        return (
            (
                batch,
                cache.fetch_vectors(
                    [("validation_image", identity) for identity in batch["source_id"]]
                ),
            )
            for batch in metadata
        )

    def evaluate(self) -> EvaluationReport:
        """Complete optional validation caching, then classify the full split."""
        index, source_counts = self._prepare_evaluation()
        batches = self._prepare_validation_batches(cached_only=False)
        return self._execute_evaluation(index, source_counts, batches)

    def evaluate_cached(self) -> EvaluationReport:
        """Classify the validation split using existing cached vectors only."""
        batches = self._prepare_validation_batches(cached_only=True)
        index, source_counts = self._prepare_evaluation()
        return self._execute_evaluation(index, source_counts, batches)

    def _prepare_evaluation(self) -> tuple[LanceDBIndex, dict[str, int] | None]:
        options = self.config.lancedb
        table = options.table
        index = LanceDBIndex.open(uri=options.uri, table=table)
        source_counts = None
        if self.index_path.exists():
            logging.info("Loading index.json from checkpoint...")
            summary = json.loads(self.index_path.read_text(encoding="utf-8"))
            # Only attach metadata identifying this database and table.
            if (
                summary["table"] == table
                and Path(summary["uri"]).expanduser().resolve() == index.uri
            ):
                source_counts = summary["source_counts"]
        if self.tracker:
            self.tracker.reference_counts(source_counts)
        return index, source_counts

    def _execute_evaluation(
        self,
        index: LanceDBIndex,
        source_counts: dict[str, int] | None,
        batches: Iterable[EmbeddingBatch],
    ) -> EvaluationReport:
        predictions = evaluation.evaluate_batches(
            batches,
            query=index.query_batch,
            class_counts=index.class_counts(),
            k=self.config.k,
            expected_count=len(self.dataset.validation_set()),
        )
        report = EvaluationReport(
            config=self.config,
            table=index.name,
            reference_source_counts=source_counts,
            predictions=predictions,
            metrics=metrics(predictions, self.classes),
        )
        run_dir = self.output / datetime.now(UTC).strftime("evaluation_%Y%m%dT%H%M%S_%fZ")
        run_dir.mkdir(parents=True)
        report_path = run_dir / "report.json"
        write_json(report_path, report.model_dump(mode="json"))
        if self.tracker:
            self.tracker.evaluation(report, report_path)
        return report

    def run(self) -> EvaluationReport:
        """Describe, index, and evaluate, closing Ollama before loading the embedder."""
        self.generate_descriptions()  # Owned Ollama context exits before embedder construction.
        self.build_vector_database()
        return self.evaluate()
