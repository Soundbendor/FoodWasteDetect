"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from classification.datasets.image_dataset import load_dataset
from classification.encoders.base import Embedder
from classification.encoders.sentence_transformers import EmbeddingModel
from classification.encoders.vertex import VertexEmbeddingModel
from classification.experiments.artifacts import write_json
from classification.experiments.batches import EmbeddingBatch
from classification.experiments.config import ExperimentConfig
from classification.experiments.descriptions import load_descriptions
from classification.experiments.evaluation import evaluate_batches
from classification.experiments.fingerprints import encoder_fingerprint
from classification.experiments.index_builder import IndexBuilder
from classification.experiments.report_types import (
    EvaluationReport,
    IndexSummary,
)
from classification.llm.generate_descriptors import DescriberLLM
from classification.llm.ollama import OllamaSession
from classification.retrieval.embedding_table import EmbeddingTable, Fingerprint
from classification.retrieval.metrics import metrics
from classification.tracking import Tracker


class Experiment:
    """Classify whole images with configured embeddings and LanceDB retrieval.

    Construction loads dataset metadata; the encoder loads on first use.
    Stages generate or reuse descriptions, build references, and publish complete
    evaluation reports. Tracking uses the supplied tracker when present.
    """

    def __init__(self, config: ExperimentConfig, *, tracker: Tracker | None = None) -> None:
        self.tracker = tracker
        # Load dataset metadata and settings
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config
        self.output = Path(config.output_dir).expanduser().resolve()
        descriptions_file = config.descriptions_file
        self.descriptions_path = (
            Path(descriptions_file).expanduser().resolve()
            if descriptions_file is not None
            else self.output / "descriptions.json"
        )
        self.dataset = load_dataset(config.dataset)
        self.classes = self.dataset.get_class_list()
        self.batch_size = config.embedding.batch_size

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
            vertex = options.vertex
            if not vertex.project or not vertex.project.strip():
                raise ValueError("embedding.vertex.project is required for the Vertex backend")
            return VertexEmbeddingModel(
                project=vertex.project,
                location=vertex.location,
                max_concurrency=vertex.max_concurrency,
                max_attempts=vertex.max_attempts,
                request_timeout=vertex.request_timeout,
                classification_prompt=vertex.classification_prompt,
                truncate_dim=options.truncate_dim,
            )
        return EmbeddingModel(
            model_id=options.model_id,
            device=options.device,
            truncate_dim=options.truncate_dim,
            batch_size=self.batch_size,
        )

    def _reference_table(
        self, *, encoder: Fingerprint, dimension: int | None = None
    ) -> EmbeddingTable:
        """Open the complete reference table, requiring the validation vectors' encoder."""
        options = self.config.lancedb
        return EmbeddingTable.open_complete(
            options.uri,
            options.table,
            "reference",
            expected=self._index_builder().expected_reference_keys,
            encoder=encoder,
            dimension=dimension,
        )

    def _evaluation_batches(self) -> Iterator[EmbeddingBatch]:
        """Encode uncached validation images in bounded batches."""
        for batch in self.dataset.iter_batches("val", self.batch_size):
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
        model = settings.llm
        with OllamaSession(settings) as session:
            llm = DescriberLLM(session.client, model)
            descriptions = llm.generate_descriptions(
                self.classes,
                self.config.description_count,
            )
        write_json(path, dict(descriptions=descriptions))
        return descriptions

    def _index_builder(self) -> IndexBuilder:
        options = self.config.lancedb
        return IndexBuilder(
            get_embedder=lambda: self.embedder,
            dataset=self.dataset,
            uri=options.uri,
            table=options.table,
            config=self.config,
            descriptions_path=self.descriptions_path,
            classes=self.classes,
            batch_size=self.batch_size,
        )

    def build_vector_database(self) -> IndexSummary:
        """Build or resume the reference table, mark it complete, and publish its summary."""
        reference = self._index_builder().build_reference_table()
        return self._publish_index(reference)

    def _publish_index(self, reference: EmbeddingTable) -> IndexSummary:
        """Write index.json before publishing the table's source counts to the tracker.

        Duplicate source keys are tolerated; a nonzero count is logged as a warning.
        """
        counts = dict(reference.source_counts())
        duplicates = reference.duplicate_count()
        if duplicates:
            logging.warning(
                "Reference table %r holds %d duplicate source key rows; source and class "
                "counts include them",
                reference.name,
                duplicates,
            )
        artifact = IndexSummary(
            table=reference.name,
            uri=str(reference.uri),
            dimension=reference.dimension,
            source_counts=counts,
            duplicate_count=duplicates,
            config=self.config,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "index.json", artifact.model_dump(mode="json"))
        if self.tracker:
            self.tracker.reference_counts(counts)
        return artifact

    def _validation_table_name(self) -> str:
        table = self.config.lancedb.validation_table
        if table is None:
            raise ValueError("lancedb.validation_table is required for cached evaluation")
        return table

    def _cached_batches(self, cache: EmbeddingTable) -> Iterator[EmbeddingBatch]:
        """Pair validation metadata, read eagerly, with lazily fetched cached vectors."""
        metadata = self.dataset.iter_batches("val", self.batch_size, decode=False)
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
        """Complete optional validation caching, then classify the full split.

        The reference table must record the configured encoder fingerprint, and the
        validation vectors' width must equal its dimension, before any validation
        image is encoded.
        """
        reference, counts = self._prepare_evaluation(encoder=encoder_fingerprint(self.config))
        if self.config.cache_validation_embeddings:
            cache = self._index_builder().build_validation_cache(self._validation_table_name())
            reference.require_compatible(
                encoder=cache.encoder_fingerprint, dimension=cache.dimension
            )
            batches = self._cached_batches(cache)
        else:
            reference.require_compatible(dimension=self.embedder.dimension)
            batches = self._evaluation_batches()
        return self._execute_evaluation(reference, counts, batches)

    def evaluate_cached(self) -> EvaluationReport:
        """Classify the validation split using existing cached vectors only.

        The reference table must record the validation cache's encoder fingerprint
        and dimension; the configured encoder settings are not consulted.
        """
        cache = EmbeddingTable.open_complete(
            self.config.lancedb.uri,
            self._validation_table_name(),
            "validation_cache",
            expected=self._index_builder().expected_validation_keys,
        )
        batches = self._cached_batches(cache)
        reference, counts = self._prepare_evaluation(
            encoder=cache.encoder_fingerprint, dimension=cache.dimension
        )
        return self._execute_evaluation(reference, counts, batches)

    def _prepare_evaluation(
        self, *, encoder: Fingerprint, dimension: int | None = None
    ) -> tuple[EmbeddingTable, dict[str, int]]:
        """Open a compatible reference table and publish its stored source-kind counts."""
        reference = self._reference_table(encoder=encoder, dimension=dimension)
        counts = dict(reference.source_counts())
        if self.tracker:
            self.tracker.reference_counts(counts)
        return reference, counts

    def _execute_evaluation(
        self,
        reference: EmbeddingTable,
        reference_source_counts: dict[str, int],
        batches: Iterable[EmbeddingBatch],
    ) -> EvaluationReport:
        predictions = evaluate_batches(
            batches,
            query=reference.query,
            class_counts=reference.class_counts(),
            k=self.config.k,
            expected_count=len(self.dataset.validation_set()),
        )
        report = EvaluationReport(
            config=self.config,
            table=reference.name,
            dimension=reference.dimension,
            reference_source_counts=reference_source_counts,
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
