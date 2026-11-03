"""Configurable whole-image classification with explicit backend construction."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from classification.datasets.image_dataset import load_dataset
from classification.encoders.base import Embedder
from classification.encoders.sentence_transformers import EmbeddingModel
from classification.encoders.vertex import VertexEmbeddingModel
from classification.experiments.artifacts import write_json
from classification.experiments.batches import EmbeddingBatch, cached_batches
from classification.experiments.config import validate_config, validate_ollama_config
from classification.experiments.descriptions import load_descriptions
from classification.experiments.evaluation import evaluate_batches
from classification.experiments.index_builder import IndexBuilder
from classification.experiments.report_types import (
    EvaluationReport,
    IndexSummary,
    LoadedIndexSummary,
)
from classification.llm.generate_descriptors import DescriberLLM
from classification.llm.ollama import OllamaSession
from classification.retrieval.lancedb import LanceDBIndex
from classification.retrieval.metrics import metrics
from classification.tracking import Tracker


class Experiment:
    """Classify whole images with configured embeddings and LanceDB retrieval.

    Construction loads dataset metadata; the encoder loads on first use.
    Stages generate or reuse descriptions, build references, and publish complete
    evaluation reports. Tracking uses the supplied tracker when present.
    """

    @staticmethod
    def validate_config(config: dict[str, Any]) -> dict[str, Any]:
        """Apply shared validation and return the supplied configuration unchanged."""
        return validate_config(config)

    def __init__(self, config: dict[str, Any], *, tracker: Tracker | None = None) -> None:
        self.tracker = tracker
        # Load dataset metadata and settings
        self.config = validate_config(config)
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
        """Construct the configured encoder on first access and reuse it."""
        if self._embedder is None:
            self._embedder = self._create_embedder()
        return self._embedder

    def _create_embedder(self) -> Embedder:
        options = self.config["embedding"]
        if options.get("model_id") == "google/ge-v2":
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
        return LanceDBIndex.open(uri=options["uri"], table=options["table"])

    def _evaluation_batches(
        self,
    ) -> Iterator[EmbeddingBatch]:
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
        if not self.config["embed_descriptions"]:
            logging.info("Description generation is disabled")
            return None
        path = self.descriptions_path
        if self.config.get("descriptions_file") is not None or path.exists():
            logging.info(f"Loading existing descriptions at {path}")
            return load_descriptions(path)
        validate_ollama_config(self.config["ollama"])
        self.output.mkdir(parents=True, exist_ok=True)
        model = self.config["ollama"]["llm"]
        with OllamaSession(self.config["ollama"]) as session:
            llm = DescriberLLM(session.client, model)
            descriptions = llm.generate_descriptions(
                self.classes,
                self.config["description_count"],
            )
        write_json(path, dict(descriptions=descriptions))
        return descriptions

    def _index_builder(self) -> IndexBuilder:
        options = self.config["lancedb"]
        return IndexBuilder(
            get_embedder=lambda: self.embedder,
            dataset=self.dataset,
            uri=options["uri"],
            table=options["table"],
            config=self.config,
            descriptions_path=self.descriptions_path,
            classes=self.classes,
            batch_size=self.batch_size,
        )

    def build_vector_database(self) -> IndexSummary:
        """Build or resume references, optimize the table, and publish its summary."""
        index, counts = self._index_builder().build_reference_index()
        return self._publish_index(index, counts)

    def _publish_index(self, index: LanceDBIndex, counts: Counter[str]) -> IndexSummary:
        """Write index.json before publishing reference counts to the tracker."""
        artifact: IndexSummary = dict(
            table=index.name,
            uri=str(index.uri),
            source_counts=counts,
            config=self.config,
        )
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "index.json", artifact)
        if self.tracker:
            self.tracker.reference_counts(dict(counts))
        return artifact

    def evaluate(self) -> EvaluationReport:
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
    ) -> EvaluationReport:
        """Evaluate supplied, ordered metadata/vector batches through standard reporting."""
        index, artifact = self._prepare_evaluation()
        return self._execute_evaluation(index, artifact, batches)

    def _prepare_evaluation(self) -> tuple[LanceDBIndex, LoadedIndexSummary]:
        table = self.config["lancedb"]["table"]
        index = self._lancedb_adapter()
        artifact: LoadedIndexSummary = {}
        path = self.output / "index.json"
        if path.exists():
            logging.info("Loading index.json from checkpoint...")
            summary: LoadedIndexSummary = json.loads(path.read_text(encoding="utf-8"))
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
        artifact: LoadedIndexSummary,
        batches: Iterable[EmbeddingBatch],
    ) -> EvaluationReport:
        predictions = evaluate_batches(
            batches,
            query=index.query,
            class_counts=index.class_counts(),
            k=self.config["k"],
            expected_count=len(self.dataset.validation_set()),
        )
        report: EvaluationReport = dict(
            config=self.config,
            table=index.name,
            reference_source_counts=artifact.get("source_counts"),
            predictions=predictions,
            metrics=metrics(predictions, self.classes),
        )
        run_dir = self.output / datetime.now(UTC).strftime("evaluation_%Y%m%dT%H%M%S_%fZ")
        run_dir.mkdir(parents=True)
        report_path = run_dir / "report.json"
        write_json(report_path, report)
        if self.tracker:
            self.tracker.evaluation(report, report_path)
        return report

    def run(self) -> EvaluationReport:
        """Describe, index, and evaluate, closing Ollama before loading the embedder."""
        self.generate_descriptions()  # Owned Ollama context exits before embedder construction.
        self.build_vector_database()
        return self.evaluate()
