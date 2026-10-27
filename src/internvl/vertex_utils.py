"""Vertex experiment orchestration and explicit-input persistence helpers."""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .experiment import Experiment
from .retrieval import LanceDBIndex
from .schemas import PointRecord
from .vertex_embeddings import VertexEmbeddingModel


def validate_cache_config(config: dict) -> None:
    """Require a separate validation table only when caching is truthy."""
    if config.get("cache_validation_embeddings", False):
        name = config["lancedb"].get("validation_table")
        if not isinstance(name, str) or not name.strip() or name == config["lancedb"]["table"]:
            raise ValueError("validation_table must be nonempty and distinct from reference table")


def _embedding_callback(
    index: LanceDBIndex, records: Sequence[PointRecord],
) -> Callable[[int, NDArray[np.float32]], None]:
    """Persist each response against its original input position."""
    def save(position: int, vector: NDArray[np.float32]) -> None:
        index.append([records[position]], vector[None, :])

    return save


def save_text_batch(
    encoder: VertexEmbeddingModel, index: LanceDBIndex, records: Sequence[PointRecord],
) -> None:
    encoder.encode_text(
        [record.value for record in records], on_embedding=_embedding_callback(index, records),
    )


def save_image_batch(
    encoder: VertexEmbeddingModel, index: LanceDBIndex,
    records: Sequence[PointRecord], images: list[Image.Image],
) -> None:
    encoder.encode_images(images, on_embedding=_embedding_callback(index, records))


@dataclass
class _ReferencePreparation:
    pending_text: list[PointRecord]
    excluded_source_ids: set[str]
    source_counts: Counter[str]


def prepare_references(
    index: LanceDBIndex, *, exists: bool, records: Sequence[PointRecord],
) -> _ReferencePreparation:
    """Scan once; count all stored rows but filter by immutable identity membership."""
    if exists:
        index.open()
    identities = index.identities() if exists else []
    saved = frozenset(identities)
    return _ReferencePreparation(
        pending_text=[r for r in records if (r.source_kind, r.source_identity) not in saved],
        excluded_source_ids={identity for kind, identity in saved if kind == "training_image"},
        source_counts=Counter(kind for kind, _ in identities),
    )


def prepare_validation_cache(
    cache: LanceDBIndex, *,
    dimension: Callable[[], int],
    insert_images: Callable[[LanceDBIndex, set[str]], object],
) -> LanceDBIndex:
    """Complete the separate cache, constructing an encoder only if needed."""
    if cache.exists():
        cache.open()
    else:
        cache.create(dimension())
    saved = frozenset(cache.identities())
    excluded = {identity for kind, identity in saved if kind == "validation_image"}
    insert_images(cache, excluded)
    return cache


def cached_batches(
    cache: LanceDBIndex, metadata_batches: Iterable[dict[str, list]],
) -> Iterator[tuple[dict[str, list], np.ndarray[tuple[int, int], np.dtype[np.float32]]]]:
    """Fetch bounded vectors in the annotation order supplied by metadata iteration."""
    for batch in metadata_batches:
        yield batch, cache.fetch_vectors([
            ("validation_image", identity) for identity in batch["source_id"]
        ])


class VertexExperiment(Experiment):
    """Resume Vertex references and optionally cache validation embeddings."""

    @staticmethod
    def validate_config(config: dict) -> dict:
        Experiment.validate_config(config)
        validate_cache_config(config)
        return config

    def _create_embedder(self) -> VertexEmbeddingModel:
        options = self.config["embedding"]
        return VertexEmbeddingModel(
            truncate_dim=options.get("truncate_dim"), **options.get("vertex", {}),
        )

    def _save_text_batch(self, index: LanceDBIndex, records: list[PointRecord]) -> None:
        save_text_batch(cast(VertexEmbeddingModel, self.embedder), index, records)

    def _save_image_batch(
        self, index: LanceDBIndex, records: list[PointRecord], images: list[Image.Image],
    ) -> None:
        save_image_batch(cast(VertexEmbeddingModel, self.embedder), index, records, images)

    def build_vector_database(self) -> dict:
        index = self._lancedb_adapter()
        exists = index.exists()
        records = self._prepare_reference_inputs()
        prepared = prepare_references(
            index, exists=exists, records=records,
        )
        if not exists:
            index.create(self.embedder.dimension)
        counts = prepared.source_counts
        counts.update(self._insert_references(
            index, prepared.pending_text, prepared.excluded_source_ids,
        ))
        return self._publish_index(index, counts)

    def evaluate(self) -> dict:
        if not self.config.get("cache_validation_embeddings", False):
            return super().evaluate()
        index, artifact = self._prepare_evaluation()
        options = self.config["lancedb"]
        cache = prepare_validation_cache(
            LanceDBIndex(uri=options["uri"], table=options["validation_table"]),
            dimension=lambda: self.embedder.dimension,
            insert_images=lambda index, excluded: self._insert_image_records(
                index, split="val", excluded_source_ids=excluded,
            ),
        )
        batches = cached_batches(
            cache, self.dataset.iter_batches("val", self.batch_size, decode=False),
        )
        return self._execute_evaluation(index, artifact, batches)
