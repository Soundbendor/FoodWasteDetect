"""Build training and validation embedding LanceDB tables."""

import logging
from collections.abc import Callable
from itertools import batched
from pathlib import Path
from typing import Literal

from datasets import Dataset
from tqdm import tqdm

from classification.datasets.image_dataset import FoodX251ImageDataset
from classification.encoders.base import Embedder
from classification.experiments.config import ExperimentConfig
from classification.experiments.descriptions import load_descriptions, text_references
from classification.retrieval.lancedb import BUILD_SNAPSHOT_KEY, LanceDBIndex
from classification.schemas import PointRecord, SourceIdentity


class IndexBuilder:
    """Build or resume reference tables and validation caches by source key.

    Request the encoder only for missing source keys. Persist each completed batch
    before advancing progress; propagate failures without rollback.
    """

    def __init__(
        self,
        *,
        get_embedder: Callable[[], Embedder],
        dataset: FoodX251ImageDataset,
        uri: str | Path,
        table: str,
        config: ExperimentConfig,
        descriptions_path: Path,
        classes: list[str],
        batch_size: int,
    ) -> None:
        self._get_embedder = get_embedder
        self.dataset = dataset
        self.uri = uri
        self.table = table
        self.config = config
        self.descriptions_path = descriptions_path
        self.classes = classes
        self.batch_size = batch_size

    @property
    def embedder(self) -> Embedder:
        return self._get_embedder()

    def _get_text_inputs(self) -> list[PointRecord]:
        """Read enabled descriptions and construct text references without generation."""
        descriptions = (
            load_descriptions(self.descriptions_path) if self.config.embed_descriptions else None
        )
        return text_references(
            self.classes,
            include_class_labels=self.config.embed_class_labels,
            descriptions=descriptions,
        )

    def _insert_text_records(
        self,
        index: LanceDBIndex,
        records: list[PointRecord],
    ) -> None:
        """Embed and save text references in batches."""
        with tqdm(
            total=len(records), desc="Text references", unit="record", mininterval=5
        ) as progress:
            for batch in batched(records, self.batch_size, strict=False):
                vectors = self.embedder.encode_text([record.value for record in batch])
                index.append(batch, vectors)
                progress.update(len(batch))

    def _insert_image_records(
        self,
        index: LanceDBIndex,
        dataset: Dataset,
        *,
        split: Literal["train", "val"],
    ) -> None:
        """Decode and persist the prepared image dataset in batches."""
        kind = "training_image" if split == "train" else "validation_image"
        description = "Image references" if split == "train" else "Validation embeddings"
        with tqdm(desc=description, unit="image", mininterval=5) as progress:
            for batch in self.dataset.iter_dataset_batches(dataset, self.batch_size):
                records = [
                    PointRecord(label, kind, source_id, fname)
                    for label, source_id, fname in zip(
                        batch["class"], batch["source_id"], batch["fname"], strict=True
                    )
                ]
                vectors = self.embedder.encode_images(batch["image"])
                index.append(records, vectors)
                progress.update(len(records))

    def _prepare_reference_inputs(self) -> tuple[Dataset | None, list[PointRecord]]:
        images = None
        if self.config.embed_images:
            images = self.dataset.select_images(
                self.dataset.train_set(),
                n_imgs_per_class=self.config.n_imgs_per_class,
            )
            missing = set(self.classes) - set(images["class"])
            if missing:
                raise ValueError(f"Missing usable training references for {missing}")
        return images, self._get_text_inputs()

    def _open_or_create(self, table: str) -> LanceDBIndex:
        """Open the named table, creating it with the encoder width only when absent."""
        return LanceDBIndex.open_or_create(
            self.uri,
            table,
            lambda: self.embedder.dimension,
            metadata={BUILD_SNAPSHOT_KEY: self.config.model_dump_json()},
        )

    def _pending_images(self, images: Dataset, kind: str, missing: list[SourceIdentity]) -> Dataset:
        """Keep selected images whose source keys are missing, in selection order."""
        pending = {identity for missing_kind, identity in missing if missing_kind == kind}
        saved = {identity for identity in images["source_id"] if identity not in pending}
        return self.dataset.select_images(images, excluded_source_ids=saved)

    def build_reference_table(self) -> LanceDBIndex:
        """Build or resume the reference table, then mark it complete and optimized.

        Reference inputs are validated before the table is opened or the encoder is
        constructed. Only missing source keys are embedded.

        Raises:
            ValueError: If no reference inputs are selected, or if expected source
                keys are still missing after insertion.
        """
        images, records = self._prepare_reference_inputs()
        expected = [(record.source_kind, record.source_identity) for record in records]
        if images is not None:
            expected.extend(("training_image", identity) for identity in images["source_id"])
        if not expected:
            raise ValueError("No reference records selected - check dataset and config.")
        index = self._open_or_create(self.table)
        missing = index.missing(expected)
        pending = frozenset(missing)
        records = [r for r in records if (r.source_kind, r.source_identity) in pending]
        if images is not None:
            images = self._pending_images(images, "training_image", missing)
        self._insert_references(index, records, images)
        logging.info("Verifying and optimizing LanceDB table...")
        index.mark_complete(expected, optimize=True)
        return index

    def _insert_references(
        self,
        index: LanceDBIndex,
        records: list[PointRecord],
        images: Dataset | None,
    ) -> None:
        self._insert_text_records(index, records)
        if images is not None:
            self._insert_image_records(index, images, split="train")

    def build_validation_cache(self, table_name: str) -> LanceDBIndex:
        """Complete missing validation embeddings and return the unoptimized cache.

        Raises:
            ValueError: If validation source keys are still missing after insertion.
        """
        validation = self.dataset.validation_set()
        expected = [("validation_image", identity) for identity in validation["source_id"]]
        cache = self._open_or_create(table_name)
        images = self._pending_images(validation, "validation_image", cache.missing(expected))
        self._insert_image_records(cache, images, split="val")
        cache.mark_complete(expected, optimize=False)
        return cache
