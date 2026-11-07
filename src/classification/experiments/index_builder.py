"""Build training and validation embedding LanceDB tables."""

import logging
from collections import Counter
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
from classification.schemas import PointRecord


class IndexBuilder:
    """Resume reference and validation tables by saved source identity.

    Request the encoder only for missing records. Persist each completed batch
    before advancing counts or progress; propagate failures without rollback.
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
    ) -> Counter[str]:
        """Embed and save text references, reporting successfully inserted records."""
        counts: Counter[str] = Counter()
        with tqdm(
            total=len(records), desc="Text references", unit="record", mininterval=5
        ) as progress:
            for batch in batched(records, self.batch_size, strict=False):
                vectors = self.embedder.encode_text([record.value for record in batch])
                index.append(batch, vectors)
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
                vectors = self.embedder.encode_images(batch["image"])
                index.append(records, vectors)
                counts.update(record.source_kind for record in records)
                progress.update(len(records))
        return counts

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

    def _prepare_new_reference_build(
        self,
    ) -> tuple[LanceDBIndex, Dataset | None, list[PointRecord], Counter[str]]:
        images, records = self._prepare_reference_inputs()
        dimension = self.embedder.dimension
        self.config.vector_dimension = dimension
        index = LanceDBIndex.create(
            self.uri,
            self.table,
            dimension,
            metadata={BUILD_SNAPSHOT_KEY: self.config.model_dump_json()},
        )
        return index, images, records, Counter()

    def _prepare_resumed_reference_build(
        self, index: LanceDBIndex
    ) -> tuple[Dataset | None, list[PointRecord], Counter[str]]:
        logging.info("Loading existing table...")
        images, records = self._prepare_reference_inputs()
        vector_ids = index.get_record_ids()
        existing_ids = frozenset(vector_ids)
        # filter existing records from incoming data
        records = [r for r in records if (r.source_kind, r.source_identity) not in existing_ids]
        if images is not None:
            excluded = {identity for kind, identity in existing_ids if kind == "training_image"}
            images = self.dataset.select_images(images, excluded_source_ids=excluded)
        return images, records, Counter(kind for kind, _ in vector_ids)

    def build_reference_index(self) -> tuple[LanceDBIndex, Counter[str]]:
        """Build or resume and optimize the reference table, returning it and its counts."""
        if LanceDBIndex.exists(self.uri, self.table):
            index = LanceDBIndex.open(self.uri, self.table)
            images, records, counts = self._prepare_resumed_reference_build(index)
        else:
            index, images, records, counts = self._prepare_new_reference_build()
        counts.update(self._insert_references(index, records, images))
        if not counts:
            raise ValueError("No records inserted into LanceDB - check dataset and config.")
        logging.info("Compressing LanceDB table...")
        index.optimize()
        return index, counts

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

    def build_validation_table(self, table_name: str) -> LanceDBIndex:
        """Complete missing validation embeddings and return the unoptimized cache."""
        if LanceDBIndex.exists(self.uri, table_name):
            cache = LanceDBIndex.open(self.uri, table_name)
        else:
            dimension = self.embedder.dimension
            self.config.vector_dimension = dimension
            cache = LanceDBIndex.create(
                self.uri,
                table_name,
                dimension,
                metadata={BUILD_SNAPSHOT_KEY: self.config.model_dump_json()},
            )
        excluded = {
            identity for kind, identity in cache.get_record_ids() if kind == "validation_image"
        }
        images = self.dataset.select_images(
            self.dataset.validation_set(), excluded_source_ids=excluded
        )
        self._insert_image_records(cache, images, split="val")
        return cache
