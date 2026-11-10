"""Build training and validation embedding LanceDB tables."""

import logging
from collections import Counter
from collections.abc import Callable
from itertools import batched
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

from classification.datasets.image_dataset import FoodX251ImageDataset
from classification.encoders.base import Embedder
from classification.experiments.config import ExperimentConfig
from classification.experiments.descriptions import load_descriptions, text_references
from classification.retrieval.lancedb import LanceDBIndex
from classification.schemas import PointRecord


class IndexBuilder:
    """Resume reference and validation tables by saved source identity.

    Request the encoder only for missing records. Persist each completed batch
    before advancing progress; propagate failures without rollback.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        dataset: FoodX251ImageDataset,
        descriptions_path: Path,
        get_embedder: Callable[[], Embedder],
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.descriptions_path = descriptions_path
        self._get_embedder = get_embedder
        self.uri = config.lancedb.uri
        self.table = config.lancedb.table
        self.batch_size = config.embedding.batch_size
        self.classes = dataset.get_class_list()

    @property
    def embedder(self) -> Embedder:
        return self._get_embedder()

    def _create_table(self, name: str) -> LanceDBIndex:
        """Record the encoder dimension in the config, then snapshot it as build metadata."""
        self.config.vector_dimension = self.embedder.dimension
        return LanceDBIndex.create(
            self.uri,
            name,
            self.embedder.dimension,
            metadata={"classification.build": self.config.model_dump_json()},
        )

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
        """Embed and save text references."""
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
        kind: str,
        description: str,
    ) -> None:
        """Decode and persist the prepared image dataset in batches."""
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

    def build_reference_index(self) -> tuple[LanceDBIndex, Counter[str]]:
        """Build or resume and optimize the reference table, returning it and its counts."""
        images, records = self._prepare_reference_inputs()
        if LanceDBIndex.exists(self.uri, self.table):
            logging.info("Loading existing table...")
            index = LanceDBIndex.open(self.uri, self.table)
            existing_ids = frozenset(index.get_record_ids())
            # filter existing records from incoming data
            records = [r for r in records if (r.source_kind, r.source_identity) not in existing_ids]
            if images is not None:
                excluded = {identity for kind, identity in existing_ids if kind == "training_image"}
                images = self.dataset.select_images(images, excluded_source_ids=excluded)
        else:
            index = self._create_table(self.table)
        self._insert_text_records(index, records)
        if images is not None:
            self._insert_image_records(
                index, images, kind="training_image", description="Image references"
            )
        # The table is the resume source of truth, so count what it now holds.
        counts = Counter(kind for kind, _ in index.get_record_ids())
        if not counts:
            raise ValueError("No records inserted into LanceDB - check dataset and config.")
        logging.info("Compressing LanceDB table...")
        index.optimize()
        return index, counts

    def build_validation_table(self, table_name: str) -> LanceDBIndex:
        """Complete missing validation embeddings and return the unoptimized cache."""
        if LanceDBIndex.exists(self.uri, table_name):
            cache = LanceDBIndex.open(self.uri, table_name)
        else:
            cache = self._create_table(table_name)
        excluded = {
            identity for kind, identity in cache.get_record_ids() if kind == "validation_image"
        }
        images = self.dataset.select_images(
            self.dataset.validation_set(), excluded_source_ids=excluded
        )
        self._insert_image_records(
            cache, images, kind="validation_image", description="Validation embeddings"
        )
        return cache
