"""Configurable whole-image classification; help/import never load backends."""

import argparse
from collections import Counter
import json
from pathlib import Path
import uuid

from .records import ReferenceRecord


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def validate_config(config):
    # Verify that at least one modality is included in the embedding database.
    sources = (
        "embed_images",
        "embed_class_labels",
        "embed_descriptions",
    )
    if not any(config[key] for key in sources):
        raise ValueError("Select at least one reference source")
    return config


class Experiment:
    def __init__(
        self,
        config,
        *,
        dataset=None,
        embedder_factory=None,
        client=None,
        describer_factory=None,
    ):
        self.config = validate_config(config)
        if dataset is None:
            from .image_dataset import load_dataset

            dataset = load_dataset(self.config["dataset"])
        self.dataset = dataset
        self.classes = dataset.get_class_list()
        self.output = Path(self.config["output_dir"]).expanduser().resolve()
        # Never write derived artifacts into the raw dataset tree.
        raw = self.config.get("dataset", {}).get("root")
        if raw and self.output.is_relative_to(Path(raw).expanduser().resolve()):
            raise ValueError("Output directory must be separate from raw data")
        self.output.mkdir(parents=True, exist_ok=True)
        self.embedder_factory = embedder_factory
        self.describer_factory = describer_factory
        self.client = client
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            factory = self.embedder_factory
            if factory is None:
                from .embeddings import EmbeddingModel

                factory = EmbeddingModel
            self._embedder = factory(**self.config["embedding"])
        return self._embedder

    def index_adapter(self, collection):
        from .retrieval import QdrantIndex

        if self.client is None:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(**self.config["qdrant"]["connection"])
        return QdrantIndex(self.client, collection, self.embedder.dimension)

    def describe(self):
        if not self.config["embed_descriptions"]:
            return None
        path = self.output / "descriptions.json"
        if path.exists():
            return json.loads(path.read_text())["descriptions"]
        factory = self.describer_factory
        if factory is None:
            from .generate_descriptors import DescriberLLM

            factory = DescriberLLM
        artifact = dict(descriptions={})
        with factory(self.config["ollama"]) as llm:
            for label in self.classes:
                artifact["descriptions"][label] = [
                    llm._describe_class(
                        label,
                        {
                            "role": "user",
                            "content": (
                                f"Can you provide a sentence describing the food {label}? "
                                "Please be as descriptive as possible, focusing on the visual "
                                f"characteristics of {label} and explicitly mentioning "
                                f"{label} in your description."
                            ),
                        },
                    ).description
                    for _ in range(self.config["description_count"])
                ]
                if any(not text.strip() for text in artifact["descriptions"][label]):
                    raise ValueError(f"Empty generated description for {label}")
        save(path, artifact)
        return artifact["descriptions"]

    def references(self) -> list[ReferenceRecord]:
        records = []
        if self.config["embed_class_labels"]:
            records += [ReferenceRecord(c, "class_name", c, c) for c in self.classes]
        if self.config["embed_descriptions"]:
            path = self.output / "descriptions.json"
            if not path.exists():
                raise ValueError("Run describe before index/evaluate")
            data = json.loads(path.read_text())["descriptions"]
            records += [
                ReferenceRecord(c, "generated_description", f"{c}:{i}", text)
                for c in self.classes
                for i, text in enumerate(data[c])
            ]
        if self.config["embed_images"]:
            records += [
                ReferenceRecord(
                    row["class"], "training_image", row["source_id"], row["fname"]
                )
                for row in self.dataset.train_set().remove_columns("image")
            ]
        return records

    def index(self):
        records = self.references()
        path = self.output / "index.json"
        collection = self.config["qdrant"]["collection_prefix"] + "_" + uuid.uuid4().hex
        artifact = dict(
            state="incomplete",
            collection=collection,
            model=self.embedder.metadata,
            source_counts=Counter(),
            config=self.config,
        )
        image_classes = set()
        save(path, artifact)
        index = self.index_adapter(collection)
        try:
            index.create()
            batch_size = self.config["embedding"].get("batch_size", 32)
            text = [r for r in records if r.source_kind != "training_image"]
            for start in range(0, len(text), batch_size):
                batch = text[start : start + batch_size]
                vectors = self.embedder.encode_text([r.value for r in batch])
                index.upsert(batch, vectors)
                artifact["source_counts"].update(r.source_kind for r in batch)
            if self.config["embed_images"]:
                references = {
                    r.source_identity: r
                    for r in records
                    if r.source_kind == "training_image"
                }
                for batch in self.dataset.iter_batches("train", batch_size):
                    try:
                        image_records = [
                            references[source_id] for source_id in batch["source_id"]
                        ]
                        vectors = self.embedder.encode_images(batch["image"])
                        index.upsert(image_records, vectors)
                        artifact["source_counts"].update(
                            r.source_kind for r in image_records
                        )
                        image_classes.update(r.label for r in image_records)
                    finally:
                        for image in batch["image"]:
                            image.close()
                if image_classes != set(self.classes):
                    raise ValueError(
                        f"Missing usable training references for {set(self.classes) - image_classes}"
                    )
            if not artifact["source_counts"]:
                raise ValueError("Empty reference index")
            artifact["state"] = "complete"
        except BaseException as exc:
            artifact["error"] = str(exc)
            raise
        finally:
            save(path, artifact)
        return artifact

    def evaluate(self):
        from .retrieval import classify, metrics

        run_dir = self.output / ("evaluation_" + uuid.uuid4().hex)
        run_dir.mkdir()
        report = dict(
            state="incomplete",
            config=self.config,
            retrieval={"k": self.config["k"], "exact": True, "distance": "Cosine"},
            predictions=[],
        )
        counts = {"available": 0, "embedded": 0}
        records = []
        try:
            records = [
                dict(row, value=row["fname"])
                for row in self.dataset.test_set().remove_columns("image")
            ]
            report["evaluation_manifest"] = records
            artifact = json.loads((self.output / "index.json").read_text())
            index = self.index_adapter(artifact["collection"])
            report["collection"] = artifact["collection"]
            report["model"] = artifact["model"]
            report["reference_source_counts"] = artifact["source_counts"]
            for batch in self.dataset.iter_batches(
                "val", self.config["embedding"].get("batch_size", 32)
            ):
                try:
                    counts["available"] += len(batch["image"])
                    vectors = self.embedder.encode_images(batch["image"])
                    counts["embedded"] += len(vectors)
                    metadata = {
                        key: values for key, values in batch.items() if key != "image"
                    }
                    rows = (
                        dict(zip(metadata, values))
                        for values in zip(*metadata.values())
                    )
                    for record, vector in zip(rows, vectors):
                        neighbors = index.query(vector, self.config["k"])
                        label, fraction = classify(neighbors)
                        report["predictions"].append(
                            dict(
                                record,
                                value=record["fname"],
                                prediction=label,
                                vote_fraction=fraction,
                                nearest=neighbors[0].label if neighbors else None,
                                neighbors=[vars(n) for n in neighbors],
                            )
                        )
                finally:
                    for image in batch["image"]:
                        image.close()
            if not counts["embedded"]:
                raise ValueError(
                    "No evaluation images could be embedded; accuracy is undefined"
                )
            report["state"] = "complete"
        except BaseException as exc:
            report["error"] = str(exc)
            raise
        finally:
            report["metrics"] = metrics(
                report["predictions"],
                len(records),
                counts["available"],
                counts["embedded"],
                self.classes,
            )
            save(run_dir / "report.json", report)
            save(run_dir / "predictions.json", report["predictions"])
            save(run_dir / "metrics.json", report["metrics"])
        return report

    def run(self):
        self.describe()  # Owned Ollama context has exited before embedding model construction.
        self.index()
        return self.evaluate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("describe", "index", "evaluate", "run"))
    parser.add_argument("--config_file", required=True)
    args = parser.parse_args(argv)
    import yaml

    with open(args.config_file) as stream:
        config = yaml.safe_load(stream)
    getattr(Experiment(config), args.stage)()


if __name__ == "__main__":
    main()
