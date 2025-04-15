import json
from collections import Counter
from typing import Tuple, List
from .embedding import EmbeddingModel

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, ScoredPoint


class VectorDB:
    def __init__(self, path: str, model: EmbeddingModel):
        self.client = QdrantClient(path=path)
        self.db_name = "foods_finetune"
        self.model = model
        # self.client.set_model("BAAI/bge-large-en-v1.5", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def add(self, dict_path: str):
        # Expecting a descriptor dictionary as
        # { 'food_item': ['description 1', 'description 2'], ... }

        with open(dict_path, "r", encoding="utf-8") as f:
            descriptors = json.load(f)

        points = []
        for idx, (k, v) in enumerate(descriptors.items()):
            for descriptor in v:
                points.append(
                    PointStruct(
                        id=idx,
                        vector=self.model.get_embedding(descriptor),
                        payload={"class": k},
                    )
                )

        if not self.client.collection_exists(self.db_name):
            self.client.create_collection(
                collection_name=self.db_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        self.client.upsert(collection_name=self.db_name, points=points)

    # Given a food image descriptor, return the most probable class and similarity score
    def query(self, query_text: str) -> List[ScoredPoint]:
        query_vector = self.model.get_embedding(query_text)
        return self.client.search(
            collection_name=self.db_name, query_vector=query_vector, limit=10
        )

    # Returns (Accuracy, Similarity) where acc is binary 1-0
    def score(
        self, search_result: List[ScoredPoint], label: str, strategy: str
    ) -> Tuple[int, float]:
        if strategy == "top1":
            category = search_result[0].payload["class"]  # type: ignore
            confidence = search_result[0].score
            score = 1 if category.strip() == label.strip() else 0
        if strategy == "voting":
            votes = Counter([x.payload["class"] for x in search_result])  # type: ignore
            candidate = votes.most_common(1)[0]
            category = candidate[0]
            confidence = candidate[1] / 10
            score = 1 if category.strip() == label.strip() else 0
        if strategy == "top5":
            score = 1 if label.strip in [x.payload["class"].strip() for x in search_result] else 0  # type: ignore
            confidence = 0
        return score, confidence
