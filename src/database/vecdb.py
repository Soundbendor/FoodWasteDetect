import json
from collections import Counter
from typing import Tuple
from .embedding import EmbeddingModel

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


class VectorDB:
    def __init__(self, path: str, model):
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
    def query(self, query_text: str, strategy: str) -> Tuple[str, float]:
        query_vector = self.model.get_embedding(query_text)
        search_result = self.client.search(
            collection_name=self.db_name, query_vector=query_vector, limit=10
        )
        # Take top-1 result
        if strategy == "top1":
            category = search_result[0].payload["class"]
            confidence = search_result[0].score
            return category, confidence, search_result[0]
        if strategy == "voting":
            votes = Counter([x.payload["class"] for x in search_result])
            return votes.most_common(1)[0][0], None, None
