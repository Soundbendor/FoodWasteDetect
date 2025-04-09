import json
from typing import Tuple

from qdrant_client import QdrantClient


class VectorDB:
    def __init__(self, path: str):
        self.client = QdrantClient(path=path)
        self.db_name = 'foods'
        # TODO: do we want to use our own embedding model? 
        self.client.set_model(self.client.DEFAULT_EMBEDDING_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def add(self, dict_path: str):
        # Expecting a descriptor dictionary as
        # { 'food_item': ['description 1', 'description 2'], ... }

        with open(dict_path, 'r', encoding='utf-8') as f:
            descriptors = json.load(f)

        metadata = []
        docs = []
        for k, v in descriptors.items():
            for descriptor in v:
                metadata.append({"class": k})
                docs.append(descriptor)
        ids = range(len(docs))
        self.client.add(
            collection_name=self.db_name,
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    # Given a food image descriptor, return the most probable class and similarity score
    def query(self, query_text: str) -> Tuple[str, float]:
        search_result = self.client.query(
            collection_name=self.db_name,
            query_text=query_text
        )
        # Take top-1 result
        category = search_result[0].metadata['class']
        confidence = search_result[0].score
        return category, confidence


