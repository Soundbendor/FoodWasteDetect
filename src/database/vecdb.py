import logging
import os
from collections import Counter
from typing import List, Optional, Union

import requests
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, PointStruct, ScoredPoint,
                                  VectorParams)
from sentence_transformers import CrossEncoder, SentenceTransformer, util


class VectorDB:
    def __init__(
        self, path: str, db_name: str, reranker: Union[str, None], embed_size: int
    ):
        self.client = self.connect(addr=path)
        self.db_name: str = db_name
        self.embed_size = embed_size
        if reranker:
            self.reranker = CrossEncoder(reranker)
        else:
            self.reranker = None

    def point_exists(self, id: int) -> bool:
        return self.client.retrieve(
            collection_name=self.db_name,
            ids=[int(id)],
            with_payload=False,
            with_vectors=False,
        )

    def connect(self, addr: str):
        """Check for qdrant server running on host. If connection fails, starts a Qdrant instance."""
        try:
            response = requests.get(f"http://{addr}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Failure to contact Qdrant server, please start vector database server from startup script in sbatch"
            )
        return QdrantClient(addr, timeout=9999)

    def make_collection(self):
        if not self.client.collection_exists(self.db_name):
            self.client.create_collection(
                collection_name=self.db_name,
                vectors_config=VectorParams(
                    size=self.embed_size, distance=Distance.COSINE
                ),
            )

    """
        Take embeddings for data points for a given food class and insert them into the database.
        Optionally provide metadata to store in vector payload
        Metadata format: { 'key': [description1, description2, ...] }
        Metadata lists must have same length as embeddings.
    """

    def add_records(
        self,
        food_type: list[str],
        embeddings: list,
        metadata: Optional[dict],
        ids: Optional[list[int]],
    ):
        # Make sure the collection exists
        self.make_collection()
        # Check to ensure metadata length matches embedding length
        if metadata:
            for k, v in metadata.items():
                if len(v) != len(embeddings):
                    raise Exception("Incompatible metadata in vector database insert.")

        # Insert vectors into database
        points = []
        for idx, vector in enumerate(embeddings):
            payload = {"class": food_type[idx]}
            if metadata:
                for k, v in metadata.items():
                    payload[k] = v.iloc[idx]
            if not ids:
                img_num = int(os.path.splitext(payload["img_path"])[0].split("_")[1])
            else:
                img_num = ids[idx]
            points.append(PointStruct(id=img_num, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.db_name, points=points)

    # Given a food image descriptor, return the most probable class and similarity score
    # INFO: Query text must be None when using image search
    def query(
        self, query_text: Union[None, str], query_vec: torch.Tensor
    ) -> List[ScoredPoint]:
        top10 = self.client.query_points(
            collection_name=self.db_name, query=query_vec, limit=10
        )
        if self.reranker:
            rerank_pairs = [[query_text, doc.payload["description"]] for doc in top10]
            rerank_scores = self.reranker.predict(rerank_pairs)
            # append rerank scores to query vectors
            for idx in range(len(rerank_scores)):
                top10[idx].score = rerank_scores[idx]
            logging.info(rerank_pairs)
        return sorted(top10, key=lambda x: x.score, reverse=True)

    def vote_classification(self, search_result: list[ScoredPoint]):
        votes = Counter([x.payload["class"] for x in search_result])  # type: ignore
        candidate = votes.most_common(1)[0]
        category = candidate[0]
        top5 = [x.payload["class"].strip() for x in search_result[:5]]
        return category, top5

    # Returns (Accuracy, Similarity) where acc is binary 1-0
    def score(
        self, search_result: list[ScoredPoint], label: str, strategy: str
    ) -> tuple[int, float, str]:
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
            score = 1 if label.strip() in [x.payload["class"].strip() for x in search_result[:5]] else 0  # type: ignore
            class_list = [x.payload["class"].strip() for x in search_result[:5]]
            logging.debug(f"{class_list}")
            confidence = 0
            category = None
        return score, confidence, category
