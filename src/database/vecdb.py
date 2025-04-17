import json
import subprocess
import time
from collections import Counter
from typing import List, Tuple

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, PointStruct, ScoredPoint,
                                  VectorParams)

from .embedding import EmbeddingModel


class VectorDB:
    def __init__(self, path: str, model: EmbeddingModel, db_name: str):
        # TODO: server startup
        # check if cn-m-1.hpc.engr.oregonstate.edu:6443 is open
        # if not, run startup script?
        self.client = self.connect(addr=path)
        self.db_name = db_name
        self.model = model

    def connect(self, addr: str):
        """Check for qdrant server running on host. If connection fails, starts a Qdrant instance."""
        connection_status = 0
        for i in range(5):
            try:
                response = requests.get(f"http://{addr}")
            except requests.exceptions.ConnectionError:
                connection_status = -1
            if connection_status < 0 or response.status_code != 200:
                # call qdrant startup script on first retry
                if i == 0:
                    proc = subprocess.Popen(
                        "sbatch src/database/start_qdrant.sbatch",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    out, error = proc.communicate()
                    print(f"QDRANT ALERT: {out}")
                    print(f"QDRANT ERROR: {error}")
                # clearly, server is starting, but we're still unable to connect
                # TODO: print request errors here
                print("DEBUG: Waiting 60s, contacting Qdrant server...")
                time.sleep(60)
            else:
                break
        else:
            raise ConnectionError(
                "Failure to contact Qdrant server, likely due to excess queue times on cn-m-1."
            )
        return QdrantClient(addr)

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
            # WARN: this is only configured for MPnet
            # Other embedding models will fail
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
    ) -> Tuple[int, float, str]:
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
            print(f"DEBUG:{class_list}")
            print(f"DEBUG: {search_result}")
            confidence = 0
            category = None
        return score, confidence, category
