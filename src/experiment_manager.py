import logging
import os

import numpy as np
import pandas as pd
from PIL import Image

from data_wrappers.dataset import Dataset
from database.clip_embedding import CLIPEmbedding
from database.vecdb import VectorDB
from util import EvalMetric


class ExperimentManager:
    def __init__(self, cfg: dict):
        self.embedder = CLIPEmbedding(
            cfg["embed_model"],
            cfg["paths"]["embed_model_save_path"],
            cfg["embed_size"],
        )
        self.db = VectorDB(
            cfg["qdrant_url"],
            cfg["collection_name"],
            cfg["reranker_model"],
            cfg["embed_size"],
        )
        self.BATCH_SIZE = 50

    def _check_img(self, img: Image) -> bool:
        return all(i >= 20 for i in img.size)

    def load_imgs(self, df: pd.DataFrame) -> pd.DataFrame:
        # load all the images
        df["img_files"] = df["patch_pths"].apply(Image.open)
        valid_imgs = df["img_files"].apply(self._check_img)
        return df[valid_imgs]

    def add_label_vectors(self, ds: Dataset, start_idx: int):
        # Get class labels from dataset
        labels = ds.cmap
        vectors = self.embedder.get_text_embedding(["An image of " + x for x in labels])
        metadata = {"label": labels}
        ids = list(range(start_idx, start_idx + len(labels)))
        self.db.add_records(labels, vectors, metadata, ids)
        return ids[-1]

    def update_vecdb(self, ds: Dataset, subset: str, start_idx: int) -> int:
        # Load patches for dataset
        patch_df = ds.get_patches(subset, False)
        # Update index to start from desired database ID position
        patch_df.index = patch_df.index.to_numpy() + start_idx
        # Split dataset into batches
        batches = np.array_split(patch_df, self.BATCH_SIZE)
        # Confirm that database exists
        self.db.make_collection()
        for patches in batches:
            if self.db.point_exists(patches.index[0]):
                print("WARN: ID already exists, skipping...")
                continue
            patches["patch_pths"] = [
                os.path.join(ds.root, subset, "patches", x)
                for x in patches["patch_name"]
            ]
            # Filters out invalid images, returns updated dataframe
            patches = self.load_imgs(patches)
            vectors = self.embedder.get_embedding_from_preloaded(
                list(patches["img_files"])
            )
            metadata = {
                "img_path": pd.Series(patches["patch_pths"]),
                "src_img": patches["src_img"].astype(object),
            }

            self.db.add_records(
                list(patches["class"]),
                vectors,
                metadata,
                list(map(int, patches.index)),
            )
        # For subsequent calls, return the max ID value placed in dataset
        return patch_df.index[-1]

    def evaluate_model(self, eval_set: pd.DataFrame) -> EvalMetric:
        metric = EvalMetric()
        for i, row in eval_set.iterrows():
            try:
                query_vec = self.embedder.get_embedding([row["patch_pth"]])[0]
            except Exception as e:
                continue
            candidate_vecs = self.db.query(None, query_vec)
            score, confidence, prediction = self.db.score(
                candidate_vecs, row["class"], "voting"
            )
            top5_score, _, _ = self.db.score(candidate_vecs, row["class"], "top5")
            metric.update_scores(candidate_vecs, self.db, row["class"])
            logging.info(f"Predicted Label: {prediction}")
            logging.info(f"True label: {row['class']}")
            logging.info(f"In Top 5? {top5_score}")
            logging.info(f"Current Accuracies: {metric.compute_accuracies()}")

        scores = metric.compute_accuracies()
        print(scores)
        return metric
