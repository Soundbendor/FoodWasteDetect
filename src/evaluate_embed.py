import argparse
import configparser

import pandas as pd
from qdrant_client.models import ScoredPoint

from database.embedding import EmbeddingModel
from database.vecdb import VectorDB
from ds.foodx251 import FoodX251
from vlm.intern import InternVLM

# goal:
# for every image in foodx-251
# read class label, image

prompt = "<image>\nPlease describe the food item in this image in a single sentence, focusing on the visual characteristics of the food."


class EvalMetric:
    def __init__(self):
        self.metrics = ["top1", "top5", "voting"]
        self.scores = pd.Series([0, 0, 0], index=self.metrics)
        self.len = 0

    def update_scores(self, q_vecs: list[ScoredPoint], db: VectorDB, label: str):
        self.scores.add([db.score(q_vecs, label, strat)[1] for strat in self.metrics])
        self.len += 1

    def compute_accuracies(self):
        return self.scores / self.len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument("config_file", help="Path to experiment config")
    return parser.parse_args()


def main():
    # Step 1: parse model config
    # Step 2: set up model, ds, db, etc.
    args = parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config_file)
    ds_path = cfg.get("Models", "ds_path")
    model = InternVLM(cfg.get("Models", "intern_path"))
    dataset = FoodX251(ds_path)
    embedder = EmbeddingModel(
        cfg.get("Models", "dictionary"), cfg.get("Models", "embed_model_save_path")
    )
    db = VectorDB(
        cfg.get("Models", "db_path"), embedder, cfg.get("Models", "collection_name")
    )

    val_set = dataset.val_set()
    metric = EvalMetric()

    # TODO: Turn this into a .apply() function
    for i, row in val_set.iterrows():
        response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
        q_vecs = db.query(response)
        # INFO: We use voting here as default
        score, confidence, prediction = db.score(q_vecs, row["class"], "voting")
        top5_score, _, _ = db.score(q_vecs, row["class"], "top5")
        metric.update_scores(q_vecs, db, row["class"])
        print(f"Predicted Label: {prediction}")
        print(f"Top 5 Score: {top5_score}")
        print(f"Description: {response}")
        print(f"True label: {row['class']}")

    scores = metric.compute_accuracies()
    print(scores)
