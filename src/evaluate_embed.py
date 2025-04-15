import configparser

import pandas as pd
from qdrant_client.models import ScoredPoint

from database.embedding import EmbeddingModel
from database.vecdb import VectorDB
from ds.foodx251 import FoodX251
from intern import InternVLM

# goal:
# for every image in foodx-251
# read class label, image

prompt = "<image>\nPlease describe the food item in this image in a single sentence, focusing on the visual characteristics of the food."
class EvalMetric:
    def __init__(self):
        self.metrics = ['top1', 'top5', 'voting']
        self.scores = pd.Series([0, 0, 0], index=self.metrics)
        self.len = 0

    def update_scores(self, q_vecs: list[ScoredPoint], db: VectorDB, label: str):
        self.scores.add([db.score(q_vecs, label, strat)[1] for strat in self.metrics])
        self.len += 1

    def compute_accuracies(self):
        return self.scores / self.len

def main(*args):
# Step 1: parse model config
# Step 2: set up model, ds, db, etc.
    config_file = args[0]
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    ds_path = cfg.get('Models', 'ds_path')
    model = InternVLM(cfg.get('Models', 'intern_path'))
    dataset = FoodX251(ds_path)
    embedder = EmbeddingModel(cfg.get('Models', 'dictionary'), cfg.get('Models', 'embed_model'))
    db = VectorDB(cfg.get('Models', 'db_path'), embedder)

    val_set = dataset.val_set()
    metric = EvalMetric()

    # TODO: Turn this into a .apply() function
    for i, row in val_set.iterrows():
        response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
        q_vecs = db.query(response)
        # INFO: We use voting here as default
        prediction, _ = db.score(q_vecs, row["class"], "voting")
        metric.update_scores(q_vecs, db, row['class'])
        print(f"Predicted Label: {prediction}")
        print(f"Description: {response}")
        print(f"True label: {row['class']}")
    
    scores = metric.compute_accuracies()
    print(scores)
