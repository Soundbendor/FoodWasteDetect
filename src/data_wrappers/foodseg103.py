import json
import os

import pandas as pd

from .dataset import SegDataset


class FoodSeg103(SegDataset):
    def __init__(self, root: str):
        # where self.root = ".../food_datasets/foodseg103/data
        self.name = "foodseg103"
        self.test_path = "test"
        self.train_path = "train"
        self.root = root
        self.cmap = self.get_class_labels()

    def get_class_labels(self) -> dict:
        # Load id2label.json
        with open(os.path.join(self.root, "id2label.json"), "r") as label_map:
            labels = json.load(label_map)
        data = {"id": [int(k) for k in labels.keys()], "label": labels.values()}
        return pd.Series(data)

    def test_set(self) -> pd.DataFrame:
        return self._load_df("test.csv")

    def train_set(self) -> pd.DataFrame:
        return self._load_df("train.csv")
