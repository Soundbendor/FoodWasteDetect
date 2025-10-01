import json
import os

import pandas as pd

from .dataset import SegDataset


class FoodSeg103(SegDataset):
    def __init__(self, root: str):
        # where self.root = ".../food_datasets/foodseg103/data
        self.root = root
        self.cmap = self.get_class_labels()

    # INFO: And now, it's a dict return type, overriding List[str]?
    # WARN: Will break Dataset.get_patches()
    def get_class_labels(self) -> dict:
        # Load id2label.json
        with open(os.path.join(self.root, "id2label.json"), "r") as label_map:
            labels = json.load(label_map)
        return labels

    def val_set(self) -> pd.DataFrame:
        return self._load_df("validation.csv")

    def train_set(self) -> pd.DataFrame:
        return self._load_df("train.csv")
