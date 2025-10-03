import os

import pandas as pd

from .dataset import SegDataset


class Food201(SegDataset):
    def __init__(self, root: str):
        self.name="food201"
        self.root = root
        self.cmap = self.get_class_labels()

    def get_class_labels(self) -> list[str]:
        """Get unindexed list of class labels."""
        return self.get_class_list()["label"].tolist()

    def get_class_list(self) -> pd.Series:
        """Return list of class names as indexed series."""
        class_label = pd.read_csv(
            os.path.join(self.root, "food201", "pixel_annotations_map.csv"),
            names=["id", "label"],
        )
        class_label["id"] = class_label["id"] - 1
        class_label = class_label.set_index("id")
        # impute missing class labels
        missing_values = set(class_label.index).symmetric_difference(set(range(208)))
        for idx in missing_values:
            class_label.loc[idx] = "Unknown"
        class_label = class_label.sort_index()
        class_label.to_csv("food201_class_labels.csv")
        return class_label

    def test_set(self):
        self._load_df("test.csv")

    def train_set(self):
        self._load_df("train.csv")
