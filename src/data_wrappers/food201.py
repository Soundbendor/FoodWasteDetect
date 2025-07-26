import os

import numpy as np
import pandas as pd

from .dataset import Dataset


class Food201(Dataset):
    def __init__(self, root: str):
        self.root = root

    def get_class_list(self) -> list[str]:
        class_label = pd.read_csv(
            os.path.join(self.root, "food201", "pixel_annotations_map.csv"),
            names=["id", "label"],
        )
        class_label = class_label.set_index('id')
        # impute missing class labels
        missing_values = set(class_label.index).symmetric_difference(set(range(208)))
        for idx in missing_values:
            class_label.loc[idx] = "Unknown"
        class_label = class_label.sort_index()
        class_label.to_csv("food201_class_labels.csv")
        return class_label


    # Build a dataframe containing image paths for a dataset subset.
    # should have [img, class_id, mask, segment, box]
    def _build_df(self, split: str):
        # Go to either train or test
        path = os.path.join(self.root, split)
        dirs = os.listdir(path)
        df = pd.DataFrame(columns=dirs)
        for dir in os.listdir(path):
            df[dir] = os.listdir(os.path.join(path, dir))
        df.to_csv(f"{split}.csv", index=False)

    def _load_df(self, fname: str):
        # load test.csv
        # if it doesn't exist, create it
        csv_pth = os.path.join(self.root, fname)
        if not os.path.isfile(csv_pth):
            # build test set csv
            pass
        return pd.read_csv(csv_pth)

    def test_set(self):
        self._load_df("test.csv")

    def train_set(self):
        # load train.csv
        # if it doesn't exist, create it
        self._load_df("train.csv")
