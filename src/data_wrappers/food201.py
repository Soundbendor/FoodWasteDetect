import os

import pandas as pd

from .dataset import Dataset


class Food201(Dataset):
    def __init__(self, root: str):
        self.root = root

    def get_class_list(self) -> list[str]:
        return pd.read_csv(os.path.join(self.root, "food201", "pixel_annotations_map.csv"), names=['id', 'label'])

    # Build a dataframe containing image paths for a dataset subset.
    # should have [img, class_id, mask, segment, box]
    def _build_df(self):
        pass

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
