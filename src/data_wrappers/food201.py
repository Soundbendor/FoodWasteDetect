import os

import pandas as pd

from .dataset import Dataset


class Food201(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.cmap = self.get_class_list()

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

    #
    # INFO: untested
    def _build_df(self, split: str):
        """
        Build a dataframe containing image paths for a dataset subset.
        should have [img, class_id, mask, segment, box]
        """
        # Go to either train or test
        path = os.path.join(self.root, split)
        dirs = os.listdir(path)
        df = pd.DataFrame(columns=dirs)
        for dir in os.listdir(path):
            df[dir] = os.listdir(os.path.join(path, dir))
        df.to_csv(f"{split}.csv", index=False)

    def _load_df(self, fname: str):
        # load csv. if it doesn't exist, create it
        csv_pth = os.path.join(self.root, fname)
        if not os.path.isfile(csv_pth):
            # build test set csv
            subset = os.path.splitext(fname)[0]
            self._build_df(subset)
        return pd.read_csv(csv_pth)

    def test_set(self):
        self._load_df("test.csv")

    def train_set(self):
        # load train.csv
        # if it doesn't exist, create it
        self._load_df("train.csv")
