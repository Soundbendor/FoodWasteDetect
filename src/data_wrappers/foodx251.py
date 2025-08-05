import os
from typing import List

import pandas as pd

from .dataset import Dataset

# root
# --- annot
# --- --- class_list.txt
# --- --- test_info.csv


class FoodX251(Dataset):
    def __init__(self, root: str) -> None:
        self.root = root
        self.train_path = os.path.join(root, "annot/train_info.csv")
        # WARN: test_path csv does not have class labels
        self.test_path = os.path.join(root, "annot/test_info.csv")
        self.val_path = os.path.join(root, "annot/val_info.csv")

    def get_class_list(self) -> List[str]:
        return self.cmap["label"]

    def _naturalize_class_labels(self, row: str) -> str:
        return row.capitalize().replace("_", " ")

    def _get_classmap(self) -> pd.DataFrame:
        cmap = pd.read_csv(
            os.path.join(self.root, "annot/class_list.txt"),
            delim_whitespace=True,
            index_col=0,
            names=["id", "label"],
        )
        cmap["label"] = cmap["label"].apply(self._naturalize_class_labels)
        return cmap

    def _map_class(self, row: str):
        return self.cmap.loc[int(row)]["label"]

    def _get_dataset(self, pth: str):
        data = pd.read_csv(pth, names=["fname", "class"])
        # substitute id numbers with class names
        data["class"] = data["class"].apply(self._map_class)
        return data

    # WARN: need to fix, csv is different
    def test_set(self) -> pd.DataFrame:
        return self._get_dataset(self.test_path)

    def val_set(self) -> pd.DataFrame:
        return self._get_dataset(self.val_path)

    def train_set(self) -> pd.DataFrame:
        return self._get_dataset(self.train_path)
