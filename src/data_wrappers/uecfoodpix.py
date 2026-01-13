import json
import os

import pandas as pd

from .dataset import SegDataset


class UECFoodPix(SegDataset):
    def __init__(self, root: str):
        # where self.root = ".../food_datasets/uecfoodpix/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE"
        self.name = "uecfoodpix"
        self.test_path = "test"
        self.train_path = "train"
        self.root = root
        self.cmap = self.get_class_labels()

    def get_class_labels(self) -> pd.Series:
        # Load category.txt
        df = pd.read_csv(
            os.path.join(self.root, "category.txt"), sep="\t", index_col="id"
        )
        return df["name"]

    # TODO: Should deprecate this interface, prefer directly calling load_df
    def test_set(self) -> pd.DataFrame:
        return self._load_df("test.csv")

    # TODO: Should deprecate this interface, prefer directly calling load_df
    def train_set(self) -> pd.DataFrame:
        return self._load_df("train.csv")
