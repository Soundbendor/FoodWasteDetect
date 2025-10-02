import json
import os

import pandas as pd

from .dataset import SegDataset


class UECFoodPix(SegDataset):
    def __init__(self, root: str):
        # where self.root = ".../food_datasets/uecfoodpix/UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE"
        self.root = root
        self.cmap = self.get_class_labels()

    # INFO: And now, it's a dict return type, overriding List[str]?
    # WARN: Will break get_patches.
    def get_class_labels(self) -> dict:
        # Load category.txt
        df = pd.read_csv(os.path.join(self.root, "category.txt"), sep="\t", index_col = "id")
        return df.to_dict()["name"]

    def val_set(self) -> pd.DataFrame:
        return self._load_df("validation.csv")

    def train_set(self) -> pd.DataFrame:
        return self._load_df("train.csv")
