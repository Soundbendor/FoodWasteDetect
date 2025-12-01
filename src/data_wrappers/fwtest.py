import os

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from .dataset import SegDataset


class FwTest(SegDataset):
    def __init__(self, root: str):
        self.name = "fw-test"
        self.test_path = "test"
        self.train_path = ""
        self.root = root
        self.cmap = self.get_class_labels()

    def get_class_labels(self) -> pd.Series:
        # TODO: Load dataset yml from roboflow
        with open(os.path.join(self.root, "data.yaml"), "r") as file:
            ds_info = yaml.load(file, Loader=yaml.Loader)
        return pd.Series(ds_info["names"])

    def train_set(self):
        return self._load_df("train.csv")
