import pandas as pd

from .dataset import Dataset


class FwTest(Dataset):
    def __init__(self, root: str):
        self.name = "fw-test"
        self.test_path = "test"
        self.train_path = ""
        self.root = root
        self.cmap = self.get_class_labels()

    def get_class_labels(self) -> pd.Series:
        pass

    def test_set(self):
        return self._load_df("test.csv")
