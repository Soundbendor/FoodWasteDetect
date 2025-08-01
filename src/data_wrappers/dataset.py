from typing import List

import pandas as pd


class Dataset:
    """Template class for datasets"""

    def __init__(self):
        self.train_path = ""
        self.test_path = ""
        self.val_path = ""
        self.root = ""

    def get_class_list(self) -> List[str]:
        return NotImplementedError  # type: ignore

    def test_set(self) -> pd.DataFrame:
        return NotImplementedError

    def train_set(self) -> pd.DataFrame:
        return NotImplementedError

    def val_set(self) -> pd.DataFrame:
        return NotImplementedError

    def get_subsets(self) -> list[tuple[str, pd.DataFrame]]:
        return list(
            zip(
                [self.train_path, self.test_path, self.val_path],
                [self.train_set, self.test_set, self.val_set],
            )
        )
