"""Whole-image dataset contracts, independent of crop/detection methods."""
from abc import ABC, abstractmethod
import hashlib
from pathlib import Path


def content_hash(path):
    try:
        with Path(path).open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


class ImageDataset(ABC):
    @abstractmethod
    def get_class_list(self): ...

    @abstractmethod
    def train_set(self): ...

    @abstractmethod
    def test_set(self): ...


class FoodX251ImageDataset(ImageDataset):
    def __init__(self, root):
        from data_wrappers.foodx251 import FoodX251
        self.legacy = FoodX251(str(Path(root).expanduser().resolve()))
        self.classes = list(self.legacy.get_class_list())
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError('Taxonomy must contain unique canonical labels')
        self.train = self._table(self.legacy.train_set(), self.legacy.train_path, 'train')
        self.test = self._table(self.legacy.val_set(), self.legacy.val_path, 'val')
        if set(self.train.fname) & set(self.test.fname):
            raise ValueError('Train/evaluation resolved-path overlap')
        train_hashes = set(self.train.sha256.dropna())
        if train_hashes & set(self.test.sha256.dropna()):
            raise ValueError('Train/evaluation content-hash overlap')

    def _table(self, table, directory, split):
        table = table.copy(deep=True)
        if not set(table['class']).issubset(self.classes):
            raise ValueError('Unknown class label')
        base = Path(self.legacy.root) / directory
        paths = [(base / str(name)).resolve() for name in table.fname]
        if any(not path.is_relative_to(base.resolve()) for path in paths):
            raise ValueError('Image path escapes its official split')
        table['source_id'] = [f'foodx251:{split}:{path.relative_to(base.resolve())}' for path in paths]
        if table.source_id.duplicated().any():
            raise ValueError('Duplicate source identities')
        table['fname'] = [str(path) for path in paths]
        table['source_split'] = split
        table['sha256'] = [content_hash(path) for path in paths]
        return table

    def get_class_list(self):
        return list(self.classes)

    def train_set(self):
        return self.train.copy(deep=True)

    def test_set(self):
        return self.test.copy(deep=True)


def load_dataset(config):
    if config['name'].lower() != 'foodx251':
        raise ValueError('Only foodx251 is supported')
    return FoodX251ImageDataset(config['root'])
