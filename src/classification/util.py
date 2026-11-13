"""Shared package utilities: the dataset registry by configuration name."""

from classification.datasets.base import FoodImageDataset
from classification.datasets.food101 import Food101ImageDataset
from classification.datasets.foodx251 import FoodX251ImageDataset
from classification.datasets.isia500 import ISIAFood500ImageDataset
from classification.datasets.uecfood256 import UECFood256ImageDataset
from classification.datasets.vegfru import VegFruImageDataset
from classification.experiments.config import DatasetConfig

DATASETS: dict[str, type[FoodImageDataset]] = {
    c.name: c
    for c in (
        FoodX251ImageDataset,
        Food101ImageDataset,
        UECFood256ImageDataset,
        VegFruImageDataset,
        ISIAFood500ImageDataset,
    )
}


def load_dataset(config: DatasetConfig) -> FoodImageDataset:
    """Load metadata for the configured dataset from ``config.root``."""
    return DATASETS[config.name](config.root)
