import os

import pandas as pd
from PIL import Image
from sentence_transformers import (InputExample, SentenceTransformer, losses,
                                   util)
from torch.utils.data import DataLoader

from data_wrappers.FoodX251 import FoodX251
from util import parse_args, parse_cfg


# Load FoodX251 dataset
def load_foodx251() -> tuple[str, pd.DataFrame]:
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(root=ds_path)
    return ds_path, ds.train_set()


ds_path, model = SentenceTransformer("jinaai/jina-clip-v2")
food201_train = load_foodx251()
# Convert into Huggingface dataset
food201_train["fpath"] = food201_train["fname"].apply(
    lambda x: os.path.join(ds_path, "train", "train_set", x)
)
food201_train["images"] = food201_train["fpath"].apply(lambda x: Image.open(x))
print(food201)
