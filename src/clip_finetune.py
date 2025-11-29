import os
import random

import pandas as pd
from PIL import Image
from sentence_transformers import (InputExample, SentenceTransformer, losses,
                                   util)
from torch.utils.data import DataLoader

from data_wrappers.foodx251 import FoodX251
from util import parse_args, parse_cfg


# Load FoodX251 dataset
def load_foodx251() -> tuple[str, pd.DataFrame]:
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(root=ds_path)
    return ds_path, ds.train_set()


ds_path, model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)
foodx251_train = load_foodx251()
# Convert into Huggingface dataset
foodx251_train["fpath"] = foodx251_train["fname"].apply(
    lambda x: os.path.join(ds_path, "train", "train_set", x)
)
# foodx251_train["images"] = foodx251_train["fpath"].apply(lambda x: Image.open(x))
print(food201)

# load model
train_dataset = []
for idx, row in foodx251_train.iterrows():
    # TODO: fix label?
    img = Image.open(row["fpath"])
    caption = "An image of a " + row["class"]
    train_dataset.append(InputExample(texts=[img, caption], label=1))
    # Append five negative caption pairs
    for i in range(5):
        neg_caption = foodx251_train["class"][random.randint(0, len(foodx251_train))]
        train_dataset.append(InputExample(texts=[img, neg_caption], label=0))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
train_loss = losses.ContrastiveLoss(model=model)
