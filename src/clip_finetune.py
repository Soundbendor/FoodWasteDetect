import os
import random

import pandas as pd
from PIL import Image
from sentence_transformers import (InputExample, SentenceTransformer, losses,
                                   util)
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_wrappers.food201 import Food201
from data_wrappers.foodseg103 import FoodSeg103
from data_wrappers.foodx251 import FoodX251
from data_wrappers.uecfoodpix import UECFoodPix
from util import parse_args, parse_cfg


# Load FoodX251 dataset
def load_foodx251() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(root=ds_path)
    return ds_path, ds.train_set(), ds.val_set()


def load_combined() -> tuple[pd.DataFrame, pd.DataFrame]:
    args = parse_args()
    cfg = parse_cfg(args.config_file)

    # Load all 3 datasets
    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)

    datasets = [food201, uecfoodpix, foodseg103]

    train_set = []
    for ds in datasets:
        df = ds.get_patches("train", False)
        # Invert classmap to work as {name: id}
        cmap = pd.Series(ds.cmap.index.values, index=ds.cmap)
        df["dataset"] = ds.name
        df["src_img"] = df.apply(lambda x: ds.name + "_" + str(x["src_img"]), axis=1)
        df["class_id"] = df.apply(lambda x: cmap[x["class"]], axis=1)
        train_set.append(df)
    train_df = pd.concat(train_set, ignore_index=True)

    test_set = []
    for ds in datasets:
        df = ds.get_patches("test", False)
        # Invert classmap to work as {name: id}
        cmap = pd.Series(ds.cmap.index.values, index=ds.cmap)
        df["dataset"] = ds.name
        df["src_img"] = df.apply(lambda x: ds.name + "_" + str(x["src_img"]), axis=1)
        df["class_id"] = df.apply(lambda x: cmap[x["class"]], axis=1)
        test_set.append(df)
    test_df = pd.concat(test_set, ignore_index=True)
    return train_df, test_df


def finetune_foodx251():
    model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)
    ds_path, foodx251_train, foodx251_test = load_foodx251()
    # Convert into Huggingface dataset
    foodx251_train["fpath"] = foodx251_train["fname"].apply(
        lambda x: os.path.join(ds_path, "train", "train_set", x)
    )
    foodx251_test["fpath"] = (
        foodx251_test["fname"]
        .apply(lambda x: os.path.join(ds_path, "val", "val_set", x))
        .sample(frac=1)
    )
    # foodx251_train["images"] = foodx251_train["fpath"].apply(lambda x: Image.open(x))
    print(foodx251_train)

    # load model
    train_dataset = []
    for idx, row in tqdm(foodx251_train.iterrows(), total=len(foodx251_train)):
        img = Image.open(row["fpath"])
        basecap = "An image of "
        caption = basecap + row["class"]
        neg_caption = (
            basecap
            + foodx251_train["class"][random.randint(0, len(foodx251_train) - 1)]
        )
        train_dataset.append(InputExample(texts=[img, caption], label=1))
        train_dataset.append(InputExample(texts=[img, neg_caption], label=0))

    eval_imgs = []
    eval_txts = []
    eval_labels = []
    print("Building evaluation dataset...")
    for idx, row in tqdm(foodx251_test.iterrows(), total=len(foodx251_test)):
        img = Image.open(row["fpath"])
        basecap = "An image of "
        caption = basecap + row["class"]
        neg_caption = (
            basecap + foodx251_test["class"][random.randint(0, len(foodx251_test) - 1)]
        )
        # Add positive sample
        eval_imgs.append(img)
        eval_txts.append(caption)
        eval_labels.append(1)
        # Add negative sample
        eval_imgs.append(img)
        eval_txts.append(neg_caption)
        eval_labels.append(0)
        if idx > 1000:
            break

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    dev_evaluator = BinaryClassificationEvaluator(
        sentences1=eval_imgs,
        sentences2=eval_txts,
        labels=eval_labels,
        name="foodx251-val",
    )
    print(train_dataloader)
    train_loss = losses.ContrastiveLoss(model=model)
    model.fit(
        [(train_dataloader, train_loss)],
        epochs=5,
        evaluator=dev_evaluator,
        output_path="clip_train_best",
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_path="clip_train",
        checkpoint_save_steps=500,
    )


def finetune_combined():
    model = SentenceTransformer("jinaai/jina-clip-v2", trust_remote_code=True)
    df_train, df_test = load_combined()
    # Convert into Huggingface dataset
    # use patch_pth
    df_test = df_test.sample(frac=1)
    print(df_train)

    # load model
    train_dataset = []
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train)):
        img = Image.open(row["patch_pth"])
        basecap = "An image of "
        caption = basecap + row["class"]
        neg_caption = basecap + df_train["class"][random.randint(0, len(df_train) - 1)]
        train_dataset.append(InputExample(texts=[img, caption], label=1))
        train_dataset.append(InputExample(texts=[img, neg_caption], label=0))

    eval_imgs = []
    eval_txts = []
    eval_labels = []
    print("Building evaluation dataset...")
    for idx, row in tqdm(df_test.iterrows(), total=1000):
        img = Image.open(row["fpath"])
        basecap = "An image of "
        caption = basecap + row["class"]
        neg_caption = basecap + df_test["class"][random.randint(0, len(df_test) - 1)]
        # Add positive sample
        eval_imgs.append(img)
        eval_txts.append(caption)
        eval_labels.append(1)
        # Add negative sample
        eval_imgs.append(img)
        eval_txts.append(neg_caption)
        eval_labels.append(0)
        if idx > 1000:
            break

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    dev_evaluator = BinaryClassificationEvaluator(
        sentences1=eval_imgs,
        sentences2=eval_txts,
        labels=eval_labels,
        name="foodbb-val",
    )
    print(train_dataloader)
    train_loss = losses.ContrastiveLoss(model=model)
    model.fit(
        [(train_dataloader, train_loss)],
        epochs=5,
        evaluator=dev_evaluator,
        output_path="foodbb_clip_train_best",
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_path="foodbb_clip_train",
        checkpoint_save_steps=500,
    )


if __name__ == "__main__":
    finetune_foodx251()
