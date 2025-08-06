import logging
import os
from itertools import chain

import numpy as np
from ultralytics import YOLOWorld

from data_wrappers.food201 import Food201
from data_wrappers.foodx251 import FoodX251
from database.clip_embedding import CLIPEmbedding
from database.img_crop import ImageCropper
from database.vecdb import VectorDB
from util import EvalMetric, parse_args, parse_cfg


# Initialize a YOLO-World
# TODO: evaluation metric
def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(ds_path)
    model = YOLOWorld(
        "yolov8x-worldv2.pt"
    )  # or select yolov8m/l-world.pt for different sizes
    model.set_classes(ds.cmap["label"])

    val_set = ds.val_set()
    metric = EvalMetric()

    n_samples = int(cfg["settings"]["eval_samples"])
    if n_samples != 0:
        val_set = val_set.sample(n=n_samples, random_state=42)

    acc = 0
    for i, row in val_set.iterrows():
        results = model.predict(f"{ds_path}/val/val_set/{row['fname']}")
        preds = list(chain.from_iterable([x.boxes.cls.tolist() for x in results]))
        preds = [int(x) for x in preds]
        ground_truth = ds.cmap.index[ds.cmap["label"] == row["class"]].tolist()[0]
        if ground_truth in preds:
            acc += 1
        logging.info(results)
        logging.info(f"Ground Truth Label: {row['class']}")
        logging.info(f"Predicted Labels: {[ds.cmap.iloc[x] for x in preds]}")
    print(f"ACCURACY: {acc / n_samples}")


def eval_food201():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    # WARN: hard-coded file
    ds_path = "/nfs/guille/eecs_research/soundbendor/beerya/food_datasets/food201/data/"
    ds = Food201(root=ds_path)
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(ds.get_class_labels())
    results = model.val(data=os.path.join(ds_path, "test"))
    print(results.box.map)
    print(results.results_dict)


def build_patch_db():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    # WARN: hard-coded file
    ds_path = cfg["paths"]["dataset"]
    ds = Food201(root=ds_path)
    ds.crop_patches("train")
    patches_df = ds.get_patches("train")

    # cropper = ImageCropper(dataset=ds, model="yolo11x.pt")
    # cropper.crop_dataset()
    # load vector database
    embedder = CLIPEmbedding(
        ds, cfg["embed_model"], cfg["paths"]["embed_model_save_path"], cfg["embed_size"]
    )
    db = VectorDB(
        cfg["qdrant_url"],
        cfg["collection_name"],
        cfg["reranker_model"],
        cfg["embed_size"],
    )

    batches = np.array_split(patches_df, 10000)
    for patches in batches:
        patch_pths = [
            os.path.join(ds_path, "train", "patches", x) for x in patches["patch_name"]
        ]
        vectors = embedder.get_embedding(patch_pths)
        metadata = {
            "img_path": patch_pths,
            "src_img": patches["src_img"],
        }
        db.add_records(patches["class"], vectors, metadata)


if __name__ == "__main__":
    build_patch_db()
