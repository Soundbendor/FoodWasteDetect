import logging
import os
import uuid
from itertools import chain

import numpy as np
import pandas as pd
from ultralytics import YOLOWorld

from data_wrappers.dataset import Dataset
from data_wrappers.food201 import Food201
from data_wrappers.foodseg103 import FoodSeg103
from data_wrappers.foodx251 import FoodX251
from data_wrappers.uecfoodpix import UECFoodPix
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


def eval_obj_det():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = Food201(root=ds_path)
    ds.crop_patches("test")
    val_set = ds.get_patches("test")
    metric = EvalMetric()

    embedder = CLIPEmbedding(
        ds, cfg["embed_model"], cfg["paths"]["embed_model_save_path"], cfg["embed_size"]
    )
    db = VectorDB(
        cfg["qdrant_url"],
        cfg["collection_name"],
        cfg["reranker_model"],
        cfg["embed_size"],
    )

    for i, row in val_set.iterrows():
        eval_patch = os.path.join(ds.root, "test", "patches", row["patch_name"])
        query_vec = embedder.get_embedding([eval_patch])[0]
        candidate_vecs = db.query(None, query_vec)
        score, confidence, prediction = db.score(candidate_vecs, row["class"], "voting")
        top5_score, _, _ = db.score(candidate_vecs, row["class"], "top5")
        metric.update_scores(candidate_vecs, db, row["class"])
        logging.info(f"Predicted Label: {prediction}")
        logging.info(f"True label: {row['class']}")
        logging.info(f"In Top 5? {top5_score}")
        logging.info(f"Current Accuracies: {metric.compute_accuracies()}")

    scores = metric.compute_accuracies()
    print(scores)


# INFO: Current experiment 09/07
def insert_class_labels():
    """
    Given a set of class labels for a dataset, compute the embeddings
    for each label and insert this into our vector database.
    """
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(root=ds_path)
    # ds.detect_patches(ds.val_path, ds.val_set)

    # Establish connection to vector database

    clip_embedder = CLIPEmbedding(
        ds, cfg["embed_model"], cfg["paths"]["embed_model_save_path"], cfg["embed_size"]
    )
    db = VectorDB(
        cfg["qdrant_url"],
        cfg["collection_name"],
        cfg["reranker_model"],
        cfg["embed_size"],
    )

    labels = ds.get_class_list()
    label_embeddings = clip_embedder.get_text_embedding(labels)

    val_set = ds.get_patches("val", detections=True)
    metric = EvalMetric()

    # insert into database
    # db.add_records(labels, label_embeddings, metadata = None, ids = [uuid.uuid4() for _ in range(len(labels))])

    # search vector database using ds.test
    for i, row in val_set.iterrows():
        eval_patch = os.path.join(
            ds.root, "val", "cropped-detections", row["patch_name"]
        )
        query_vec = clip_embedder.get_embedding([eval_patch])[0]
        candidate_vecs = db.query(None, query_vec)
        # score based on if image class is detected at all
        score, confidence, prediction = db.score(candidate_vecs, row["label"], "voting")
        top5_score, _, _ = db.score(candidate_vecs, row["label"], "top5")
        metric.update_scores(candidate_vecs, db, row["label"])
        logging.info(f"Predicted Label: {prediction}")
        logging.info(f"True label: {row['label']}")
        logging.info(f"In Top 5? {top5_score}")
        logging.info(f"Current Accuracies: {metric.compute_accuracies()}")


def get_id(pth: str):
    basename = os.path.splitext(os.path.basename(pth))[0]
    return int(basename.split("-")[1].split("_")[0]) + int(basename.split("_")[-2])


def build_patch_db():

    args = parse_args()
    cfg = parse_cfg(args.config_file)
    # WARN: hard-coded file
    ds_path = cfg["paths"]["dataset"]
    ds = Food201(root=ds_path)
    # ds.crop_patches("train")
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

    batches = np.array_split(patches_df, 500)
    for patches in batches:
        patch_pths = [
            os.path.join(ds_path, "train", "patches", x) for x in patches["patch_name"]
        ]
        vectors = embedder.get_embedding(patch_pths)
        metadata = {
            "img_path": pd.Series(patch_pths),
            "src_img": patches["src_img"],
        }

        ids = [get_id(pth) for pth in patch_pths]
        db.add_records(list(patches["class"]), vectors, metadata, ids)


class ExperimentManager:
    def __init__(self, cfg: dict):
        self.embedder = CLIPEmbedding(
            cfg["embed_model"],
            cfg["paths"]["embed_model_save_path"],
            cfg["embed_size"],
        )
        self.db = VectorDB(
            cfg["qdrant_url"],
            cfg["collection_name"],
            cfg["reranker_model"],
            cfg["embed_size"],
        )
        self.BATCH_SIZE = 500

    def update_vecdb(self, ds: Dataset, subset: str, start_idx: int) -> int:
        # Load patches for dataset
        patch_df = ds.get_patches(subset, False)
        # Update index to start from desired database ID position
        patch_df.index = patch_df.index.to_numpy() + start_idx
        # Split dataset into batches
        batches = np.array_split(patch_df, self.BATCH_SIZE)
        for patches in batches:
            if self.db.point_exists(patches.index[0]):
                print("WARN: ID already exists, skipping...")
                continue
            patch_pths = [
                os.path.join(ds.root, subset, "patches", x)
                for x in patches["patch_name"]
            ]
            vectors = self.embedder.get_embedding(patch_pths)
            metadata = {
                "img_path": pd.Series(patch_pths),
                "src_img": patches["src_img"].astype(object),
            }

            self.db.add_records(
                list(patches["class"]),
                vectors,
                metadata,
                list(map(int, patches.index)),
            )
        # For subsequent calls, return the max ID value placed in dataset
        return patch_df.index[-1]


if __name__ == "__main__":
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)

    exp = ExperimentManager(cfg)

    start_id = 0
    for ds in [food201, uecfoodpix, foodseg103]:
        start_id = exp.update_vecdb(ds, "train", start_id)
