import logging
import os
import uuid
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

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
        self.BATCH_SIZE = 50

    def _check_img(self, img: Image) -> bool:
        return all(i >= 20 for i in img.size)

    def load_imgs(self, df: pd.DataFrame) -> pd.DataFrame:
        # load all the images
        df["img_files"] = df.apply(Image.open, df["patch_pths"])
        valid_imgs = df["img_files"].apply(self._check_img, axis=1)
        return df[valid_imgs]


    def update_vecdb(self, ds: Dataset, subset: str, start_idx: int) -> int:
        # Load patches for dataset
        patch_df = ds.get_patches(subset, False)
        # Update index to start from desired database ID position
        patch_df.index = patch_df.index.to_numpy() + start_idx
        # Split dataset into batches
        batches = np.array_split(patch_df, self.BATCH_SIZE)
        # Confirm that database exists
        self.db.make_collection()
        for patches in batches:
            if self.db.point_exists(patches.index[0]):
                print("WARN: ID already exists, skipping...")
                continue
            patches["patch_pths"] = [
                os.path.join(ds.root, subset, "patches", x)
                for x in patches["patch_name"]
            ]
            # Filters out invalid images, returns updated dataframe
            patches = self.load_imgs(patches)
            vectors = self.embedder.get_embedding_from_preloaded(patches["img_files"])
            metadata = {
                "img_path": pd.Series(patches["patch_pths"]),
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

    def evaluate_model(self, eval_set: pd.DataFrame) -> EvalMetric:
        metric = EvalMetric()
        for i, row in eval_set.iterrows():
            query_vec = self.embedder.get_embedding([row["patch_pth"]])[0]
            candidate_vecs = self.db.query(None, query_vec)
            score, confidence, prediction = self.db.score(
                candidate_vecs, row["class"], "voting"
            )
            top5_score, _, _ = self.db.score(candidate_vecs, row["class"], "top5")
            metric.update_scores(candidate_vecs, self.db, row["class"])
            logging.info(f"Predicted Label: {prediction}")
            logging.info(f"True label: {row['class']}")
            logging.info(f"In Top 5? {top5_score}")
            logging.info(f"Current Accuracies: {metric.compute_accuracies()}")

        scores = metric.compute_accuracies()
        print(scores)
        return metric


def finetune_yolov11():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    combined_pth = cfg["paths"]["combined"]
    model = YOLO("yolv11x")
    metrics = model.train(data=os.path.join(combined_pth, "dataset.yaml"))
    print(metrics.box.map)


def experiment_combined_dataset():

    args = parse_args()
    cfg = parse_cfg(args.config_file)
    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)

    datasets = [food201, uecfoodpix, foodseg103]

    # Step 2: Compute instance embeddings
    exp = ExperimentManager(cfg)

    start_id = 0
    for ds in [food201, uecfoodpix, foodseg103]:
        start_id = exp.update_vecdb(ds, "train", start_id)

    food201_test_patch = food201.get_patches("test", False)[:1000]
    uecfoodpix_test_patch = uecfoodpix.get_patches("test", False)[:1000]
    foodseg103_test_patch = foodseg103.get_patches("validation", False)[:1000]
    eval_set = pd.concat(
        [food201_test_patch, uecfoodpix_test_patch, foodseg103_test_patch],
        ignore_index=True,
    )
    exp.evaluate_model(eval_set)


def evaluate_pipeline():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    # Step 1: Load all 3 datasets

    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)

    datasets = [food201, uecfoodpix, foodseg103]

    # Step 2: Use fine-tuned YOLO to extract patches for each dataset

    # Pre-load validation subsets
    for ds in datasets:
        ds.test_set()
        ds.detect_patches(ds.test_path, ds.test_set, "best.pt")

    food201_dets = food201.get_patches("test", True)
    uecfoodpix_dets = uecfoodpix.get_patches("test", True)
    foodseg103_dets = foodseg103.get_patches("test", True)

    # Step 3: Predict det categories using CLIP

    exp = ExperimentManager(cfg)
    eval_set = pd.concat(
        [food201_dets, uecfoodpix_dets, foodseg103_dets],
        ignore_index=True,
    )
    print(eval_set)

    # Load patches.csv as well
    # Group by image name
    food201_test_patch = food201.get_patches("test", False)
    uecfoodpix_test_patch = uecfoodpix.get_patches("test", False)
    foodseg103_test_patch = foodseg103.get_patches("test", False)
    labels = pd.concat(
        [food201_test_patch, uecfoodpix_test_patch, foodseg103_test_patch],
        ignore_index=True,
    )
    print(labels)

    # Each groupby: Dataframe of patches for a given image
    eval_group = eval_set[:1000].groupby("source_img")
    # Each groupby:
    label_group = labels.groupby("src_img")
    # Load all known boxes for each image
    # For each patch for that image
    # Classify the patch using CLIP
    # If patch class exists in known boxes, count towards accuracy
    acc_score = 0
    total_patches = 1
    for img_name, dets_df in eval_group:
        label_boxes = label_group.get_group(os.path.splitext(img_name)[0])
        for _, row in dets_df.iterrows():
            # TODO: update patch patch to point to cropped-detections
            pth = row["patch_pth"]
            pth = pth.replace("patches", "cropped-detections")
            try:
                query_vec = exp.embedder.get_embedding([pth])[0]
            except FileNotFoundError:
                print("Error! File not found.")
                continue
            candidate_vecs = exp.db.query(None, query_vec)
            # todo: eval metric
            prediction = exp.db.vote_classification(candidate_vecs)
            print(f"DEBUG: {prediction}")
            # TODO: compute mAP@50
            # TODO: fix accuracy metric to improve accuracy
            if any(label_boxes["class"].str.contains(prediction)):
                acc_score += 1
            total_patches += 1
        print(acc_score / total_patches)
    print(acc_score / total_patches)


if __name__ == "__main__":
    experiment_combined_dataset()
