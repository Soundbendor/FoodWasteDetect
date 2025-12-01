import logging
import os
import uuid
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from ultralytics import YOLO

from data_wrappers.dataset import Dataset
from data_wrappers.food201 import Food201
from data_wrappers.foodseg103 import FoodSeg103
from data_wrappers.foodx251 import FoodX251
from data_wrappers.fwtest import FwTest
from data_wrappers.uecfoodpix import UECFoodPix
from database.clip_embedding import CLIPEmbedding
from database.img_crop import ImageCropper
from database.vecdb import VectorDB
from util import EvalMetric, ExperimentResult, parse_args, parse_cfg


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
        df["img_files"] = df["patch_pths"].apply(Image.open)
        valid_imgs = df["img_files"].apply(self._check_img)
        return df[valid_imgs]

    def add_label_vectors(self, ds: Dataset, start_idx: int):
        # Get class labels from dataset
        labels = ds.cmap
        vectors = self.embedder.get_text_embedding(labels)
        metadata = {"label": labels}
        ids = list(range(start_idx, start_idx + len(labels)))
        self.db.add_records(labels, vectors, metadata, ids)
        return ids[-1]

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
            vectors = self.embedder.get_embedding_from_preloaded(
                list(patches["img_files"])
            )
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
            try:
                query_vec = self.embedder.get_embedding([row["patch_pth"]])[0]
            except Exception as e:
                continue
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
    combined_pth = cfg["paths"]["dataset"]
    model = YOLO("best.pt")
    metrics = model.val(data=os.path.join(combined_pth, "dataset.yaml"))
    print(metrics.box.map)
    print(metrics)


def experiment_single_dataset():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    # ds_path = cfg["paths"]["foodseg103"]
    ds_path = cfg["paths"]["uecfoodpix"]
    # ds_path = cfg["paths"]["food201"]

    # ds = Food201(root=ds_path)
    ds = UECFoodPix(root=ds_path)
    # ds = FoodSeg103(root=ds_path)
    exp = ExperimentManager(cfg)

    # Step 1: Generate training embeddings

    eval_set = ds.get_patches("test", False)[:1000]

    # exp.update_vecdb(ds, "train", 0)
    exp.evaluate_model(eval_set)


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
        # start_id = exp.update_vecdb(ds, "train", start_id)
        pass

    food201_test_patch = food201.get_patches("test", False)[:1000]
    uecfoodpix_test_patch = uecfoodpix.get_patches("test", False)[:1000]
    foodseg103_test_patch = foodseg103.get_patches("test", False)[:1000]
    eval_set = pd.concat(
        [food201_test_patch, uecfoodpix_test_patch, foodseg103_test_patch],
        ignore_index=True,
    )
    exp.evaluate_model(eval_set)


def evaluate_pipeline():
    def update_top1(labels: list[str], preds: list[str]) -> int:
        c = 0
        for y in labels:
            for y_hat in preds:
                if y.strip().lower() == y_hat.strip().lower():
                    # Prediction matched once, remove from list
                    preds.remove(y_hat)
                    c += 1
                    break
        return c

    def update_top5(labels: list[str], preds: list[list[str]]) -> int:
        c = 0
        for y in labels:
            for y_hat in preds:
                y_hat_new = [x.strip().lower() for x in y_hat]
                if y.strip().lower() in y_hat_new:
                    # Prediction matched once, remove from list
                    preds.remove(y_hat)
                    c += 1
                    break
        return c

    args = parse_args()
    cfg = parse_cfg(args.config_file)
    # Step 1: Load all 3 datasets

    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)

    # TODO: here
    datasets = [food201, uecfoodpix, foodseg103]
    # datasets = [food201, uecfoodpix]

    # Step 2: Use fine-tuned YOLO to extract patches for each dataset

    exp = ExperimentManager(cfg)

    start_id = 0
    # INFO: Update vector database
    # for ds in [foodseg103]:
    # start_id = exp.update_vecdb(ds, "train", start_id)

    # Pre-load validation subsets
    # for ds in datasets:
    # ds.test_set()
    # ds.detect_patches(ds.test_path, ds.test_set, "best.pt")

    # TODO: here
    food201_dets = food201.get_patches("test", True)
    uecfoodpix_dets = uecfoodpix.get_patches("test", True)
    foodseg103_dets = foodseg103.get_patches("test", True)

    # Step 3: Predict det categories using CLIP

    # TODO: here
    eval_set = pd.concat(
        [uecfoodpix_dets, food201_dets, foodseg103_dets],
        ignore_index=True,
    )
    print(eval_set)

    # Load patches.csv as well
    # Group by image name
    food201_test_patch = food201.get_patches("test", False)
    uecfoodpix_test_patch = uecfoodpix.get_patches("test", False)
    foodseg103_test_patch = foodseg103.get_patches("test", False)
    # TODO: here
    labels = pd.concat(
        [uecfoodpix_test_patch, food201_test_patch, foodseg103_test_patch],
        ignore_index=True,
    )
    print(labels)

    # Each groupby: Dataframe of patches for a given image
    eval_set["source_img"] = eval_set["source_img"].astype(str)
    eval_group = eval_set.sample(frac=1)[:1000].groupby("source_img")
    # Each groupby:
    labels["src_img"] = labels["src_img"].astype(str)
    label_group = labels.groupby("src_img")
    print(label_group.groups.keys())
    # Load all known boxes for each image
    # For each patch for that image
    # Classify the patch using CLIP
    # If patch class exists in known boxes, count towards accuracy
    top5_acc_score = 0
    acc_score = 0
    total_precision = 1
    total_recall = 1
    for img_name, dets_df in eval_group:
        try:
            label_boxes = label_group.get_group(os.path.splitext(img_name)[0])
        except Exception as e:
            print("Key not found in label set")
            continue
        preds = []
        top5_preds = []
        for _, row in dets_df.iterrows():
            pth = row["patch_pth"]
            pth = pth.replace("patches", "cropped-detections")
            try:
                print(pth)
                query_vec = exp.embedder.get_embedding([pth])[0]
            except FileNotFoundError:
                print("Error! File not found.")
                continue
            candidate_vecs = exp.db.query(None, query_vec)
            # todo: eval metric
            prediction, top5_classes = exp.db.vote_classification(candidate_vecs)
            print(f"DEBUG: {prediction}")
            preds.append(prediction)
            top5_preds.append(top5_classes)
        # compute accuracy score for this image
        total_precision += len(dets_df)
        total_recall += len(label_boxes)
        acc_score += update_top1(label_boxes["class"], preds)
        top5_acc_score += update_top5(label_boxes["class"], top5_preds)
        print(f"Top 5 Score: {top5_acc_score / total_precision}")
        print(f"Top 1 Score: {acc_score / total_precision}")
        print(f"Recall: {acc_score / total_recall}")
    print(top5_acc_score / total_precision)


def load_dets():
    args = parse_args()
    cfg = parse_cfg(args.config_file)

    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)

    for ds in [food201]:
        ds.detect_patches(ds.test_path, ds.test_set, model_pth="best.pt")


def clip_update_preds():
    """
    Create a new detection dataframe (and csv) with updated classifications from YOLO.
    """

    # load dataset
    def get_clip_classifications(
        ds: Dataset, df: pd.DataFrame, cmap: pd.Series, exp: ExperimentManager
    ):
        pred_ids = []
        top5_ids = []
        query_vecs = []
        for batch in np.array_split(df["patch_pth"], len(df) / 100):
            imgs = []
            for img_name in batch:
                try:
                    imgs.append(Image.open(img_name))
                except FileNotFoundError:
                    continue
            query_vecs.extend(exp.embedder.get_embedding_from_preloaded(imgs))

        for idx, detection in df.iterrows():
            query_vec = query_vecs[idx]
            # TODO: convert class name to idx
            candidate_vecs = exp.db.query(None, query_vec)
            prediction, top5_classes = exp.db.vote_classification(candidate_vecs)
            # WARN: needs full cmap, not dataset-specific one
            pred_ids.append(cmap[prediction.strip()])
            top5_ids.append([cmap[x.strip()] for x in top5_classes])
        df["class_id"] = pred_ids
        df["top5_ids"] = top5_ids
        return df

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
    # Load detections from YOLO
    preds_set = []
    # Load CMAP from YOLO combined dataset
    cmap_pth = "/nfs/stak/users/beerya/soundbendor/food_datasets/combined_food_seg/dataset.yaml"
    with open(cmap_pth, "r") as stream:
        cmap = yaml.load(stream, Loader=yaml.Loader)["names"]
        # TODO: invert from {id: name} to {name: id}
        cmap = {v: k for k, v in cmap.items()}

    """Generate vectors"""
    exp = ExperimentManager(cfg)
    start_id = 0
    for ds in datasets:
        start_id = exp.update_vecdb(ds, "train", start_id)

    for ds in datasets:
        df = ds.get_patches("test", True)
        # cmap = pd.Series(ds.cmap.index.values, index=ds.cmap)
        df["source_img"] = df.apply(
            lambda x: ds.name + "_" + str(x["source_img"]), axis=1
        )
        df = get_clip_classifications(ds, df, cmap, exp)
        preds_set.append(df)

    preds = pd.concat(preds_set, ignore_index=True)
    print(preds)

    gt_set = []
    for ds in datasets:
        df = ds.get_patches("test", False)
        # Invert classmap to work as {name: id}
        cmap = pd.Series(ds.cmap.index.values, index=ds.cmap)
        df["dataset"] = ds.name
        df["src_img"] = df.apply(lambda x: ds.name + "_" + str(x["src_img"]), axis=1)
        df["class_id"] = df.apply(lambda x: cmap[x["class"]], axis=1)
        gt_set.append(df)
    labels = pd.concat(gt_set, ignore_index=True)
    print(labels)

    # Where each groupby is a Dataframe of ground truth objects for a given image
    preds["source_img"] = preds["source_img"].astype(str)
    preds_group = preds.sample(frac=1).groupby("source_img")
    # Each groupby:
    labels["src_img"] = labels["src_img"].astype(str)
    label_group = labels.groupby("src_img")
    print(label_group.groups.keys())

    # Initialize metrics class
    n_classes = sum([len(ds.cmap) for ds in datasets])
    metrics = ExperimentResult(preds_group, label_group, n_classes)

    precision, recall = metrics.get_pr()
    map50, map50_95 = metrics.get_map50()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP-50: {map50}")


# TODO: Build toggle to enable switching between [food201, foodseg103, ..., combined]
def baseline_yolo_experiment():
    # load dataset
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

    # Load detections from YOLO
    preds_set = []
    for ds in datasets:
        df = ds.get_patches("test", True)
        df["source_img"] = df.apply(
            lambda x: ds.name + "_" + str(x["source_img"]), axis=1
        )
        preds_set.append(df)

    preds = pd.concat(preds_set, ignore_index=True)
    print(preds)

    # Load ground truth data
    # WARN: patch_pth may get overwritten by legacy code here
    gt_set = []
    for ds in datasets:
        df = ds.get_patches("test", False)
        # Invert classmap to work as {name: id}
        cmap = pd.Series(ds.cmap.index.values, index=ds.cmap)
        df["dataset"] = ds.name
        df["src_img"] = df.apply(lambda x: ds.name + "_" + str(x["src_img"]), axis=1)
        df["class_id"] = df.apply(lambda x: cmap[x["class"]], axis=1)
        gt_set.append(df)
    labels = pd.concat(gt_set, ignore_index=True)
    print(labels)

    # Where each groupby is a Dataframe of ground truth objects for a given image
    preds["source_img"] = preds["source_img"].astype(str)
    preds_group = preds.sample(frac=1).groupby("source_img")
    # Each groupby:
    labels["src_img"] = labels["src_img"].astype(str)
    label_group = labels.groupby("src_img")
    print(label_group.groups.keys())

    # Initialize metrics class
    n_classes = sum([len(ds.cmap) for ds in datasets])
    metrics = ExperimentResult(preds_group, label_group, n_classes)

    precision, recall = metrics.get_pr()
    map50 = metrics.get_map50()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP-50: {map50}")
    # TODO: save predictions as experiment CSV


def final_experiment():
    """
    Create a new detection dataframe (and csv) with updated classifications from YOLO.
    """

    # load dataset
    def get_clip_classifications(
        ds: Dataset, df: pd.DataFrame, cmap: pd.Series, exp: ExperimentManager
    ):
        pred_names = []
        pred_ids = []
        top5_ids = []
        query_vecs = []
        for batch in np.array_split(df["patch_pth"], len(df) / 100):
            imgs = []
            for img_name in batch:
                try:
                    imgs.append(Image.open(img_name))
                except FileNotFoundError:
                    continue
            query_vecs.extend(exp.embedder.get_embedding_from_preloaded(imgs))

        for idx, detection in df.iterrows():
            query_vec = query_vecs[idx]
            # TODO: convert class name to idx
            candidate_vecs = exp.db.query(None, query_vec)
            prediction, top5_classes = exp.db.vote_classification(candidate_vecs)
            # WARN: needs full cmap, not dataset-specific one
            pred_ids.append(cmap[prediction.strip()])
            top5_ids.append([cmap[x.strip()] for x in top5_classes])
            pred_names.append(prediction.strip())
        df["class_id"] = pred_ids
        df["top5_ids"] = top5_ids
        df["class_name"] = pred_names
        return df

    args = parse_args()
    cfg = parse_cfg(args.config_file)

    # Load all food compost dataset
    foodseg103_pth = cfg["paths"]["foodseg103"]
    uecfoodpix_pth = cfg["paths"]["uecfoodpix"]
    food201_pth = cfg["paths"]["food201"]
    fw_pth = cfg["paths"]["fw_test"]

    food201 = Food201(root=food201_pth)
    uecfoodpix = UECFoodPix(root=uecfoodpix_pth)
    foodseg103 = FoodSeg103(root=foodseg103_pth)
    fw_test = FwTest(root=fw_pth)

    datasets = [food201, uecfoodpix, foodseg103]
    # Load detections from YOLO
    # Load CMAP from YOLO combined dataset
    cmap_pth = "/nfs/stak/users/beerya/soundbendor/food_datasets/combined_food_seg/dataset.yaml"
    with open(cmap_pth, "r") as stream:
        cmap = yaml.load(stream, Loader=yaml.Loader)["names"]
        # TODO: invert from {id: name} to {name: id}
        cmap = {v: k for k, v in cmap.items()}

    """Generate vectors"""
    exp = ExperimentManager(cfg)
    start_id = 0
    for ds in datasets:
        pass
    # start_id = exp.update_vecdb(ds, "train", start_id)

    """Add label vectors"""
    # start_id = exp.add_label_vectors(fw_test, start_id)

    # Get YOLO extracted patches
    fw_test.crop_patches("train")
    fw_test.detect_patches("train", fw_test.train_set, "best.pt")
    pred_df = fw_test.get_patches("train", True)
    pred_df["source_img"] = pred_df.apply(
        lambda x: fw_test.name + "_" + str(x["source_img"]), axis=1
    )
    pred_df = get_clip_classifications(fw_test, pred_df, cmap, exp)

    # Get ground truth
    gt_df = fw_test.get_patches("train", False)
    # Invert classmap to work as {name: id}
    cmap = pd.Series(fw_test.cmap.index.values, index=fw_test.cmap)
    gt_df["dataset"] = fw_test.name
    gt_df["src_img"] = gt_df.apply(
        lambda x: fw_test.name + "_" + str(x["src_img"]), axis=1
    )
    gt_df["class_id"] = gt_df.apply(lambda x: cmap[x["class"]], axis=1)

    # Where each groupby is a Dataframe of ground truth objects for a given image
    pred_df["source_img"] = pred_df["source_img"].astype(str)
    preds_group = pred_df.sample(frac=1).groupby("source_img")
    # Each groupby:
    gt_df["src_img"] = gt_df["src_img"].astype(str)
    label_group = gt_df.groupby("src_img")
    print(label_group.groups.keys())

    # Initialize metrics class
    n_classes = sum([len(ds.cmap) for ds in datasets])
    metrics = ExperimentResult(preds_group, label_group, n_classes, cmap)

    # WARN: metrics class currently compares class IDs
    precision, recall = metrics.get_pr(use_txt=True)
    map50, map50_95 = metrics.get_map50(use_txt=True)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"mAP-50: {map50}")


if __name__ == "__main__":
    final_experiment()
