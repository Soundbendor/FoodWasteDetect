import copy
import os

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.data.converter import yolo_bbox2segment
from ultralytics.engine.results import Results
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

from data_wrappers.food201 import Food201
from data_wrappers.fwtest import FwTest
from experiment_report import ExperimentReport
from util.util import parse_args, parse_cfg

SUBSET = "test"


def convert_food201_class_idx(
    dataset: dict[str, pd.DataFrame], class_names: list[str]
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Remove "Unknown" vales from Food201 dataset class labels
    To avoid introducing noise into the label space
    """
    food201_class_conversion = {}
    new_idx = 0
    new_class_labels = []
    for original_idx, class_name in enumerate(class_names):
        if class_name == "Unknown":
            food201_class_conversion[original_idx] = -1
        else:
            food201_class_conversion[original_idx] = new_idx
            new_idx += 1
            new_class_labels.append(class_name)

    # Convert all class indices to new scheme
    for img_name, img_df in dataset.items():
        img_df["class_id"] = img_df["class_id"].apply(
            lambda x: food201_class_conversion[x]
        )

    return dataset, new_class_labels


def box_to_seg():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["fw-test"]
    yolo_bbox2segment(im_dir=os.path.join(ds_path, "images"), sam_model="sam3.pt")


def eval_yolo_26_seg():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"][args.dataset]
    model = YOLO(args.model)  # or select yolov8m/l-world.pt for different sizes

    results = model.train(data=ds_path, epochs=80)
    print(results)
    metrics = model.val()
    print(metrics)


def extract_vocabulary(pth: str) -> list[str]:
    with open(pth, "r") as file:
        dataset = yaml.safe_load(file)

    return dataset["names"]


def eval_yolo_new():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    food201_pth = cfg["paths"]["food201"]
    foodseg103_pth = cfg["paths"]["foodseg103"]
    uec_pth = cfg["paths"]["uecfoodpix"]
    unified = cfg["paths"]["unified"]
    model = YOLOE(args.model)  # or select yolov8m/l-world.pt for different sizes

    # food201_names = extract_vocabulary(food201_pth).values()
    # foodseg103_names = extract_vocabulary(uec)

    food201_results = model.val(data=food201_pth)
    foodseg103_results = model.val(data=foodseg103_pth)
    uec_results = model.val(data=uec_pth)

    print(f"\n\n\n FOOD201 \n\n\n")
    print(food201_results)
    print(f"\n\n\n FOODSEG103 \n\n\n")
    print(foodseg103_results)
    print(f"\n\n\n UECFOODPIX \n\n\n")
    print(uec_results)


def eval_yolo_e():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"][args.dataset]
    model = YOLOE(
        "yoloe-26x-seg.pt"
    )  # or select yolov8m/l-world.pt for different sizes

    # TODO: Load test set from data path
    # For each sample, load ground truth
    test_path = os.path.join(ds_path, "test")
    test_imgs = sorted(os.listdir(os.path.join(test_path, "images")))
    test_labels = sorted(os.listdir(os.path.join(test_path, "labels")))

    test_ds = {}
    for img_pth, label_pth in zip(test_imgs, test_labels):
        basename = os.path.splitext(os.path.basename(img_pth))[0]
        # open label file
        class_labels = []
        # seggs!!!
        segs = []
        with open(label_pth, "r") as label_file:
            for annot in label_file.readlines():
                # first digit is class label, all proceeding are seg coordinates
                annot = annot.split()
                class_labels.append(int(annot[0]))
                segs.append(np.array(annot[1:], dtype=float))

        test_ds[basename] = {
            "masks": torch.from_numpy(np.array(segs)),
            "labels": torch.from_numpy(np.array(class_labels)),
            "img_pth": img_pth,
        }

    test_preds = copy.deepcopy(test_ds)
    # run inference on test dataframe
    for sample_name, targets_dict in test_ds.values():
        results = model.predict(targets_dict["img_pth"])
        masks = []
        labels = []
        for result in results:
            # INFO: is xy the right format for pixel-wise masks?
            masks.append(result.masks.xy)
            labels.append(result.boxes.cls)
        test_preds[sample_name]["masks"] = torch.from_numpy(np.array(masks))
        test_preds[sample_name]["labels"] = torch.from_numpy(np.array(labels))
    # WARN: The interface for ExperimentReport expects a Pandas GroupBy
    # We are providing dictionaries
    metrics = ExperimentReport(None, None)
    metrics.get_map50(test_preds, test_df)


def train_yolo_e():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"][args.dataset]
    model = YOLOE(
        "yoloe-26x-seg.pt"
    )  # or select yolov8m/l-world.pt for different sizes

    results = model.train(data=ds_path, epochs=80, trainer=YOLOEPESegTrainer)
    print(results)
    metrics = model.val()
    print(metrics)


# Initialize a YOLO-World model
def eval_yolo_world():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FwTest(ds_path)
    model = YOLOWorld(
        "yolov8x-worldv2.pt"
    )  # or select yolov8m/l-world.pt for different sizes

    # Get ground truth boxes for metrics class
    class_names = list(ds.get_class_labels())
    gt_dataset = ds.get_box_dataset(SUBSET)
    gt_dataset, class_names = convert_food201_class_idx(gt_dataset, class_names)
    # Remove all "Unknown" values from dataset

    # Step 1: Make a new class map
    # Where each key is an old index [0...207], values are new values
    pred_ds = {}
    gt_subset = {}

    model.set_classes(class_names)
    # Each dataframe represent one image.
    ds_img_keys = pd.Series(gt_dataset.keys())

    n_samples = int(cfg["settings"]["eval_samples"])
    if n_samples != 0:
        ds_img_keys = ds_img_keys.sample(n=n_samples, random_state=42)

    for img_name in ds_img_keys:
        img_pth = os.path.join(ds_path, SUBSET, "images", img_name)
        results = model.predict(img_pth)[0]
        pred_ds[img_name] = get_predicted_detections(results)
        gt_subset[img_name] = gt_dataset[img_name]

    # Initialize metrics
    metrics = ExperimentReport(pred_ds, gt_subset)
    precision, recall = metrics.get_pr()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    metrics = metrics.get_map50()
    # print(f"mAP50: {metrics["map_50"]}")
    # print(f"mAP50-95: {metrics["map"]}")


def train_yolo_world():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = Food201(ds_path)
    model = YOLOWorld(
        "yolov8x-worldv2.pt"
    )  # or select yolov8m/l-world.pt for different sizes

    # Get ground truth boxes for metrics class
    class_names = list(ds.get_class_labels())
    gt_dataset = ds.get_box_dataset(SUBSET)

    # WARN: We're preserving the "Unknown" values for now, to see how they impact training YOLO-World.
    # Should remove in future runs.

    # TODO: Should validate that bounding box conversions saved in these files are valid and YOLO-compliant
    # gt_dataset, class_names = convert_food201_class_idx(gt_dataset, class_names)
    # Remove all "Unknown" values from dataset

    # Path to dataset.yaml for converted Food201
    training_ds = cfg["paths"]["food201_yolo"]
    model.train(data=training_ds, epochs=100)


def get_predicted_detections(results: Results) -> pd.DataFrame:
    """
    Convert YOLO predictions to DataFrame

    Take YOLO prediction and outputs DataFrame of [class_id, box, conf]
    Uses xywhn YOLOv8 compliant boxes
    """
    class_ids = results.boxes.cls.tolist()  # type: ignore
    boxes = results.boxes.xyxy.tolist()  # type: ignore
    confs = results.boxes.conf.tolist()
    return pd.DataFrame.from_dict(
        {"class_id": class_ids, "xyxy_box": boxes, "conf": confs}
    )


def main():
    eval_yolo_world()
