import os

import pandas as pd
from ultralytics import YOLOE, YOLOWorld
from ultralytics.engine.results import Results
from ultralytics.models.yolo.yoloe import YOLOEPETrainer

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


def eval_yolo_e():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["food201"]
    ds = Food201(ds_path)
    model = YOLOE(
        "yoloe-26x-seg.pt"
    )  # or select yolov8m/l-world.pt for different sizes

    # Get ground truth boxes for metrics class
    class_names = list(ds.get_class_labels())
    # Remove all "Unknown" values from dataset
    gt_dataset = ds.get_box_dataset(SUBSET)
    gt_dataset, class_names = convert_food201_class_idx(gt_dataset, class_names)
    model.set_classes(class_names)

    results = model.train(
        data=cfg["paths"]["food201_yolo"], epochs=10, trainer=YOLOEPETrainer
    )
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
