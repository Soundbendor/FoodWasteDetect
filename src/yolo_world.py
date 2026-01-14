import logging
import os
from itertools import chain

import pandas as pd
from ultralytics import YOLOWorld
from ultralytics.engine.results import Results

from data_wrappers.food201 import Food201
from experiment_report import ExperimentReport
from util import parse_args, parse_cfg

SUBSET = "test"


# Initialize a YOLO-World
# TODO: evaluation metric
def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = Food201(ds_path)
    model = YOLOWorld(
        "yolov8x-worldv2.pt"
    )  # or select yolov8m/l-world.pt for different sizes
    class_names = list(ds.get_class_labels())
    model.set_classes(class_names)

    # Get ground truth boxes for metrics class
    gt_dataset = ds.get_box_dataset(SUBSET)
    pred_ds = {}

    # Each dataframe represent one image.
    ds_img_keys = pd.Series(gt_dataset.keys())

    n_samples = int(cfg["settings"]["eval_samples"])
    if n_samples != 0:
        ds_img_keys = ds_img_keys.sample(n=n_samples, random_state=42)

    # TODO: Set up YOLO-World to report precision/recall
    for img_name in ds_img_keys:
        img_pth = os.path.join(ds_path, SUBSET, "images", img_name)
        results = model.predict(img_pth)[0]
        pred_ds[img_name] = get_predicted_detections(results)

    # Initialize metrics
    metrics = ExperimentReport(pred_ds, gt_dataset)
    precision, recall = metrics.get_pr()
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


def get_predicted_detections(results: Results) -> pd.DataFrame:
    """
    Convert YOLO predictions to DataFrame

    Take YOLO prediction and outputs DataFrame of [class_id, box, conf]
    Uses xywhn YOLOv8 compliant boxes
    """
    class_ids = results.boxes.cls.tolist()  # type: ignore
    boxes = results.boxes.xywhn.tolist()  # type: ignore
    confs = results.boxes.conf.tolist()
    return pd.DataFrame.from_dict({"class_id": class_ids, "box": boxes, "conf": confs})


if __name__ == "__main__":
    main()
