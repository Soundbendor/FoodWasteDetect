import argparse
import ast
import logging
import os

import numpy as np
import pandas as pd
import yaml
from mean_average_precision import MetricBuilder
from pandas.core.groupby import DataFrameGroupBy
from qdrant_client.models import ScoredPoint

from database.vecdb import VectorDB


# TODO: Refactor old experiments, deprecate EvalMetric
class EvalMetric:
    def __init__(self):
        self.metrics = ["top1", "top5", "voting"]
        self.scores = pd.Series([0, 0, 0], index=self.metrics)
        self.len = 0

    def update_scores(self, q_vecs: list[ScoredPoint], db: VectorDB, label: str):
        self.scores = self.scores.add(
            [db.score(q_vecs, label, strat)[0] for strat in self.metrics]
        )
        self.len += 1

    def compute_accuracies(self):
        return self.scores / self.len


class ExperimentResult:
    # INFO: Expects preds, results saved in df, loaded and grouped by file name
    def __init__(
        self, preds: DataFrameGroupBy, truths: DataFrameGroupBy, n_classes: int
    ):
        # Each element of (preds, truths) represents 1 image
        # Contains
        self.preds = preds
        self.truths = truths
        self.total_acc_count = 0
        self.n_classes = n_classes
        self.ds_map = {
            "foodseg103": pd.read_csv("src/util/foodseg103_map.csv", index_col=0),
            "uecfoodpix": pd.read_csv("src/util/uec_map.csv", index_col=0),
        }

    def map_class_index(self, class_id: int, ds_name: str) -> int:
        """
        Given a class label and a dataset, return the correct class ID in the unified dataset space
        """
        if ds_name == "food201":
            return class_id
        return self.ds_map[ds_name].iloc[class_id]["new_id"]

    def get_gts(self):
        """
        Counts the number of correct bounding box predictions
        """
        pass

    def _get_img_pr(self, labels: list[int], preds: list[int]) -> tuple[float, float]:
        c = 0
        n_preds = len(preds)
        if len(preds) == 0:
            if len(labels) == 0:
                return (1, 1)
            return (0, 0)

        for y in labels:
            for idx, y_hat in enumerate(preds):
                if y == y_hat:
                    preds.pop(idx)
                    c += 1
                    break

        # precision: number of correct detections / number of total detections
        p = c / n_preds
        # recall: number of correct detections / number of ground truth labels
        r = c / len(labels)
        return (p, r)

    def get_pr(self) -> tuple[float, float]:
        """
        Returns (precision, recall) values
        """
        sum_p = []
        sum_r = []
        for img_name, preds_df in self.preds:
            # Get matching ground truth boxes
            try:
                gt_boxes = self.truths.get_group(os.path.splitext(img_name)[0])
            except Exception:
                print(f"File {img_name} not found in the label set!")
                continue

            label_ids = gt_boxes.apply(
                lambda x: self.map_class_index(x["class_id"], x["dataset"]), axis=1
            )
            # WARN: We might be artificially inflating scores here
            # by ignoring cases where YOLO incorrectly classifies a detection
            # as a background result.
            # Additionally, the presence of "unknown" classifications indicates an
            # issue with our YOLO model.
            if any(type(x) == pd.Series for x in gt_boxes["class_id"]):
                continue
            pred_classes = preds_df["class_id"]
            img_p, img_r = self._get_img_pr(list(label_ids), list(pred_classes))

            print(f"DEBUG - Precision: {img_p}, Recall: {img_r}")
            sum_p.append(img_p)
            sum_r.append(img_r)
        return (np.mean(sum_p), np.mean(sum_r))

    def new_map50(self, iou_threshold=0.5) -> float:

        def unpack_gt_box(coords: str) -> list[int]:
            return list(map(int, coords[1:-1].split()))

        def compute_iou(box1, box2):
            x1, y1, x2, y2 = box1
            x1g, y1g, x2g, y2g = box2

            xi1 = max(x1, x1g)
            yi1 = max(y1, y1g)
            xi2 = min(x2, x2g)
            yi2 = min(y2, y2g)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x2g - x1g) * (y2g - y1g)
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area

        aps = []
        for img_name, preds_df in self.preds:
            tp = 0
            fp = 0
            gt = []
            preds = []
            try:
                gt_boxes = self.truths.get_group(os.path.splitext(img_name)[0])
            except Exception:
                print(f"File {img_name} not found in the label set!")
                continue
            # Re-map ground truth class IDs
            label_ids = gt_boxes.apply(
                lambda x: self.map_class_index(x["class_id"], x["dataset"]), axis=1
            )
            # WARN: We might be artificially inflating scores here
            if any(type(x) == pd.Series for x in gt_boxes["class_id"]):
                continue

            for idx, box in gt_boxes.iterrows():
                # [x1, y1, x2, y2, label_id]
                gt.append([*unpack_gt_box(box["box_coords"]), label_ids[idx]])
            for idx, pred in preds_df.iterrows():
                # [x1, y1, x2, y2, label_id, confidence]
                preds.append(
                    [*ast.literal_eval(pred["xyxy"])[0], pred["class_id"], pred["conf"]]
                )

            used = [False] * len(gt)
            for detection in preds:
                matched = False
                for idx, a in enumerate(gt):
                    if used[idx]:
                        continue

                    iou = compute_iou(detection[:4], a[:4])
                    if iou >= iou_threshold and a[4] == detection[4]:
                        tp += 1
                        used[idx] = True
                        matched = True
                        break
                if not matched:
                    fp += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(gt) if len(gt) > 0 else 0
            aps.append(precision * recall)
        return float(np.mean(aps) if len(aps) > 0 else 0)

    # def get_map50(self) -> float:
    #     """
    #     Calculate mAP-50 across all predictions
    #     """
    #
    #     gt = []
    #     preds = []
    #     for img_name, preds_df in self.preds:
    #         # Get matching ground truth boxes
    #         try:
    #             gt_boxes = self.truths.get_group(os.path.splitext(img_name)[0])
    #         except Exception:
    #             print(f"File {img_name} not found in the label set!")
    #             continue
    #
    #         label_ids = gt_boxes.apply(
    #             lambda x: self.map_class_index(x["class_id"], x["dataset"]), axis=1
    #         )
    #         if any(type(x) == pd.Series for x in gt_boxes["class_id"]):
    #             continue
    #         for idx, box in gt_boxes.iterrows():
    #             gt.append([*unpack_gt_box(box["box_coords"]), label_ids[idx], 0, 0])
    #         for idx, pred in preds_df.iterrows():
    #             preds.append(
    #                 [*ast.literal_eval(pred["xyxy"]), pred["class_id"], pred["conf"]]
    #             )
    #
    #     metric_fn = MetricBuilder.build_evaluation_metric(
    #         "map_2d", async_mode=True, num_classes=self.n_classes
    #     )
    #     metric_fn.add(np.array(preds), np.array(gt))
    #     print(
    #         f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}"
    #     )
    #
    #     # compute PASCAL VOC metric at the all points
    #     print(
    #         f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}"
    #     )
    #
    #     # compute metric COCO metric
    #     print(
    #         f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}"
    #     )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument("--config_file", help="Path to experiment config")
    parser.add_argument("--partition", type=int)
    return parser.parse_args()


def parse_cfg(cfg_file: str) -> dict:
    logging.basicConfig(level=logging.INFO)
    with open(cfg_file, "r") as stream:
        return yaml.load(stream, Loader=yaml.Loader)
