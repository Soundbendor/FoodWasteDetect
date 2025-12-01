import argparse
import ast
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from mean_average_precision import MetricBuilder
from pandas.core.groupby import DataFrameGroupBy
from qdrant_client.models import ScoredPoint
from sklearn.metrics import average_precision_score

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

    def reverse_class_map(self, class_id: int, ds_name: str) -> int:
        if ds_name == "food201":
            return class_id
        mp = self.ds_map[ds_name]
        mp = pd.Series(mp.index.values, index=mp["new_id"])
        print(mp)
        print(class_id)
        try:
            return mp[int(class_id)]
        # value was classified as something outside of label space
        except Exception as e:
            return 0

    def _unpack_gt_box(self, coords: str) -> list[int]:
        return list(map(int, coords[1:-1].split()))

    def get_pr(self, iou_threshold=0.5) -> tuple[float, float]:

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

        tp = 0
        fp = 0
        positives = 0
        for img_name, preds_df in self.preds:
            preds = []
            gt = []
            try:
                gt_boxes = self.truths.get_group(os.path.splitext(img_name)[0])
            except Exception:
                print(f"File {img_name} not found in the label set!")
                continue
            # Re-map ground truth class IDs
            label_ids = gt_boxes.apply(
                lambda x: self.map_class_index(x["class_id"], x["dataset"]), axis=1
            )
            if any(type(x) == pd.Series for x in gt_boxes["class_id"]):
                continue

            for idx, box in gt_boxes.iterrows():
                # [x1, y1, x2, y2, label_id]
                gt.append([*self._unpack_gt_box(box["box_coords"]), label_ids[idx]])

            for idx, pred in preds_df.iterrows():
                # [x1, y1, x2, y2, label_id, confidence]
                preds.append(
                    [*ast.literal_eval(pred["xyxy"])[0], pred["class_id"], pred["conf"]]
                )

            preds.sort(key=lambda x: x[5], reverse=True)
            used = [False] * len(gt)
            positives += len(gt)
            # Used for keeping track of tp/fp for mAP score
            for detection in preds:
                pred_id = detection[4]
                matched = False
                for idx, a in enumerate(gt):
                    gt_id = a[4]
                    if used[idx]:
                        continue

                    iou = compute_iou(detection[:4], a[:4])
                    if iou >= iou_threshold and gt_id == pred_id:
                        tp += 1
                        used[idx] = True
                        matched = True
                        break
                if not matched:
                    fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / positives if positives > 0 else 0
        return precision, recall

    # def get_map50_95(self) -> float:
    #     maps = []
    #     for iou_threshold in np.arange(0.50, 1, step=0.05):
    #         map, _, _ = self.new_map50(iou_threshold)
    #     return np.mean(maps)

    def get_map50(self) -> tuple[float, float]:
        """
        Calculate mAP-50 across all predictions
        """

        gt = []
        preds = []
        for img_name, preds_df in self.preds:
            # Get matching ground truth boxes
            try:
                gt_boxes = self.truths.get_group(os.path.splitext(img_name)[0])
            except Exception:
                print(f"File {img_name} not found in the label set!")
                continue

            # TODO: Bring everything in to the original label space
            label_ids = gt_boxes.apply(
                lambda x: self.map_class_index(x["class_id"], x["dataset"]), axis=1
            )
            # pred_ids = preds_df.apply(
                # lambda x: self.reverse_class_map(x["class_id"], "foodseg103"), axis=1
            # )

            if any(type(x) == pd.Series for x in gt_boxes["class_id"]):
                continue
            for idx, box in gt_boxes.iterrows():
                gt.append(
                        # Original
                    [*self._unpack_gt_box(box["box_coords"]), int(label_ids[idx]), 0, 0]
                        # New
                    # [*self._unpack_gt_box(box["box_coords"]), box["class_id"], 0, 0]
                )
            for idx, pred in preds_df.iterrows():
                preds.append(
                        # Old (for combined)
                    [*ast.literal_eval(pred["xyxy"])[0], pred["class_id"], pred["conf"]]
                        # New (for dataset-specific)
                    # [*ast.literal_eval(pred["xyxy"])[0], int(pred_ids[idx]), pred["conf"]]
                )


        metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=414
        )
        print(preds)
        print(gt)
        metric_fn.add(np.array(preds), np.array(gt))
        # Scale labels down to range
        # should be such that min(label) - delta = 0
        # delta is equal to -min(label)


        # compute PASCAL VOC metric at the all points
        map50 = metric_fn.value(iou_thresholds=0.5)["mAP"]
        map50_95 = metric_fn.value(
            iou_thresholds=np.arange(0.5, 1.0, 0.05),
            recall_thresholds=np.arange(0.0, 1.01, 0.01),
            mpolicy="soft",
        )["mAP"]
        print(f"VOC PASCAL mAP in all points: {map50}")

        # compute metric COCO metric
        print(f"COCO mAP: {map50_95}")
        return map50, map50_95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument("--config_file", help="Path to experiment config")
    parser.add_argument("--partition", type=int)
    return parser.parse_args()


def parse_cfg(cfg_file: str) -> dict:
    logging.basicConfig(level=logging.INFO)
    with open(cfg_file, "r") as stream:
        return yaml.load(stream, Loader=yaml.Loader)
