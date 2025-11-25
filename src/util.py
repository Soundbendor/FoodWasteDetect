import argparse
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
    def __init__(self, preds: DataFrameGroupBy, truths: DataFrameGroupBy):
        # Each element of (preds, truths) represents 1 image
        # Contains
        self.preds = preds
        self.truths = truths
        self.total_acc_count = 0
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

            gt_boxes["label_ids"] = gt_boxes.apply(
                lambda x: self.map_class_index(x["class_id"], x["dataset"]), axis=1
            )
            pred_classes = preds_df["class_id"]
            img_p, img_r = self._get_img_pr(
                list(gt_boxes["label_ids"]), list(pred_classes)
            )
            print(f"DEBUG - Precision: {img_p}, Recall: {img_r}")
            sum_p.append(img_p)
            sum_r.append(img_r)
        return (np.mean(sum_p), np.mean(sum_r))

    def get_map50(self) -> float:
        """
        Calculate mAP-50 across all predictions
        """
        pass
        # Needs list of detections with properties
        # bbox: List[xmin, ymin, xmax, ymax]
        # class_id:
        # confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument("--config_file", help="Path to experiment config")
    parser.add_argument("--partition", type=int)
    return parser.parse_args()


def parse_cfg(cfg_file: str) -> dict:
    logging.basicConfig(level=logging.INFO)
    with open(cfg_file, "r") as stream:
        return yaml.load(stream, Loader=yaml.Loader)
