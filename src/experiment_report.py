import numpy as np
import pandas as pd
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class ExperimentReport:
    def __init__(self, pred: dict[str, pd.DataFrame], actual: dict[str, pd.DataFrame]):
        self.pred = pred
        self.actual = actual

    # WARN: This IoU computation is not compatible with xywhn center-xy coordinates.
    def compute_iou(self, box1, box2):
        """
        Given two boxes, compute the IOU.
        """
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

    def get_pr(self, iou_threshold: float = 0.00) -> tuple[float, float]:
        """
        At the end of an experiment, calculate the classwise precision and recall
        """
        # INFO: Assumes pred and actual share identical key names
        true_positive = 0
        false_positive = 0
        total_positive = 0
        for img_name in self.pred.keys():
            # Where preds and truths here refer to the boxes for this image
            preds = self.pred[img_name]
            truths = self.actual[img_name]
            # Total number of positives is equal to the number of ground truth labels in the dataset
            total_positive += len(truths)
            # To compute p/r, we only compute IOU in cases of matching class IDs
            # Sort predictons  by confidence
            preds = preds.sort_values(by="conf", ascending=False)
            for _, pred in preds.iterrows():
                matched = False
                matched_dets = set()
                # For each remaining detection
                for gt_idx, gt in truths.iterrows():
                    if gt_idx in matched_dets:
                        continue
                    if pred["class_id"] == gt["class_id"]:
                        iou = self.compute_iou(pred["box"], gt["box"])
                        if iou >= iou_threshold:
                            true_positive += 1
                            matched_dets.add(gt_idx)
                            matched = True
                            break
                # If none of the ground truths match this prediction, it is a false detection.
                if not matched:
                    false_positive += 1
        # TODO: Prevent divide by zero errors
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_positive
        return precision, recall

    def get_results_tensor(
        self, dataset: dict[str, pd.DataFrame], is_pred: bool
    ) -> list[dict]:
        """
        From a Dataset of predictions, reformat into a TorchMetrics compliant list.

        Should contain list of dicts, where each dict represents an image.
            boxes: {n x 4} dimension matrix, where n is number of detections
            scores: {n} dimension vector, confidence scores for each det.
            labels: {n} dimension vector, 0-indexed class labels for each det.
        """
        preds = []
        for img_name, img_df in dataset.items():
            result = {
                "boxes": torch.from_numpy(
                    np.array(img_df["xyxy_box"].tolist()).astype(np.float64)
                ),
                "labels": torch.from_numpy(
                    img_df["class_id"].to_numpy().astype(np.int64)
                ),
            }
            if is_pred:
                result["scores"] = torch.from_numpy(img_df["conf"].to_numpy())
            preds.append(result)
        return preds

    def get_map50(self) -> dict:
        """
        Use TorchMetrics to get Mean Average Precision and Mean Average Recall
        """
        # INFO: Box type set to cxcywh to comply with YOLO bounding box formats
        # INFO: MAP also supports 'segm' iou type for instance segmentation evaluation
        metric = MeanAveragePrecision(
            iou_type="bbox", box_format="xyxy", class_metrics=True
        )
        preds = self.get_results_tensor(self.pred, is_pred=True)
        targets = self.get_results_tensor(self.actual, is_pred=False)

        metric.update(preds, targets)
        results = metric.compute()
        return results
