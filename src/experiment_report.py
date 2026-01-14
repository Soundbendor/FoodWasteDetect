import pandas as pd


class ExperimentReport:
    def __init__(self, pred: dict[str, pd.DataFrame], actual: dict[str, pd.DataFrame]):
        self.pred = pred
        self.actual = actual

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

    def get_pr(self, iou_threshold: float = 0.50) -> tuple[float, float]:
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
            preds = preds.sort_values(by="conf")
            for _, pred in preds.iterrows():
                matched = False
                # For each remaining detection
                for gt_idx, gt in truths.iterrows():
                    if pred["class_id"] == gt["class_id"]:
                        iou = self.compute_iou(pred["box"], gt["box"])
                        if iou >= iou_threshold:
                            true_positive += 1
                            truths.drop(gt_idx)
                            matched = True
                            break
                # If none of the ground truths match this prediction, it is a false detection.
                if not matched:
                    false_positive += 1
        # TODO: Prevent divide by zero errors
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_positive
        return precision, recall

    def get_map50(self) -> float:
        # TODO: Implement map50 calculation
        pass
