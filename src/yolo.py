import logging
from itertools import chain

from ultralytics import YOLOWorld

from data_wrappers.foodx251 import FoodX251
from util import EvalMetric, parse_args, parse_cfg


# Initialize a YOLO-World
# TODO: evaluation metric
def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg['paths']['dataset']
    ds = FoodX251(ds_path)
    model = YOLOWorld("yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt for different sizes
    model.set_classes(ds.cmap['label'])

    val_set = ds.val_set()
    metric = EvalMetric()

    n_samples = int(cfg['settings']['eval_samples'])
    if n_samples != 0:
        val_set = val_set.sample(n=n_samples, random_state=42)

    acc = 0
    for i, row in val_set.iterrows():
        results = model.predict(f"{ds_path}/val/val_set/{row['fname']}")
        preds = list(chain.from_iterable([x.boxes.cls.tolist() for x in results]))
        preds = [int(x) for x in preds]
        ground_truth = ds.cmap.index[ds.cmap['label'] == row['class']].tolist()[0]
        if ground_truth in preds:
            acc += 1
        logging.info(results)
        logging.info(f"Ground Truth Label: {row['class']}")
        logging.info(f"Predicted Labels: {[ds.cmap.iloc[x] for x in preds]}")
    print(f"ACCURACY: {acc / n_samples}")
