import argparse
import functools
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics.data.utils import visualize_image_annotations
from ultralytics.utils.ops import xyxy2xywh

from data_wrappers.food201 import Food201

# for each file in /labels directory
# open file
# for each line in file
# skip first number (class label)
# load sequence of numbers as numpy array
# pass numpy array to ultralytics
# save in yolo format: {class label} {boxes}, space-delimited.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Segment2Boxes",
        description="Convert Ultralytics YOLO-format image \
        segmentation dataset to bounding boxes",
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    return parser.parse_args()


def compute_bounding_box(segment: np.ndarray, w: int, h: int) -> np.ndarray:
    # Convert into (x, y) coordinates
    coords = segment.reshape(-1, 2)
    # Scale normalized coordinates by image width/height
    coords[:, 0] *= w
    coords[:, 1] *= h
    box = [
        np.min(coords[:, 0]),
        np.min(coords[:, 1]),
        np.max(coords[:, 0]),
        np.max(coords[:, 1]),
    ]
    return np.array(list(map(int, box)))


def draw_box_img(img, boxes: List[np.ndarray], out_pth: str) -> None:
    source_img = img.convert("RGB")
    draw = ImageDraw.Draw(source_img)
    for box in boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="red")
    source_img.save(out_pth, "JPEG")


# Read a segmentation coordinate file in YOLO format.
# Returns {'class_id': [coordinates]}
def read_segments(fpath: str) -> dict:
    labels = {}
    with open(fpath, "r") as file:
        for segment in file.readlines():
            seg_label = np.fromstring(segment, sep=" ")
            labels[seg_label[0]] = seg_label[1:]
    return labels


# Take (x, y, w, h) left-corner aligned coordinates
# and convert to (x_center, y_center, w, h) image-scale normalized coordinates
def coco_to_yolo(coco_box: List[float], img_w: int, img_h: int) -> List[float]:
    # unpack coco box coordinates
    x_center, y_center, box_w, box_h = coco_box
    # move (x, y) from left-corner to center
    # x_center = x_left + (box_w / 2)
    # y_center = y_left + (box_h / 2)
    # normalize to image scale
    return [x_center / img_w, y_center / img_h, box_w / img_w, box_h / img_h]


def main(ds_split: str):
    seg_path = os.path.join(ds_split, "labels")
    outdir = os.path.join(ds_split, "boxes")
    box_dir = os.path.join(ds_split, "boxed_imgs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    print("Generating bounding boxes...")
    for fname in tqdm(os.listdir(seg_path)):
        fpath = os.path.join(seg_path, fname)
        labels = read_segments(fpath)
        basename = os.path.splitext(fname)[0]
        img_pth = os.path.join(ds_split, "images", f"{basename}.jpg")
        with Image.open(img_pth) as source_img:
            w, h = source_img.size
            # Compute bounding box coordinates (x0, y0, x1, y1)
            bboxes = [compute_bounding_box(s, w, h) for s in labels.values()]
            # Draw boxes and render image
            out_pth = os.path.join(box_dir, f"{basename}.jpg")
            draw_box_img(source_img, bboxes, out_pth)

            # Convert coordinates to (x, y, w, h), normalize, and write to file
            try:
                coco_boxes = xyxy2xywh(np.array(bboxes))
            except Exception as e:
                print(e)
                with open(os.path.join(outdir, f"{os.path.splitext(fname)[0]}_NAN.txt"), "w") as box_file:
                    box_file.write("Empty")
                continue

            # Normalize coordinates
            coco_converter = functools.partial(coco_to_yolo, img_w=w, img_h=h)
            yolo_boxes = list(map(coco_converter, coco_boxes))

            # Write coordinates to label file
            with open(os.path.join(outdir, fname), "w") as box_file:
                for idx, label in enumerate(labels.keys()):
                    label_string = (
                        f"{int(label)} {' '.join([str(x) for x in yolo_boxes[idx]])}\n"
                    )
                    box_file.write(label_string)


def visualize_annotations(ds_path: str):
    # load dataset
    ds = Food201(ds_path)
    cmap = ds.get_class_list()["label"].tolist()
    # randomly sample 5 images
    for img_fname in os.listdir(os.path.join(ds_path, "test", "images"))[:5]:
        basename = os.path.splitext(img_fname)[0]
        img_pth = os.path.join(ds_path, "test", "images", img_fname)
        label_pth = os.path.join(ds_path, "test", "boxes", basename + ".txt")
        # TODO: get classmap
        visualize_image_annotations(img_pth, label_pth, cmap)
        plt.savefig(f"{basename}.jpg")


if __name__ == "__main__":
    args = parse_args()
    # visualize_annotations(args.dataset_path)
    # for ds_split in os.listdir(args.dataset_path):
    main(args.dataset_path)
