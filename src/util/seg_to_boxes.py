import argparse
import os
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics.utils.ops import xyxy2xywh

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
    draw = ImageDraw.Draw(img)
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
            coco_boxes = xyxy2xywh(np.array(bboxes))
            yolo_boxes = []
            for coco_box in coco_boxes:
                yolo_boxes.append(
                    [coco_box[0] / w, coco_box[1] / w, coco_box[2] / h, coco_box[3] / h]
                )
            # Normalize coordinates
            with open(os.path.join(outdir, fname), "w") as box_file:
                for idx, label in enumerate(labels.keys()):
                    label_string = (
                        f"{int(label)} {' '.join([str(x) for x in yolo_boxes[idx]])}\n"
                    )
                    box_file.write(label_string)


if __name__ == "__main__":
    args = parse_args()
    for ds_split in os.listdir(args.dataset_path):
        main(ds_split)
