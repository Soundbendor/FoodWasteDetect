import argparse
import os

import numpy as np
from ultralytics.utils.ops import segments2boxes

# for each file in /labels directory
# open file
# for each line in file
# skip first number (class label)
# load sequence of numbers as numpy array
# pass numpy array to ultralytics
# save in yolo format: {class label} {boxes}, space-delimited.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='Segment2Boxes', description='Convert Ultralytics YOLO-format image \
        segmentation dataset to bounding boxes')
    parser.add_argument('--dataset_path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    for fname in os.listdir(args.dataset_path):
        fpath = os.path.join(args.dataset_path, fname)
        seg_file = np.loadtxt(fpath)
        print(seg_file)
        # with open(fpath, 'r') as file:
            # for segment in file.readlines():
                # skip first number
                

