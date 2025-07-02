import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
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
    seg_path = os.path.join(args.dataset_path, 'labels')
    outdir = os.path.join(args.dataset_path, 'boxes')
    os.makedirs(outdir, exist_ok=True)
    print("Generating bounding boxes...")
    for fname in tqdm(os.listdir(seg_path)):
        fpath = os.path.join(seg_path, fname)
        segments = []
        labels = []
        with open(fpath, 'r') as file:
            for segment in file.readlines():
                seg_label = np.fromstring(segment, sep=' ')
                segments.append(seg_label[1:])
                labels.append(seg_label[0])
        bboxes = segments2boxes([np.array(s).reshape(-1, 2) for s in segments])
        with open(os.path.join(outdir, fname), 'w') as box_file:
            for idx, label in enumerate(labels):
                label_string = f"{int(label)} {' '.join([str(x) for x in bboxes[idx]])}\n"
                box_file.write(label_string)
        # Draw bounding boxes over image
        basename = os.path.splitext(fname)[0]
        img_pth = os.path.join(args.dataset_path, 'images', f"{basename}.jpg")
        out_pth = os.path.join(args.dataset_path, 'boxed_imgs', f"{basename}.jpg")
        source_img = Image.open(img_pth).convert('RGB')
        draw = ImageDraw.Draw(source_img)
        draw.rectangle([bboxes[0], bboxes[1], bboxes[0] + bboxes[3], bboxes[1] + bboxes[4]], fill="black")
        source_img.save(out_pth, "JPEG")

main()

