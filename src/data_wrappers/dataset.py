import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics.utils.ops import xywhn2xyxy


class Dataset:
    """Template class for datasets"""

    def __init__(self):
        self.train_path = ""
        self.test_path = ""
        self.val_path = ""
        self.root = ""
        self.cmap = self._get_classmap()

    def get_class_list(self) -> List[str]:
        raise NotImplementedError  # type: ignore

    def _get_classmap(self) -> pd.Series:
        raise NotImplementedError

    def test_set(self) -> pd.DataFrame:
        raise NotImplementedError

    def train_set(self) -> pd.DataFrame:
        raise NotImplementedError

    def val_set(self) -> pd.DataFrame:
        raise NotImplementedError

    def get_subsets(self) -> list[tuple[str, pd.DataFrame]]:
        return list(
            zip(
                [self.train_path, self.test_path, self.val_path],
                [self.train_set, self.test_set, self.val_set],
            )
        )

    def crop_patches(self, subset_pth: str) -> None:
        """
        Given a set of bounding boxes, and a set of images,
        Crop each image according to its bounding boxes and save patches
        Save class labels and parent images to patches.csv
        """
        imgs_pth = os.path.join(self.root, subset_pth, "images")
        records = []
        # First, check if patches already exist.
        if os.path.isdir(os.path.join(imgs_pth, "patches")):
            print("WARN: Patches already exist in this directory.")
            return
        for img in os.listdir(imgs_pth):
            basename = os.path.splitext(img)[0]
            img_pth = os.path.join(imgs_pth, img)
            label_pth = os.path.join(subset_pth, "boxes", basename + ".txt")
            src_img = Image.open(img_pth)
            width, height = src_img.size
            with open(label_pth, "r") as label_file:
                for idx, line in enumerate(label_file.readlines()):
                    yolo_label = line.split()
                    class_label = self.cmap[yolo_label[0]]
                    # WARN: does ultralytics conversion work for single label?
                    coordinates = xywhn2xyxy(np.array(yolo_label[1:]), width, height)
                    patch = src_img.crop(coordinates)
                    patch_name = f"{basename}_{idx}_{class_label}.jpg"
                    patch.save(os.path.join(subset_pth, "patches", patch_name))
                    records.append(
                        {
                            "patch_name": patch_name,
                            "class": class_label,
                            "box_coords": coordinates,
                            "src_img": basename,
                        }
                    )
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(subset_pth, "patches.csv"))

    def get_patches(self, subset: str) -> pd.DataFrame:
        # 1) Check if patches directory exists
        if subset in ["train", "test", "val"]:
            df_path = os.path.join(self.root, subset, "patches.csv")
            # 2) Check if patches.csv exists
            if os.path.isfile(df_path):
                return pd.read_csv(df_path)
            raise FileNotFoundError()
        raise ValueError("Must use train, test, or val")

    def get_boxes(self, subset: str) -> str:
        """
        Given test, train, or val subset, return the path to bounding boxes for this set.
        """
        if subset in ["train", "test", "val"]:
            boxes_pth = os.path.join(self.root, subset, "boxes")
            if os.path.isdir(boxes_pth):
                return boxes_pth
            raise FileNotFoundError(f"Boxes directory does not exist for {subset}")
        raise ValueError("Must use train, test, or val")
