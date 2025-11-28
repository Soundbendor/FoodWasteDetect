import os
from typing import Callable, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics.utils.ops import xywhn2xyxy

from database.img_crop import ImageCropper


class Dataset:
    """Template class for datasets"""

    def __init__(self):
        self.name = ""
        self.train_path = ""
        self.test_path = ""
        self.val_path = ""
        self.root = ""
        self.cmap = self._get_classmap()

    def get_class_labels(self) -> pd.Series:
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

    def detect_patches(
        self, subset_pth: str, subset_loader: Callable, model_pth: str
    ) -> None:
        """
        Use ObjectCropper solution to extract patches from
        image classification dataset. Save in detected-patches directory
        and make patches_dets.csv.
        """
        imgs_pth = os.path.join(self.root, subset_pth, "images")
        dets_pth = os.path.join(self.root, subset_pth, "cropped-detections")
        if os.path.isdir(os.path.join(imgs_pth, "cropped-detections")):
            print("WARN: Patches already exist in this directory.")
            return
        os.makedirs(dets_pth, exist_ok=True)
        cropper = ImageCropper(dataset=self, model=model_pth)
        cropper.crop_dataset(subset_pth, subset_loader)

    def crop_patches(self, subset_pth: str) -> None:
        """
        Given a set of bounding boxes, and a set of images,
        Crop each image according to its bounding boxes and save patches
        Save class labels and parent images to patches.csv
        """
        imgs_pth = os.path.join(self.root, subset_pth, "images")
        patches_pth = os.path.join(self.root, subset_pth, "patches")
        os.makedirs(patches_pth, exist_ok=True)
        records = []
        # First, check if patches already exist.
        if os.path.isdir(os.path.join(imgs_pth, "patches")):
            print("WARN: Patches already exist in this directory.")
            return
        for img in tqdm(os.listdir(imgs_pth)):
            basename = os.path.splitext(img)[0]
            img_pth = os.path.join(imgs_pth, img)
            label_pth = os.path.join(self.root, subset_pth, "boxes", basename + ".txt")
            src_img = Image.open(img_pth)
            width, height = src_img.size
            # If the file has no detections, skip it.
            if not os.path.isfile(label_pth):
                continue
            with open(label_pth, "r") as label_file:
                for idx, line in enumerate(label_file.readlines()):
                    yolo_label = line.split()
                    class_label = self.cmap[int(yolo_label[0])]
                    coordinates = xywhn2xyxy(
                        np.array(yolo_label[1:], dtype=np.float32), width, height
                    )
                    patch = src_img.crop(coordinates)
                    patch_name = f"{basename}_{idx}_{class_label}.jpg"
                    try:
                        patch.save(os.path.join(patches_pth, patch_name))
                    except ValueError:
                        # empty detection, skip
                        continue
                    records.append(
                        {
                            "patch_name": patch_name,
                            "class": class_label,
                            "box_coords": coordinates,
                            "src_img": basename,
                        }
                    )
        df = pd.DataFrame.from_records(records)
        df.to_csv(os.path.join(os.path.join(self.root, subset_pth), "patches.csv"))

    def get_patches(
        self, subset: str, detections: bool, override=False
    ) -> pd.DataFrame:
        def update_sample_path(fname: str, subset: str, folder: str, root: str) -> str:
            return os.path.join(root, subset, "patches", fname)

        # 1) Check if patches directory exists
        if subset in ["train", "test", "val", "validation"]:
            if detections:
                df_path = os.path.join(self.root, subset, "dets.csv")
                folder = "cropped-detections"
            else:
                df_path = os.path.join(self.root, subset, "patches.csv")
                folder = "patches"
            # 2) Check if patches.csv exists
            if os.path.isfile(df_path):
                df = pd.read_csv(df_path)
                if override:
                    return df
                df["patch_pth"] = df["patch_name"].apply(
                    lambda x: update_sample_path(x, subset, folder, self.root)
                )
                return df
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


class SegDataset(Dataset):
    def __init__(self):
        super().__init__()

    def _build_df(self, split: str):
        """
        Build a dataframe containing image paths for a dataset subset.
        should have [img, class_id, mask, segment, box]
        """
        # Go to either train or test
        # WARN: Order is not guaranteed here.
        path = os.path.join(self.root, split)
        dirs = [os.path.join(path, x) for x in os.listdir(path)]
        dirs = [x for x in dirs if os.path.isdir(x)]
        basenames = [os.path.basename(x) for x in dirs]
        paths = zip(dirs, basenames)
        # WARN: This is a bad way to build the training dataframe.
        # If any folder in train/test dir is not equal to size of dataset,
        # this throws an obscure error.
        df = pd.DataFrame(columns=basenames)
        for dir, basename in paths:
            # Explicitly ignore patches directory
            if basename == "patches":
                continue
            # We need a special handler for boxes, to fix missing files
            # TODO: complete or delete
            # if basename == "boxes":
            #     for file in os.listdir("boxed_imgs"):
            #         fname = os.path.splitext(file)[0]
            #         if
            df[basename] = sorted(os.listdir(dir))
        df.to_csv(os.path.join(self.root, f"{split}.csv"), index=False)

    def _load_df(self, fname: str):
        # load csv. if it doesn't exist, create it
        csv_pth = os.path.join(self.root, fname)
        if not os.path.isfile(csv_pth):
            # build test set csv
            subset = os.path.splitext(fname)[0]
            self._build_df(subset)
        return pd.read_csv(csv_pth)
