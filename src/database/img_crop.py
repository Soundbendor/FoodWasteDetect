import os

import pandas as pd
from ultralytics import solutions

from data_wrappers.dataset import Dataset


class ImageCropper:
    def __init__(self, ds: Dataset) -> None:
        # TODO: Convert get_class_list to return list
        self.ds = ds
        self.classes = ds.get_class_list()
        self.model = solutions.ObjectCropper(
            model="yolo11x.pt", crop_dir="cropped-detections"
        )

    def crop_dataset(self):
        # for img in dataset
        # load training set
        # iterate over each training folder
        for subset_dirname, dataloader in self.ds.get_subsets():
            try:
                subset = dataloader()
            except NotImplementedError:
                print(f"Error: Subset {dataloader} not implemented for dataset!")
                continue

            ds_path = os.path.join(self.ds.root, subset_dirname)
            for img_name in subset["fname"]:
                img_pth = os.path.join(ds_path, img_name)
                results = self.model.process(img_pth)
                print(f"Cropped Images: {results.total_crop_objects}")
