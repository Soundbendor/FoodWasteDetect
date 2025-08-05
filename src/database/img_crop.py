import os

import pandas as pd
from ultralytics import solutions

from data_wrappers.dataset import Dataset


class ImageCropper:
    def __init__(self, dataset: Dataset, model: str) -> None:
        # TODO: Convert get_class_list to return list, rather than dataframe
        self.ds = dataset
        self.classes = dataset.get_class_list()
        self.model = solutions.ObjectCropper(model=model)

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

            # WARN: Problem, how do we match labels to classifications?
            # Rather than do detections, do we "warm start" using known bounding boxes
            df = pd.DataFrame(columns=["patch_name", "label", "source_img"])
            ds_path = os.path.join(self.ds.root, subset_dirname)
            # Update path to save crops
            self.model.crop_dir = os.path.join(ds_path, "cropped-detections")
            for img_name in subset["fname"]:
                img_pth = os.path.join(ds_path, img_name)
                results = self.model.process(img_pth)
                print(f"Cropped Images: {results.total_crop_objects}")
