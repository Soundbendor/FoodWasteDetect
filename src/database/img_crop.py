import os
from typing import Callable

import pandas as pd
import cv2
from ultralytics import solutions



class ImageCropper:
    # WARN: This is probably a bad interface, since Dataset depends 
    # on ImageCropper and ImageCropper depends on Dataset.
    def __init__(self, dataset, model: str) -> None:
        # TODO: Convert get_class_list to return list, rather than dataframe
        self.ds = dataset
        self.classes = dataset.get_class_list()
        self.model = solutions.ObjectCropper(model=model)

    def crop_dataset(self):
        # for img in dataset
        # load training set
        # iterate over each training folder
        for subset_dirname, dataloader in self.ds.get_subsets():
            # WARN: subset_dirname points to 
            try:
                subset = dataloader()
            except NotImplementedError:
                print(f"Error: Subset {dataloader} not implemented for dataset!")
                continue

            records = []
            ds_path = os.path.join(self.ds.root, subset_dirname)
            # Update path to save crops
            self.model.crop_dir = os.path.join(os.path.dirname(ds_path), "cropped-detections")
            # WARN: Depends on subset (e.g. val_set) returning ['fname', 'class']
            for idx, (img_name, class_label) in subset.iterrows():
                img_pth = os.path.join(ds_path, img_name)
                # load image
                img = cv2.imread(img_pth)
                results = self.model.process(img)
                init_crop_idx = self.model.crop_idx = results.total_crop_objects
                # for each detection
                for crop_idx in range(init_crop_idx, self.model.crop_idx):
                    records.append({'patch_name': f"crop_{self.model.crop_idx}.jpg", 
                                    "label": class_label,
                                    "source_img": img_name})
                print(f"Cropped Images: {results.total_crop_objects}")
            df = pd.DataFrame.from_records(records)
            df_path = os.path.join(self.ds.root, subset_dirname, "dets.csv")
            df.to_csv(df_path)
