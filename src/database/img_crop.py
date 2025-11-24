import os
from pprint import pprint
from typing import Callable

import cv2
import pandas as pd
from ultralytics import YOLO


class ImageCropper:
    # WARN: This is probably a bad interface, since Dataset depends
    # on ImageCropper and ImageCropper depends on Dataset.
    def __init__(self, dataset, model: str) -> None:
        # TODO: Convert get_class_list to return list, rather than dataframe
        self.ds = dataset
        self.classes = dataset.get_class_labels()
        self.model = YOLO(model)

    def crop_dataset(self, subset_dirname: str, subset_loader: Callable):
        # for img in dataset
        # load training set
        # iterate over each training folder
        # Given a subset
        subset = subset_loader()

        records = []
        ds_path = os.path.join(self.ds.root, subset_dirname)
        # Update path to save crops
        crop_dir = os.path.join(os.path.dirname(ds_path), "cropped-detections")

        # TODO: Pass batch of image paths to YOLO
        img_pths = [
            os.path.join(ds_path, "images", img_name) for img_name in subset["images"]
        ]
        results = self.model(img_pths, stream=True)

        # TODO: Extract each patch given bounding box coords and save it locally.
        for result in results:
            # Save the cropped predictions for figure creation.
            result.save(filename="")
            # Save the cropped detection
            # TODO: figure out what the names of these cropped detections are
            result.save_crop(crop_dir)
            for box in result.boxes:
                record = {
                    "patch_name": "TODO.jpg",
                    "source_img": result.path,
                    "conf": box.conf,
                    "xyxy": box.xyxy,
                    # TODO: should we save class ID or string?
                    "class_id": box.cls,
                }
                pprint(record)
                records.append(record)

        # prev_total_crops = 1
        # # WARN: Due to new changes, this interface is now incompatible with FoodX-251.
        # for img_name in subset["images"]:
        #     img_pth = os.path.join(ds_path, "images", img_name)
        #     # load image
        #     img = cv2.imread(img_pth)
        #     results = self.model.process(img)
        #     cv2.destroyAllWindows()
        #     del img
        #     # for each detection
        #     for crop_idx in range(prev_total_crops, self.model.crop_idx):
        #         record = {
        #             "patch_name": f"crop_{crop_idx}.jpg",
        #             "source_img": img_name,
        #         }
        #         print(record)
        #         records.append(record)
        #     prev_total_crops = self.model.crop_idx
        #     print(f"Cropped Images: {results.total_crop_objects}")
        # df = pd.DataFrame.from_records(records)
        df_path = os.path.join(self.ds.root, subset_dirname, "dets.csv")
        df.to_csv(df_path)
