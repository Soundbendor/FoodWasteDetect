import os
from pathlib import Path
from pprint import pprint
from typing import Callable

import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box


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
        yolo_draws = os.path.join(os.path.dirname(ds_path), "yolo-draws")
        yolo_labels = os.path.join(os.path.dirname(ds_path), "yolo-txts")
        os.makedirs(yolo_draws, exist_ok=True)
        os.makedirs(yolo_labels, exist_ok=True)

        # TODO: Pass batch of image paths to YOLO
        img_pths = [
            os.path.join(ds_path, "images", img_name) for img_name in subset["images"]
        ]
        # result should be list of generators, each one representing a new batch
        pred_generators = list(np.array_split(img_pths, 50))

        # TODO: Extract each patch given bounding box coords and save it locally.
        for batch in pred_generators:
            batch_results = self.model(list(batch), stream=True)
            for result in batch_results:
                # Save the annotated original image and text label.
                src_img = os.path.basename(result.path)
                basename = os.path.splitext(src_img)[0]
                result.save(filename=os.path.join(yolo_draws, f"{basename}.jpg"))
                result.save_txt(
                    txt_file=os.path.join(yolo_labels, f"{basename}.txt"),
                    save_conf=True,
                )
                # Save detections as separate cropped imgs
                for idx, box in enumerate(result.boxes):
                    # Save detection img in dets_pth
                    cname = result.names[int(box.cls)]
                    fpth = os.path.join(crop_dir, f"{basename}_crop_{idx}_{cname}.jpg")
                    save_one_box(
                        box.xyxy, result.orig_img.copy(), file=Path(fpth), BGR=True
                    )
                    record = {
                        "patch_name": os.path.basename(fpth),
                        "patch_pth": fpth,
                        "source_img": src_img,
                        "conf": float(box.conf),
                        "xyxy": box.xyxy.tolist(),
                        "class_id": int(box.cls),
                        "class_name": cname,
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
        df = pd.DataFrame.from_records(records)
        df_path = os.path.join(self.ds.root, subset_dirname, "dets.csv")
        df.to_csv(df_path)
