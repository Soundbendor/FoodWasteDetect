import argparse
import math
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Tuple

import requests
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# Expected behaviors:
# - Load datasets from remote sources
# - Organize dataset files according to YOLO standard
# - Load unifying class map
# - Modify label files to map dataset to unified class indices
# - Drop annotations from label files according to dropcols from unifying map
# - Unify directories
# - Allow specification of which datasets to run for the experiment
# - Load from cached files
# - Train-test-validation splits

DS_DIR = Path("/nfs/guille/eecs_research/soundbendor/beerya/food_datasets")


class DataLoader:
    def __init__(self):
        pass

    def get_dataset(self, ds_name: str):
        match ds_name:
            case 'foodseg103':
                self._download_foodseg103()
            case 'food201':
                self._download_food201()
            case 'uecfoodpix':
                self._download_uecfoodpix()
            case _:
                print(f"Dataset {ds_name} not found!")

    # Download large file and return path to saved file
    def _stream_download(self, url: str, ds_name: str, ext: str) -> Path:
        pth = DS_DIR.joinpath(f"{ds_name}.{ext}")
        # Check to see if dataset already stored locally
        if not self._checkpath(pth):
            print(f"Downloading {ds_name} from remote, please wait...")
            with requests.get(url, stream=True) as response:
                chunk_size = 10 * 1024
                n_chunks = math.ceil(int(response.headers.get("content-length", 0)) / chunk_size)
                with open(pth, "wb") as f:
                    for chunk in tqdm(response.iter_content(chunk_size=10 * 1024), total=n_chunks):
                        f.write(chunk)
        else:
            print(f"Loading {ds_name} from cache")
        return pth

    def _extract_tar(self, tar_path: Path, ds_name: str) -> Path:
        pth = DS_DIR.joinpath(ds_name)
        if not self._checkpath(pth):
            with tarfile.open(tar_path) as tarball:
                tarball.extractall(path=pth)
        return pth

    def _extract_zip(self, zip_path: Path, ds_name: str, pw: str = "") -> Path:
        pth = DS_DIR.joinpath(ds_name)
        if not self._checkpath(pth):
            with zipfile.ZipFile(zip_path) as zip:
                zip.extractall(path=pth, pwd = pw)
        return pth


    def _checkpath(self, pth: Path):
        try:
            os.makedirs(pth.parent)
        except OSError:
            pass
        if os.path.isfile(pth):
            return True
        if os.path.isdir(pth):
            if os.listdir(pth):
                return True
        return False

    def _makedir(self, parent: Path, dirname: str) -> Tuple[Path, bool]:
        newdir = parent.joinpath(dirname)
        newdir.mkdir(exist_ok=True)
        if any(newdir.iterdir()):
            return (newdir, True)
        # if empty, return false
        return (newdir, False)

    def _download_food101(self) -> Path:
        url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        tar_path = self._stream_download(url, "food101", "tar.gz")
        ds_path = self._extract_tar(tar_path, "food101/")
        return ds_path

    def _download_uecfoodpix(self) -> Path:
        url = "https://mm.cs.uec.ac.jp/uecfoodpix/UECFOODPIXCOMPLETE.tar"
        tar_path = self._stream_download(url, "uecfoodpix", "tar")
        ds_path = self._extract_tar(tar_path, "uecfoodpix/")
        splits = ['test/', 'train/']
        data_path = ds_path.joinpath('UECFOODPIXCOMPLETE/data/UECFoodPIXCOMPLETE/')
        for s in splits:
            subset_pth = data_path.joinpath(s)
            labels_pth, is_labels = self._makedir(subset_pth, 'labels/')
            if not is_labels:
                convert_segment_masks_to_yolo_seg(subset_pth.joinpath('mask'), labels_pth, 103)
            else:
                print("INFO: Labels directory populated, skipping...")
        return ds_path

    def _download_foodseg103(self):
        # we know that FoodSeg103 will always yield a DatasetDict, so we explicitly declare it to make pyright happy
        ds: DatasetDict = load_dataset("EduardoPacheco/FoodSeg103", cache_dir=str(DS_DIR)) #type: ignore
        # make new directory structure
        masks_pth, _ = self._makedir(DS_DIR, "foodseg103/masks/")
        images_pth, is_imgs = self._makedir(DS_DIR, "foodseg103/images/")
        labels_pth, is_labels = self._makedir(DS_DIR, "foodseg103/labels/")
        # dataset has train/validation split
        for split in ds:
            img_split_pth, _ = self._makedir(images_pth, f'{split}/')
            labels_split_pth, _ = self._makedir(labels_pth, f'{split}/')
            mask_split_pth, _ = self._makedir(masks_pth, f'{split}/')
            # if we have not already populated the images directory...
            if not is_imgs:
                # for every sample, we extract the image and pixel mask (label) and write it to disk
                # forward declared to resolve typechecking issues
                sample: dict
                print(f"INFO: Saving {split} set")
                for sample in tqdm(ds[split]): # type: ignore
                    # print(f"DEBUG: Saving image {split}/{sample['id']}.jpg")
                    sample['image'].save(img_split_pth.joinpath(f'{sample['id']}.jpg'))
                    sample['label'].save(mask_split_pth.joinpath(f'{sample['id']}.png'))
            else:
                print("INFO: Images exist, skipping write to disk...")
            if not is_labels:
                convert_segment_masks_to_yolo_seg(mask_split_pth, labels_split_pth, classes=104)
            else:
                print("INFO: YOLO Labels exist, skipping conversion")

    def _download_food201(self):
        url = "https://storage.googleapis.com/food201/food201.zip"
        # first, ensure we have food101
        food101_pth = self._download_food101()
        zip_path = self._stream_download(url, "food201", "zip")
        ds_path = self._extract_zip(zip_path, "food201/")

        img_pth, is_imgs = self._makedir(ds_path, "images/")
        masks_pth, is_masks = self._makedir(ds_path, "masks/")
        labels_pth, is_labels = self._makedir(ds_path, 'labels/')
        pixel_classes = ds_path.joinpath("food201/pixel_annotations/")

        # Flatten the pixel masks into a single directory
        # so that we can hand these labels off to YOLO for conversion later
        # before: pixel_masks/classname/id.png
        # after: masks/classname-id.png
        if not is_masks:
            for root, _, files in pixel_classes.walk():
                for file in files:
                    # pixel_masks/classname_basename.png
                    fpath = root.joinpath(file)
                    target_path = masks_pth.joinpath(f"{fpath.parent.name}-{file}")
                    if os.path.exists(target_path):
                        raise Exception("WARN: Path collision detected or conversion has already been done!")
                    # Copy file from fpath to target_path
                    shutil.copy(fpath, target_path)
                    # print(f"DEBUG: Copying {fpath} to {target_path}")
        else:
            print("INFO: Masks directory already flattened, skipping...")

        # From here, we should have masks/{class}_{ID}.jpg
        # Pass this directory to YOLO pixel mask conversion
        if not is_labels:
            convert_segment_masks_to_yolo_seg(masks_pth, labels_pth, classes=208)
        else:
            print("INFO: Labels already exist, skipping...")

        # Copy images from Food101 into Food201
        food101_pth = food101_pth.joinpath("food-101/images/")
        if not is_imgs:
            print("INFO: Copying images from food-101 into food-201")
            for file in tqdm(masks_pth.iterdir()):
                if file.is_file():
                    # split the path
                    class_name, id = file.name.split('-')
                    id = os.path.splitext(id)[0]
                    target_path = food101_pth.joinpath(f"{class_name}/{id}.jpg")
                    new_path = img_pth.joinpath(f"{class_name}-{id}.jpg")
                    shutil.copyfile(target_path, new_path)
        else:
            print("INFO: Images already copied from Food101, skipping...")


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', help="Name of the dataset to download.")
    return parser.parse_args()


def main():
    args = parseargs()
    dl = DataLoader()
    dl.get_dataset(args.ds)


if __name__ == "__main__":
    main()
