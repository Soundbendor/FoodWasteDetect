import os
import shutil

import pandas as pd


def main():
    # load food201
    ds_path = "/nfs/guille/eecs_research/soundbendor/beerya/food_datasets/food201"
    # load class map
    # INFO: not a csv
    test_df = pd.read_csv(
        os.path.join(ds_path, "food201", "test_pixel_annotations.txt")
    )
    train_df = pd.read_csv(
        os.path.join(ds_path, "food201", "train_pixel_annotations.txt")
    )
    test_dir = os.path.join(ds_path, "test")
    train_dir = os.path.join(ds_path, "train")
    folders = ["masks", "images", "labels"]
    for dir in folders:
        os.makedirs(os.path.join(test_dir, dir), exist_ok=True)
        os.makedirs(os.path.join(train_dir, dir), exist_ok=True)
    # for each file in list
    # format is test_pixel_annotations/food_type/id_num
    # target format is food201/(data_type)/food_type - id_num . ext
    # extension is .jpg for images, .png for masks, .txt for labels
    # target directories are food201/images, food201/labels, food201/masks

    # Read each file path from dataframe
    for dest_folder, dest_df in zip([train_dir, test_dir], [train_df, test_df]):
        for record in dest_df:
            # train records are in format food_type / id_num
            # test records are in format "test_pixel_annotations" / food_type / id_num
            food_type, id_num = record.split("/")[-2:]
            for file_type in folders:
                record_basename = f"{food_type}-{id_num}"
                # move images
                shutil.copy(
                    os.path.join(ds_path, file_type, record_basename),
                    os.path.join(dest_folder, file_type),
                )


if __name__ == "__main__":
    main()
