import os

import pandas as pd


def main():
    # load food201
    ds_path = "/nfs/guille/eecs_research/soundbendor/beerya/food_datasets/food201"
    # load class map
    # INFO: not a csv
    test_df = pd.read_csv(
        os.path.join(ds_path, "food201", "test_pixel_annotations.txt")
    )
    # for each file in list
    # format is
