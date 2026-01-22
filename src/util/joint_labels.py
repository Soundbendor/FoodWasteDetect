import functools
import os
import re
from functools import partial

import pandas as pd

ROOT_PTH = "/nfs/guille/eecs_research/soundbendor/beerya/food_datasets/combined_food_seg/labels/test"
food201_map = pd.read_csv("food201.csv", index_col=1).to_dict("index")
# load foodseg103 map
foodseg103_map = pd.read_csv("foodseg103_map.csv", index_col=1).to_dict("index")
# load uecfoodpix map
uec_map = pd.read_csv("uec_map.csv", index_col=1).to_dict("index")

dataset_csv_map = {"foodseg103": foodseg103_map, "uec": uec_map}


def change_id(line: str, id_map: dict) -> str:
    words = line.split()
    words[0] = str(id_map[int(words[0])]["newID"])
    return " ".join(words) + "\n"


for file in os.listdir(ROOT_PTH):
    fpath = os.path.join(ROOT_PTH, file)
    with open(fpath, "r") as rfile:
        lines = rfile.readlines()
    if re.search(r".*NAN", file):
        continue
    ds_name = file.split("_")[0]
    if ds_name in dataset_csv_map:
        id_mapper = partial(change_id, id_map=dataset_csv_map[ds_name])
    else:
        # food201 records do not have dataset prefix
        id_mapper = partial(change_id, id_map=food201_map)
    lines = list(map(id_mapper, lines))
    with open(fpath, "w") as wfile:
        wfile.writelines(lines)
