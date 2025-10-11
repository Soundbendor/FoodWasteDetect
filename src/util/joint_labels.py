import functools
import os
import re
from functools import partial

import pandas as pd

ROOT_PTH = "/nfs/guille/eecs_research/soundbendor/beerya/food_datasets/combined_food_seg/labels/test"
# load foodseg103 map
foodseg103_map = pd.read_csv("foodseg103_map.csv").to_dict("index")
# load uecfoodpix map
uec_map = pd.read_csv("uec_map.csv").to_dict("index")


def change_id(line: str, id_map: pd.DataFrame) -> str:
    words = line.split()
    words[0] = id_map[words[0]]["new_id"]
    return " ".join(words)


for file in os.listdir(ROOT_PTH):
    fpath = os.path.join(ROOT_PTH, file)
    with open(fpath, "r") as rfile:
        lines = rfile.readlines()
    if re.match(r"foodseg103_.*", file):
        id_mapper = partial(change_id, id_map=foodseg103_map)
    if re.match(r"uec_.*", file):
        id_mapper = partial(change_id, id_map=uec_map)
    lines = list(map(lines, id_mapper))
    with open(fpath, "w") as wfile:
        wfile.writelines(lines)
