import logging
import os

import pandas as pd

from ds.foodx251 import FoodX251
from util import parse_args, parse_cfg
from vlm.intern import InternVLM

prompt = "<image>\nPlease describe the food item in this image in a single sentence, focusing on the visual characteristics of the food."

def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg.get("Models", "ds_path")
    model = InternVLM(cfg.get("Models", "intern_path"))
    dataset = FoodX251(ds_path)
   
    save_path = f"assets/foodx251_captions_{args.partition}.txt"
    train_set = dataset.train_set()

    # partition training dataset
    train_set = train_set[(args.partition - 1) * len(train_set) : args.partition * len(train_set)]
    print(len(train_set))

    # Check for a cache file
    if os.path.isfile(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns = ['class', 'caption', 'idx'])

    buffer = []
    for i, row in train_set.iterrows():
        # If we have already made a generation for this image, skip it.
        if i in df['idx']:
            logging.info("Skipping image...")
            continue
        response = model.infer(f"{ds_path}/train/train_set/{row['fname']}", prompt)
        logging.info(response)
        out = {'idx': i, 'class': row['class'], 'caption': response}
        buffer.append(out)
        # save to disk every 20 iters
        if i % 20 == 0:
            df_update = pd.DataFrame.from_records(buffer)
            df = pd.concat([df, df_update])
            df.to_csv(save_path)
            buffer = []

