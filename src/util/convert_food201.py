import os

import pandas as pd


def main():
    root_pth = "/nfs/guille/eecs_research/soundbendor/beerya/food_datasets/food201/"
    origin_label_key = pd.read_csv(
        os.path.join(root_pth, "food201/pixel_annotations_map.csv"),
        names=["id", "label"],
        header=None,
    )
    normalized_label_key = pd.read_csv(
        os.path.join(root_pth, "food201/multilabel_map.csv"),
        names=["id", "label"],
        header=None,
    )
    # [test, train]
    conversion_key = dict(zip(origin_label_key["id"], normalized_label_key["id"]))
    for dir in os.listdir(os.path.join(root_pth, "data")):
        # [images, ...]
        folder_pth = os.path.join(root_pth, "data", dir, "boxes")
        for box_pth in os.listdir(folder_pth)[0:2]:
            print(box_pth)
            img_pth = os.path.join(folder_pth, box_pth)
            # modify bounding box
            data = []
            with open(img_pth, "r") as f:
                for line in f:
                    line_arr = f.split()
                    line_arr[0] = conversion_key[line_arr[0]]
                    data.append(" ".join(line_arr))
            with open(img_pth, "w") as f:
                f.writelines(data)


if __name__ == "__main__":
    main()
