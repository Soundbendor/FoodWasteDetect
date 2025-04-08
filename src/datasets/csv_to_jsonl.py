import json
import pandas as pd
from PIL import Image

def naturalize_class_labels(row: str) -> str:
    return row.capitalize().replace('_', ' ')

# Load class maps
cmap = pd.read_csv("annot/class_list.txt", delim_whitespace=True, index_col=0, names=['id', 'label'])
cmap['label'] = cmap['label'].apply(naturalize_class_labels)
print(cmap)

# For every label in class map, change it to a natural language format.


# Load training data
train_data = pd.read_csv("annot/train_info.csv", names=['fname', 'id'])
print(train_data)

records = []
for i, row in train_data.iterrows():
    print(row)
    # Get width and height of image
    pth = f"train/train_set/{row['fname']}"
    img = Image.open(pth)
    w, h = img.size
    d = {
            "id": i,
            "image": pth,
            "width": w,
            "height": h,
            "conversations":[
                {"from": "human", "value": "<image>\nPlease describe the food items present in this image."},
                {"from": "gpt", "value": cmap.loc[row['id']]['label']},
            ]
        }
    records.append(d)


with open('foodx_251_train.jsonl', 'w') as fout:
    json.dump(records, fout)
