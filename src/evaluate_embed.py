from datasets.foodx251 import FoodX251
from intern import InternVLM
from db.vecdb import VectorDB

ds_path = "compost3.jpg"

# goal:
# for every image in foodx-251
# read class label, image

model = InternVLM("../../intern_fw_test/InternVL2_5-8B-MPO")
ds_path = "/nfs/guille/eecs_research/soundbendor/beerya/food_cap_datasets/FoodX-251"
dataset = FoodX251(ds_path)
prompt = "<image>\nPlease classify the food item in this image. Answer with a single item if possible."
db = VectorDB("db/qdrant")

test_set = dataset.val_set()
# only has class names

acc = 0
for i, row in test_set.iterrows():
    response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
    category, confidence = db.query(response)
    print(f"Item: {category}, Confidence: {confidence}")
    print(f"Description: {response}")
    print(f"True label: {row['class']}")
    if row["class"] == category:
        acc += 1

acc = acc / len(test_set)

print(f"Final Score: {acc}")
