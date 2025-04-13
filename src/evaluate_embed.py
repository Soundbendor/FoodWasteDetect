from ds.foodx251 import FoodX251
from intern import InternVLM
from database.vecdb import VectorDB

# goal:
# for every image in foodx-251
# read class label, image

model = InternVLM("../intern_fw_test/InternVL2_5-8B-MPO")
ds_path = "/nfs/guille/eecs_research/soundbendor/beerya/food_cap_datasets/FoodX-251"
dataset = FoodX251(ds_path)
prompt = "<image>\nPlease describe the food item in this image in a single sentence, focusing on the visual characteristics of the food."
db = VectorDB("db/qdrant", "assets/embed_models/mpnet-base-food/final")

test_set = dataset.val_set()
# only has class names

acc = 0
for i, row in test_set.iterrows():
    response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
    category, confidence, q_response = db.query(response, "voting")
    print(f"\nItem: {category}")
    print(f"Description: {response}")
    print(f"True label: {row['class']}")
    print(f"DEBUG: {q_response}")
    if row["class"] == category:
        acc += 1

acc = acc / len(test_set)

print(f"Final Score: {acc}")
