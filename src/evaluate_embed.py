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

top1_acc = 0
top5_acc = 0
top1_vote_acc = 0
for i, row in test_set.iterrows():
    response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
    q_vecs = db.query(response)
    prediction, vote_score = db.score(q_vecs, row["class"], "voting")
    top1_prediction, top1_score = db.score(q_vecs, row["class"], "top1")
    top5_prediction, top5_score = db.score(q_vecs, row["class"], "top5")
    print(f"Predicted Label: {prediction}")
    print(f"Description: {response}")
    print(f"True label: {row['class']}")
    top1_acc += top1_score
    top5_acc += top5_score
    top1_vote_acc += vote_score

top1_acc = top1_acc / len(test_set)
lop5_acc = top5_acc / len(test_set)
top1_vote_acc = top1_vote_acc / len(test_set)
print(f"Final Score (Top 1): {top1_acc}")
print(f"Final Score (Top 5): {top5_acc}")
print(f"Final Score (Voting Top 1): {top1_vote_acc}")
