from sentence_transformers import SentenceTransformer

from datasets.foodx251 import FoodX251
from intern import InternVLM

ds_path = 'compost3.jpg'

# goal: 
# for every image in foodx-251
# read class label, image 

model = InternVLM("../../intern_fw_test/InternVL2_5-8B-MPO")
ds_path = "/nfs/guille/eecs_research/soundbendor/beerya/food_cap_datasets/FoodX-251"
dataset = FoodX251(ds_path)
prompt = '<image>\nPlease classify the food item in this image. Answer with a single item if possible.'
embed_model = SentenceTransformer("all-mpnet-base-v2")

test_set = dataset.val_set()
# only has class names

cos_mean = 0
for i, row in test_set.iterrows():
    response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
    response = response.replace('.', '')
    # TODO: pattern match on label from row['class'] to compute accuracy
    print(f"Item: {row['class']}")
    print(response)
    embeddings = embed_model.encode([row['class'], response])
    sim = embed_model.similarity(embeddings, embeddings)
    print(sim)
    cos_mean += sim[0, 1]

cos_mean = cos_mean / len(test_set)

print(f"Final Score: {cos_mean}")

