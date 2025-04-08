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

test_set = dataset.val_set()
# only has class names

accuracy = 0
for i, row in test_set.iterrows():
    response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
    # TODO: pattern match on label from row['class'] to compute accuracy
    print(f"Item: {row['class']}")
    print(response)
    if row['fname'] in response:
        accuracy += 1

accuracy = accuracy / len(test_set)

