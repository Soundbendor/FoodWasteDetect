from db.vecdb import VectorDB
from datasets.foodx251 import FoodX251
from llm.describer import Describer

db = VectorDB("db/qdrant")
llm = Describer()
ds = FoodX251("/nfs/stak/users/beerya/soundbendor/food_cap_datasets/FoodX-251")

# Make sure our LLM is online
llm.start_server()
dict_path = llm.generate_descriptors(ds)
db.add(dict_path)

# Test database with a query
query = "They are delicate, pastel-pink macarons featuring crisp almond-flour meringue shells sandwiched around a sweet, creamy filling."
category, confidence = db.query(query)
print(f"Food Class: {category}")
