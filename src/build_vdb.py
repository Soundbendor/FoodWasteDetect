from database.vecdb import VectorDB
from database.embedding import EmbeddingModel
from ds.foodx251 import FoodX251
from llm.describer import DescriberLLM

embedding_model = "all-mpnet-base-v2"
dict_path = "assets/descriptor_dictionary.json"
encoder = EmbeddingModel(dict_path, model_path)
db = VectorDB("db/qdrant", encoder)
llm = DescriberLLM()
ds = FoodX251("/nfs/stak/users/beerya/soundbendor/food_cap_datasets/FoodX-251")

# Make sure our LLM is online
# llm.start_server()
# dict_path = llm.generate_descriptors(ds)

# Fine-tune embedding model
encoder.train()

# Update vector database using new embedding model
db.add(dict_path)

# Test database with a query
query = "They are delicate, pastel-pink macarons featuring crisp almond-flour meringue shells sandwiched around a sweet, creamy filling."
category, confidence, q_vecs = db.query(query, "voting")
print(f"Food Class: {category}")
