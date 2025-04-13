from db.vecdb import VectorDB
from db.embedding import EmbeddingModel
from datasets.foodx251 import FoodX251
from llm.describer import DescriberLLM

embedding_model = "all-mpnet-base-v2"
dict_path = "assets/descriptor_dictionary.json"
db = VectorDB("db/qdrant")
llm = DescriberLLM()
encoder = EmbeddingModel(dict_path)
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
