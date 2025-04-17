import logging

from database.embedding import EmbeddingModel
from database.vecdb import VectorDB
from util import parse_args, parse_cfg

# Test database with a query
query = "They are delicate, pastel-pink macarons featuring crisp almond-flour meringue shells sandwiched around a sweet, creamy filling."

def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    dict_path = cfg.get("Models", "dictionary")
    embedder = EmbeddingModel(dict_path, cfg.get("Models", "embed_model"))
    db = VectorDB(
        cfg.get("Models", "db_path"), embedder, cfg.get("Models", "collection_name")
    )
    # Goal, from dictionary, build vector database
    logging.info("Building Database...")
    db.add(dict_path)
    logging.info("Database Built!")
    logging.info(f"Test Query: {query}")
    q_vecs = db.query(query)
    accuracy, _ = db.score(q_vecs, "Macaron", "voting")
    prediction = q_vecs[0].payload["class"]
    logging.info(f"Prediction: {prediction}")
