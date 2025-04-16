import argparse
import configparser

from database.embedding import EmbeddingModel
from database.vecdb import VectorDB

# TODO: take this shit, move it to a new module (or two)
# Make sure our LLM is online
# llm.start_server()
# dict_path = llm.generate_descriptors(ds)

# Fine-tune embedding model
# encoder.train()

# Test database with a query
query = "They are delicate, pastel-pink macarons featuring crisp almond-flour meringue shells sandwiched around a sweet, creamy filling."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument('config_file', help='Path to experiment config')
    return parser.parse_args()


def main(config_file: str):
    args = parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config_file)
    dict_path = cfg.get('Models', 'dictionary')
    embedder = EmbeddingModel(dict_path, cfg.get('Models', 'embed_model'))
    db = VectorDB(cfg.get('Models', 'db_path'), embedder)
    # Goal, from dictionary, build vector database
    print("INFO: Building Database...")
    db.add(dict_path)
    print("Database Built!")
    print(f"Test Query: {query}")
    q_vecs = db.query(query)
    prediction, _ = db.score(q_vecs, "Macaron", "voting")
    print(f"Prediction: {prediction}")

