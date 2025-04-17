import logging

from database.embedding import EmbeddingModel
from database.vecdb import VectorDB
from ds.foodx251 import FoodX251
from util import EvalMetric, parse_args, parse_cfg
from vlm.intern import InternVLM

prompt = "<image>\nPlease describe the food item in this image in a single sentence, focusing on the visual characteristics of the food."


def main():
    # Step 1: parse model config
    # Step 2: set up model, ds, db, etc.
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg.get("Models", "ds_path")
    model = InternVLM(cfg.get("Models", "intern_path"))
    dataset = FoodX251(ds_path)
    embedder = EmbeddingModel(
        cfg.get("Models", "dictionary"), cfg.get("Models", "embed_model_save_path")
    )
    db = VectorDB(
        cfg.get("Models", "db_path"), embedder, cfg.get("Models", "collection_name")
    )

    val_set = dataset.val_set()
    metric = EvalMetric()

    n_samples = int(cfg.get("Settings", "n_samples"))
    if n_samples != 0:
        val_set = val_set.sample(n=n_samples, random_state=42)

    # TODO: Turn this into a .apply() function
    for i, row in val_set.iterrows():
        response = model.infer(f"{ds_path}/val/val_set/{row['fname']}", prompt)
        q_vecs = db.query(response)
        # INFO: We use voting here as default
        score, confidence, prediction = db.score(q_vecs, row["class"], "voting")
        top5_score, _, _ = db.score(q_vecs, row["class"], "top5")
        metric.update_scores(q_vecs, db, row["class"])
        logging.info(f"Predicted Label: {prediction}")
        logging.info(f"True label: {row['class']}")
        logging.info(f"In Top 5? {top5_score}")
        logging.info(f"Description: {response}")
        logging.info(f"Current Accuracies: {metric.compute_accuracies()}")

    scores = metric.compute_accuracies()
    print(scores)
