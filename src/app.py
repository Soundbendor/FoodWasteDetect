"""Image-embedding experiments; whole-image experiments live in internvl.experiment."""

import logging

from data_wrappers.foodx251 import FoodX251
from database.clip_embedding import CLIPEmbedding
from database.vecdb import VectorDB
from util import EvalMetric, parse_args, parse_cfg


def eval_img_vdb():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(ds_path)
    embedder = CLIPEmbedding(
        ds, cfg["embed_model"], cfg["paths"]["embed_model_save_path"], cfg["embed_size"]
    )
    db = VectorDB(cfg["qdrant_url"], cfg["collection_name"], None, cfg["embed_size"])

    val_set = ds.val_set()
    metric = EvalMetric()

    n_samples = int(cfg["settings"]["eval_samples"])
    if n_samples != 0:
        val_set = val_set.sample(n=n_samples, random_state=42)

    for i, row in val_set.iterrows():
        # Generate embedding of test image
        query_vec = embedder.get_embedding([f"{ds_path}/val/val_set/{row['fname']}"])[0]
        candidate_vecs = db.query(None, query_vec)

        score, confidence, prediction = db.score(candidate_vecs, row["class"], "voting")
        top5_score, _, _ = db.score(candidate_vecs, row["class"], "top5")
        metric.update_scores(candidate_vecs, db, row["class"])
        logging.info(f"Predicted Label: {prediction}")
        logging.info(f"True label: {row['class']}")
        logging.info(f"In Top 5? {top5_score}")
        logging.info(f"Current Accuracies: {metric.compute_accuracies()}")

    scores = metric.compute_accuracies()
    print(scores)


def build_img_vdb():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg["paths"]["dataset"]
    ds = FoodX251(ds_path)
    embedder = CLIPEmbedding(
        ds, cfg["embed_model"], cfg["paths"]["embed_model_save_path"], cfg["embed_size"]
    )
    db = VectorDB(
        cfg["qdrant_url"],
        cfg["collection_name"],
        cfg["reranker_model"],
        cfg["embed_size"],
    )

    food_imgs = ds.train_set().groupby(by="class")
    # Generate CLIP embeddings
    logging.info("Building Database...")
    for food_name, img_names in food_imgs:
        img_paths = [f"{ds_path}/train/train_set/{x}" for x in img_names["fname"]]
        vectors = embedder.get_embedding(img_paths)
        metadata = {"img_path": img_names["fname"]}
        db.add_records(food_name, vectors, metadata)
