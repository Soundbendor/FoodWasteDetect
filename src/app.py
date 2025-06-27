import json
import logging
import os
from collections import defaultdict

import pandas as pd

from data_wrappers.foodx251 import FoodX251
from database.embedding import EmbeddingModel
from database.vecdb import VectorDB
from llm.describer import DescriberLLM
from util import EvalMetric, parse_args, parse_cfg
from vlm.intern import InternVLM

intern_prompt = "<image>\nPlease describe the food item in this image in a single sentence, focusing on the visual characteristics of the food."
test_query = "They are delicate, pastel-pink macarons featuring crisp almond-flour meringue shells sandwiched around a sweet, creamy filling."

'''
    From a dataset, extract all class labels
    For each of these class labels, generate descriptions with a LLM
'''
def generate_descriptions():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    llm = DescriberLLM(cfg['llm'])
    llm.start_server()
    dataset = FoodX251(cfg['paths']["dataset"])
    llm.generate_descriptors(dataset, cfg['paths']['dictionary'])

'''
    Evaluate VLM performance on open-vocabulary image classification using an embedding database
'''
def test_classification():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    model = InternVLM(cfg['paths']['intern'])
    dataset = FoodX251(cfg['paths']['dataset'])
    # TODO: there has to be a better way to handle save paths for the embedding model?
    embedder = EmbeddingModel(
        cfg['paths']['dictionary'], 
        cfg['paths']['anchor_path'],
        cfg['paths']['embed_model_save_path'],
        cfg['paths']['anchor_path']
    )
    db = VectorDB(
        cfg['qdrant_url'], cfg['collection_name'], cfg['reranker_model'], cfg['embed_size']
    )

    ds_path = cfg['paths']['dataset']
    val_set = dataset.val_set()
    metric = EvalMetric()

    n_samples = int(cfg['settings']['eval_samples'])
    if n_samples != 0:
        val_set = val_set.sample(n=n_samples, random_state=42)

    # TODO: Turn this into a .apply() function
    for i, row in val_set.iterrows():
        # Caption image with InternV2.5
        img_caption = model.infer(f"{ds_path}/val/val_set/{row['fname']}", intern_prompt)
        # Generate embedding from caption
        query_vec = embedder.get_embedding(img_caption)
        # Search vector database for most similar vector
        candidate_vecs = db.query(img_caption, query_vec)
        # Score our result
        # INFO: We use voting here as default
        score, confidence, prediction = db.score(candidate_vecs, row["class"], "voting")
        top5_score, _, _ = db.score(candidate_vecs, row["class"], "top5")
        metric.update_scores(candidate_vecs, db, row["class"])
        logging.info(f"Predicted Label: {prediction}")
        logging.info(f"True label: {row['class']}")
        logging.info(f"In Top 5? {top5_score}")
        logging.info(f"Description: {img_caption}")
        logging.info(f"Current Accuracies: {metric.compute_accuracies()}")

    scores = metric.compute_accuracies()
    print(scores)

'''
    Take a dataset and build a vector db using some pretrained embedding model
'''
def build_vdb():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    dict_path = cfg.get("Models", "dictionary")
    embedder = EmbeddingModel(
        dict_path, cfg.get("Models", "anchor_path"), cfg.get("Models", "embed_model_save_path"), cfg.get("Models", "embed_model")
    )
    db = VectorDB(
        cfg['qdrant_url'], cfg['collection_name'], cfg['reranker_model'], cfg['embed_size']
    )

    # load dataset
    with open(dict_path, "r", encoding="utf-8") as f:
        descriptors = json.load(f)

    logging.info("Building Database...")
    for k, v in descriptors.items():
        # generate embeddings
        vectors = embedder.get_embedding(v)
        metadata = {'description': v}
        db.add_records(k, vectors, metadata)

    logging.info("Database Built!")
    logging.info(f"Test Query: {test_query}")
    query_vec = embedder.get_embedding(test_query)
    candidate_vecs = db.query(test_query, query_vec)
    accuracy, conf, prediction = db.score(candidate_vecs, "Macaron", "voting")
    logging.info(f"Prediction: {prediction}")

'''
    Take split foodx251 captions, merge them to a single csv
'''
# WARN: Hard-coded to only work for foodx251
def merge_files():
    n_splits = 4
    base_path = 'assets/foodx251_captions_'
    dfs = []
    for n in range(n_splits):
        path = f"{base_path}{n+1}.txt"
        dfs.append(pd.read_csv(path, index_col = 'idx')[['class', 'caption']])
    for d in dfs:
        print(d)
    df = pd.concat(dfs)
    print(df)
    df.to_csv('assets/foodx251_captions.txt')

'''
    Convert foodx251 caption csv to dictionary JSON
    Used for converting zero-shot captions to embedding database
'''
# WARN: Hard-coded to only work for foodx251
def convert_csv_to_dictionary():
    df = pd.read_csv("assets/foodx251_captions.txt", index_col='idx')
    dataset = defaultdict(list)
    for idx, row in df.iterrows():
        dataset[row['class']].append(row['caption'])
    with open('foodx251_captions.json', 'w') as f:
        json.dump(dataset, f, indent=4)


'''
    Given a dataset of images, use InternV2.5 to generate zero-shot captions
'''
def caption_imgs():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    ds_path = cfg.get("Models", "ds_path")
    model = InternVLM(cfg.get("Models", "intern_path"))
    dataset = FoodX251(ds_path)
   
    save_path = f"assets/foodx251_captions_{args.partition}.txt"
    train_set = dataset.train_set()

    # partition training dataset
    train_set = train_set[int((args.partition - 1) * (0.25 * len(train_set))) : int(args.partition * (0.25 * len(train_set)))]
    print(len(train_set))

    # Check for a cache file
    if os.path.isfile(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns = ['class', 'caption', 'idx'])

    buffer = []
    for i, row in train_set.iterrows():
        # If we have already made a generation for this image, skip it.
        if i in df['idx']:
            logging.info("Skipping image...")
            continue
        response = model.infer(f"{ds_path}/train/train_set/{row['fname']}", intern_prompt)
        logging.info(response)
        out = {'idx': i, 'class': row['class'], 'caption': response}
        buffer.append(out)
        # save to disk every 20 iters
        if i % 20 == 0:
            df_update = pd.DataFrame.from_records(buffer)
            df = pd.concat([df, df_update])
            df.to_csv(save_path)
            buffer = []



'''
    Fine-tune an embedding model on food descriptions
'''
def finetune_embeddings():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    dict_path = cfg.get("Models", "dictionary")
    embedder = EmbeddingModel(
        dict_path, cfg.get("Models", "anchor_path"), cfg.get("Models", "embed_model_save_path"), cfg.get("Models", "embed_model")
    )
    # train embedding model
    trainer = embedder.train(save_path=cfg.get("Models", "embed_model_save_path"))
    metrics = trainer.state.log_history
    print(metrics)

