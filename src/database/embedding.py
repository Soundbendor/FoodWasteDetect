import json
import logging

import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.evaluation import SimilarityFunction, TripletEvaluator
from sentence_transformers.losses import BatchAllTripletLoss
from sentence_transformers.training_args import (
    BatchSamplers,
    SentenceTransformerTrainingArguments,
)

# goal:
# load dataset from JSON file, dictionary
# parse list of "label:, description: pairs"
# --- check ordering of columns
# import into Dataset
# finetune using BatchAll loss


class EmbeddingModel:

    def __init__(self, dictionary_path: str, anchor_path: str,  model_path: str):
        self.ds, self.class_map = self._load_dataset(dictionary_path)
        self.eval_ds = self._load_triplet_dataset(anchor_path)
        self.model = SentenceTransformer(model_path)
        self.loss = BatchAllTripletLoss(self.model)
        self.eval_func = self.evaluator()

    def _load_dataset(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logging.info(data)
        class_names = []
        descriptors = []
        class_map = {}
        for idx, (k, v) in enumerate(data.items()):
            class_map[k] = idx
            for description in v:
                descriptors.append(description)
                class_names.append(idx)

        # where image captions from InternV2.5 are anchors
        # positive/negatives sourced from gemma
        ds = {"label": class_names, "descriptions": descriptors}
        ds = Dataset.from_dict(ds)
        # logging.info(f"Dataset: {ds}")
        return ds, class_map


    # goal: load dataset in (anchor, positive, negative) pairs
    # relies on _load_dataset
    # should be Dataset class with
    def _load_triplet_dataset(self, path: str):
        # load anchor captions
        df = pd.read_csv(path)
        # has [class], [caption]
        # sort to aggregate each anchor class
        anchors_by_class = df.groupby(['class'])
        # convert descriptions dataset to pandas
        descriptions = self.ds.to_pandas()
        
        # for each anchor
        # randomly select positive and negative?
        dfs = []
        for label, anchors in anchors_by_class:
            # extract all matching positive samples
            class_idx = self.class_map[label[0]]
            positives = descriptions[descriptions['label'] == class_idx].sample(n=len(anchors), replace=True)
            negatives = descriptions[descriptions['label'] != class_idx].sample(n=len(anchors), replace=False)
            print(len(positives))
            # merge all 3 of these?
            triplets = pd.concat(anchors, positives, negatives, axis=1)
            print(triplets)


    def evaluator(self):
        # Our goal is to define some system to make sure descriptors of food categories are pushed away from one another
        # We need a subset of our dataset that's formatted as triplets?
        return TripletEvaluator(
            anchors=self.eval_ds["anchor"],
            positives=self.eval_ds["positive"],
            negatives=self.eval_ds["negative"],
            main_distance_function=SimilarityFunction.COSINE,
            name="all-nli-dev",
        )

    def get_embedding(self, txt: str):
        return self.model.encode(txt)

    def train(self, save_path: str):
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.set_config(),
            train_dataset=self.ds,
            loss=self.loss,
        )
        trainer.train()
        self.model.save_pretrained(save_path)

    def set_config(self):
        return SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir="models/all-mpnet-base-v2",
            # Optional training parameters:
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
            # Optional tracking/debugging parameters:
            # WARN: eventually, we want to define an evaluation function using a test dataset
            # We will want to generate Internv2.5 captions from foodx251 training data and form triplets from this data
            # Then, use triplet loss evaluation
            eval_strategy="no",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=100,
        )
