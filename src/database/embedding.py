import json

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import (
    SentenceTransformerTrainingArguments,
    BatchSamplers,
)
from sentence_transformers.losses import BatchAllTripletLoss


# goal:
# load dataset from JSON file, dictionary
# parse list of "label:, description: pairs"
# --- check ordering of columns
# import into Dataset
# finetune using BatchAll loss


class EmbeddingModel:

    def __init__(self, dictionary_path: str, model_path: str):
        self.ds = self._load_dataset(dictionary_path)
        self.model = SentenceTransformer(model_path)
        self.loss = BatchAllTripletLoss(self.model)

    def _load_dataset(self, path: str):
        # WARN: this doesn't work!
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        class_names = []
        descriptors = []
        for idx, (k, v) in enumerate(data.items()):
            for description in v:
                descriptors.append(description)
                class_names.append(idx)

        ds = {"label": class_names, "descriptions": descriptors}
        ds = Dataset.from_dict(ds)
        return ds

    def evaluator(self):
        # Our goal is to define some system to make sure descriptors of food categories are pushed away from one another
        # We need a subset of our dataset that's formatted as triplets?
        pass

    def get_embedding(self, txt: str):
        return self.model.encode(txt)

    def train(self):
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=self.set_config(),
            train_dataset=self.ds,
            loss=self.loss,
        )
        trainer.train()
        self.model.save_pretrained("assets/embed_models/mpnet-base-food/final")

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
            # Unsure whether to do this before or after InternV2.5 finetuning
            eval_strategy="no",
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=100,
        )
