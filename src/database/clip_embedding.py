from typing import List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class CLIPEmbedding():
    def __init__(self, dataset, model_name: str, save_dir: str):
        self.ds = dataset
        self.save_dir = save_dir
        self.model = SentenceTransformer(model_name)


    def get_embedding(self, img_names: List[str]) -> torch.Tensor:
        return self.model.encode([Image.open(filepath) for filepath in img_names])

