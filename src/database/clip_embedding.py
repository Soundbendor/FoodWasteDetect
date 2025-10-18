from typing import List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class CLIPEmbedding:
    def __init__(self, model_name: str, save_dir: str, vec_dim: int):
        self.save_dir = save_dir
        self.model = SentenceTransformer(
            model_name, truncate_dim=vec_dim, trust_remote_code=True
        )

    def _check_img(self, img: Image) -> bool:
        return all(i >= 20 for i in img.size)

    def get_embedding(self, img_names: List[str]) -> torch.Tensor:
        return self.model.encode(list(filter(self._check_img, [Image.open(filepath) for filepath in img_names])))

    def get_text_embedding(self, labels: List[str]) -> torch.Tensor:
        return self.model.encode(labels)
