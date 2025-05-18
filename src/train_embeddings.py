from database.embedding import EmbeddingModel
from util import parse_args, parse_cfg


def main(config_file: str):
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    dict_path = cfg.get("Models", "dictionary")
    embedder = EmbeddingModel(dict_path, cfg.get("Models", "embed_model"))
    # train embedding model
    embedder.train(save_path=cfg.get("Models", "embed_model_save_path"))
    

