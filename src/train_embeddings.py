from database.embedding import EmbeddingModel
from util import parse_args, parse_cfg


def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)
    dict_path = cfg.get("Models", "dictionary")
    embedder = EmbeddingModel(dict_path, cfg.get("Models", "anchor_path"), cfg.get("Models", "embed_model"))
    # train embedding model
    trainer = embedder.train(save_path=cfg.get("Models", "embed_model_save_path"))
    metrics = trainer.state.log_history
    print(metrics)

if __name__ == '__main__':
    main()
    

