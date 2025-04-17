from ds.foodx251 import FoodX251
from llm.describer import DescriberLLM
from util import parse_args, parse_cfg


def main():
    args = parse_args()
    cfg = parse_cfg(args.config_file)

    # start LLM
    llm = DescriberLLM(cfg.get("Models", "llm"))
    llm.start_server()

    # load dataset, get class map
    dataset = FoodX251(cfg.get("Models", "ds_path"))

    llm.generate_descriptors(dataset, cfg.get("Models", "dictionary"))
