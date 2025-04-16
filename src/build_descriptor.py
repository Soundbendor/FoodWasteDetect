import argparse
import configparser

from ds.foodx251 import FoodX251
from llm.describer import DescriberLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intern-FW: Build Descriptor Dictionary"
    )
    parser.add_argument("config_file", help="Path to experiment config")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = configparser.ConfigParser()
    cfg.read(args.config_file)

    # start LLM
    llm = DescriberLLM(cfg.get("Models", "llm"))
    llm.start_server()

    # load dataset, get class map
    dataset = FoodX251(cfg.get("Models", "ds_path"))

    llm.generate_descriptors(dataset, cfg.get("Models", "dictionary"))
    print("Building description dictionary")
