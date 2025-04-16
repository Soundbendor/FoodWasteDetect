import argparse
import configparser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument('config_file', help='Path to experiment config')
    return parser.parse_args()


def main(config_file: str):
    pass

