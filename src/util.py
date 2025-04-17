import argparse
import configparser

import pandas as pd
from qdrant_client.models import ScoredPoint


class EvalMetric:
    def __init__(self):
        self.metrics = ["top1", "top5", "voting"]
        self.scores = pd.Series([0, 0, 0], index=self.metrics)
        self.len = 0

    def update_scores(self, q_vecs: list[ScoredPoint], db: VectorDB, label: str):
        self.scores = self.scores.add(
            [db.score(q_vecs, label, strat)[0] for strat in self.metrics]
        )
        self.len += 1

    def compute_accuracies(self):
        return self.scores / self.len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intern-FW Experiment Pipeline")
    parser.add_argument("config_file", help="Path to experiment config")
    return parser.parse_args()

def parse_cfg(cfg_file: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file)
    return cfg
