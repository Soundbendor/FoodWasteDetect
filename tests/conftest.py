"""Shared synthetic progress-bar inspection."""

import io

import pytest


@pytest.fixture
def progress_bars(monkeypatch):
    from tqdm import tqdm

    bars = []

    class RecordingBar(tqdm):
        def __init__(self, *args, **kwargs):
            self.updates = []
            super().__init__(*args, file=io.StringIO(), **kwargs)
            bars.append(self)

        def update(self, n=1):
            self.updates.append(n)
            return super().update(n)

    monkeypatch.setattr("classification.experiments.evaluation.tqdm", RecordingBar)
    monkeypatch.setattr("classification.experiments.index_builder.tqdm", RecordingBar)
    monkeypatch.setattr("classification.llm.generate_descriptors.tqdm", RecordingBar)
    return bars
