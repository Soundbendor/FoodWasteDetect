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

    monkeypatch.setattr("internvl.experiment.tqdm", RecordingBar)
    monkeypatch.setattr("internvl.generate_descriptors.tqdm", RecordingBar)
    return bars
