"""Shared encoder interface contracts without pretrained construction."""

import inspect
import subprocess
import sys

import pytest

from classification.encoders.base import Embedder
from classification.encoders.sentence_transformers import EmbeddingModel
from classification.encoders.vertex import VertexEmbeddingModel


@pytest.mark.parametrize("encoder", [EmbeddingModel, VertexEmbeddingModel])
def test_concrete_encoders_implement_interface(encoder):
    assert issubclass(encoder, Embedder)
    assert not inspect.isabstract(encoder)


def test_interface_import_does_not_load_backends_or_experiments():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from classification.encoders.base import Embedder
assert not any(name in sys.modules for name in (
    'classification.encoders.sentence_transformers', 'classification.encoders.vertex',
    'classification.experiments.classification', 'classification.vertex_utils',
    'torch', 'sentence_transformers', 'google.genai',
))
""",
        ],
        check=True,
    )
