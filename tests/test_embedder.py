"""Shared encoder interface contracts without pretrained construction."""

import inspect
import subprocess
import sys

import pytest

from internvl.embedder import Embedder, ImageInput
from internvl.embeddings import EmbeddingModel
from internvl.embeddings import ImageInput as ExportedImageInput
from internvl.vertex_embeddings import VertexEmbeddingModel


@pytest.mark.parametrize("implemented", [(), ("encode_text",), ("encode_images",)])
def test_incomplete_encoders_cannot_be_instantiated(implemented):
    methods = {name: lambda self, inputs: None for name in implemented}
    incomplete = type("Incomplete", (Embedder,), methods)
    with pytest.raises(TypeError, match="abstract"):
        incomplete()


def test_base_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        Embedder()


@pytest.mark.parametrize("encoder", [EmbeddingModel, VertexEmbeddingModel])
def test_concrete_encoders_implement_interface(encoder):
    assert issubclass(encoder, Embedder)
    assert not inspect.isabstract(encoder)


def test_image_input_reexport():
    assert ExportedImageInput is ImageInput


def test_interface_import_does_not_load_backends_or_experiments():
    subprocess.run(
        [sys.executable, "-c", """
import sys
from internvl.embedder import Embedder
assert not any(name in sys.modules for name in (
    'internvl.embeddings', 'internvl.vertex_embeddings',
    'internvl.experiment', 'internvl.vertex_utils',
    'torch', 'sentence_transformers', 'google.genai',
))
"""],
        check=True,
    )
