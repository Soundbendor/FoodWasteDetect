"""Synthetic embedding contracts, independent of datasets and real models."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

if any(
    importlib.util.find_spec(name) is None
    for name in ("numpy", "PIL", "sentence_transformers")
):
    raise unittest.SkipTest("requires NumPy, Pillow, and Sentence Transformers")

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from internvl.embeddings import EmbeddingModel

ENCODE_OPTIONS = {
    "batch_size": 2,
    "convert_to_numpy": True,
    "precision": "float32",
    "show_progress_bar": False,
}


class SyntheticEncoder(torch.nn.Module):
    """Exercise library batching and output handling without a pretrained model."""

    def __init__(self):
        super().__init__()
        self.modalities = ["text", "image"]
        self.batch_sizes = []

    def tokenize(self, inputs, **kwargs):
        self.batch_sizes.append(len(inputs))
        values = [
            len(item) if isinstance(item, str) else item.getpixel((0, 0))[0]
            for item in inputs
        ]
        vectors = torch.ones((len(inputs), 1024))
        vectors[:, 0] = torch.tensor(values)
        return {"vectors": vectors}

    def forward(self, features, **kwargs):
        return {"sentence_embedding": features["vectors"]}


class SentenceTransformerIntegrationTests(unittest.TestCase):
    def test_library_batches_and_preserves_vectors_and_input_order(self):
        encoder = SyntheticEncoder()
        backend = SentenceTransformer(modules=[encoder], device="cpu")
        with patch("internvl.embeddings._load_backend", return_value=backend):
            model = EmbeddingModel("jinaai/jina-clip-v2", 2, device="cpu")
        texts = ["a", "longer text", "abc"]
        vectors = model.encode_text(iter(texts))
        expected = np.ones((3, 1024), dtype=np.float32)
        expected[:, 0] = [len(text) for text in texts]
        np.testing.assert_allclose(vectors, expected, atol=1e-7)
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(encoder.batch_sizes, [2, 1])
        with Image.new("L", (9, 3), 7) as image:
            vectors = model.encode_images([image])
            self.assertEqual(image.getpixel((0, 0)), 7)
        self.assertEqual(vectors.shape, (1, 1024))
        expected_image = np.ones((1, 1024), dtype=np.float32)
        expected_image[0, 0] = 7
        np.testing.assert_array_equal(vectors, expected_image)


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.backend = MagicMock()
        self.loader = self.enterContext(
            patch("internvl.embeddings._load_backend", return_value=self.backend)
        )
        self.model = EmbeddingModel("jinaai/jina-clip-v2", 2, device="cpu")
        self.loader.assert_called_once_with("jinaai/jina-clip-v2", "cpu")

    def test_text_encoding_preserves_order_and_options(self):
        expected = np.eye(3, 1024, dtype=np.float32)

        def encode(texts, **kwargs):
            self.assertEqual(texts, ["0", "http://example.test/image.png", "2"])
            return expected

        self.backend.encode.side_effect = encode
        values = self.model.encode_text(
            iter(["0", "http://example.test/image.png", "2"])
        )
        np.testing.assert_array_equal(values, expected)
        self.backend.encode.assert_called_once()
        self.assertEqual(self.backend.encode.call_args.kwargs, ENCODE_OPTIONS)

    def test_empty_inputs_return_float32_matrices_without_encoding(self):
        for encode in (self.model.encode_text, self.model.encode_images):
            with self.subTest(encode=encode.__name__):
                values = encode([])
                self.assertEqual(values.shape, (0, 1024))
                self.assertEqual(values.dtype, np.float32)
        self.backend.encode.assert_not_called()

    def test_encoding_failure_propagates(self):
        self.backend.encode.side_effect = RuntimeError("failed")
        with self.assertRaises(RuntimeError):
            self.model.encode_text(["food"])

    def test_invalid_outputs_raise(self):
        for values, message in [
            (np.ones((1, 512)), "Invalid embedding"),
            (np.ones((2, 1024)), "Invalid embedding"),
            (np.ones((1, 1, 1024)), "Invalid embedding"),
            (np.full((1, 1024), np.nan), "Invalid embedding"),
            (np.full((1, 1024), np.inf), "Invalid embedding"),
            (np.zeros((1, 1024)), "Model returned zero-length embeddings"),
        ]:
            with self.subTest(shape=values.shape, message=message):
                self.backend.encode.return_value = values
                with self.assertRaisesRegex(ValueError, message):
                    self.model.encode_text(["food"])

    def test_mixed_images_preserve_order_pixels_and_caller_ownership(self):
        def encode(images, **kwargs):
            self.assertTrue(all(image.mode == "RGB" for image in images))
            self.assertTrue(all(image.size == (9, 3) for image in images))
            values = np.array(
                [[image.getpixel((0, 0))[0]] + [1] * 1023 for image in images],
                dtype=np.float32,
            )
            return values / np.linalg.norm(values, axis=1, keepdims=True)

        self.backend.encode.side_effect = encode
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            with Image.new("L", (9, 3), 7) as original:
                with Image.new("RGB", (9, 3), (14, 0, 0)) as saved:
                    saved.save(path)
                values = self.model.encode_images([original, path, str(path)])
                self.assertEqual(original.getpixel((0, 0)), 7)
            self.assertEqual(values.shape, (3, 1024))
            self.assertEqual(values.dtype, np.float32)
            self.assertLess(values[0, 0], values[1, 0])
            np.testing.assert_array_equal(values[1], values[2])
        for args in self.backend.encode.call_args_list:
            self.assertEqual(args.kwargs, ENCODE_OPTIONS)
        self.assertEqual(
            [len(args.args[0]) for args in self.backend.encode.call_args_list],
            [3],
        )

    def test_path_opened_originals_close_on_failure(self):
        for failure in ("load", "convert", "encode", "validate"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "image.png"
                with Image.new("L", (9, 3), 7) as original:
                    original.save(path)
                    opened = []
                    image_open = Image.open
                    image_convert = Image.Image.convert

                    def open_image(item, opener=image_open, captured=opened):
                        image = opener(item)
                        captured.append(image)
                        return image

                    def convert_image(image, mode, convert=image_convert, fail=failure):
                        if fail == "convert":
                            raise ValueError("conversion failed")
                        return convert(image, mode)

                    self.backend.encode.reset_mock(side_effect=True)
                    self.backend.encode.return_value = np.zeros((2, 1024))
                    inputs = [original, path]
                    error = ValueError
                    if failure == "load":
                        inputs = [path, Path(directory) / "missing.png"]
                        error = FileNotFoundError
                    elif failure == "convert":
                        inputs = [path]
                    elif failure == "encode":
                        self.backend.encode.side_effect = RuntimeError("failed")
                        error = RuntimeError
                    with (
                        patch("PIL.Image.open", side_effect=open_image),
                        patch.object(Image.Image, "convert", convert_image),
                        self.assertRaises(error),
                    ):
                        self.model.encode_images(inputs)
                    self.assertEqual(original.getpixel((0, 0)), 7)
                    self.assertTrue(opened)
                    self.assertTrue(all(image.fp is None for image in opened))
                    for image in opened:
                        with self.assertRaises(ValueError):
                            image.getpixel((0, 0))

    def test_unsupported_model_is_rejected_before_loading(self):
        self.loader.reset_mock()
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            EmbeddingModel("other-model", device="cpu")
        self.loader.assert_not_called()


class BackendLoadingTests(unittest.TestCase):
    def test_constructor_loads_sentence_transformer_with_model_settings(self):
        with patch("internvl.embeddings.SentenceTransformer") as constructor:
            del constructor.return_value[0].modalities
            model = EmbeddingModel("jinaai/jina-clip-v2", device="cpu")
        self.assertIs(model.model, constructor.return_value)
        self.assertEqual(model.model[0].modalities, ["text", "image"])
        constructor.assert_called_once_with(
            "jinaai/jina-clip-v2",
            device="cpu",
            trust_remote_code=True,
            truncate_dim=None,
            model_kwargs={"torch_dtype": torch.float32},
            config_kwargs={
                "use_text_flash_attn": False,
                "use_vision_xformers": False,
            },
        )

    def test_loading_failure_propagates(self):
        with (
            patch(
                "internvl.embeddings._load_backend",
                side_effect=OSError("model unavailable"),
            ),
            self.assertRaisesRegex(OSError, "model unavailable"),
        ):
            EmbeddingModel("jinaai/jina-clip-v2", device="cpu")
