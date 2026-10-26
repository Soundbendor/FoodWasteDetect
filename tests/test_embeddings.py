"""Synthetic embedding contracts, independent of datasets and real models."""

import importlib.util
from importlib.metadata import version
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

    def __init__(self, dimension=1024):
        super().__init__()
        self.dimension = dimension
        self.tasks = []
        self.modalities = ["text", "image"]
        self.batch_sizes = []

    def tokenize(self, inputs, **kwargs):
        self.batch_sizes.append(len(inputs))
        values = [
            len(item) if isinstance(item, str) else (item["image"] if isinstance(item, dict) else item).getpixel((0, 0))[0]
            for item in inputs
        ]
        vectors = torch.ones((len(inputs), self.dimension))
        vectors[:, 0] = torch.tensor(values)
        return {"vectors": vectors}

    def forward(self, features, task=None, **kwargs):
        self.tasks.append(task)
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


    def test_v4_library_applies_passage_prompt_and_routes_task_for_both_modalities(self):
        encoder = SyntheticEncoder(2048)
        backend = SentenceTransformer(
            modules=[encoder], device="cpu", prompts={"passage": "Passage: "}
        )
        # Match v4's modules.json routing through the real library.
        backend.module_kwargs = {"0": ["task", "truncate_dim"]}
        with patch("internvl.embeddings._load_backend", return_value=backend):
            model = EmbeddingModel("jinaai/jina-embeddings-v4", 2, device="cpu")
        texts = ["rice", "a bowl of noodles", "pie"]
        vectors = model.encode_text(iter(texts))
        expected = np.ones((3, 2048), dtype=np.float32)
        expected[:, 0] = [len("Passage: " + text) for text in texts]
        np.testing.assert_array_equal(vectors, expected)
        self.assertEqual(encoder.batch_sizes, [2, 1])
        self.assertEqual(encoder.tasks, ["retrieval", "retrieval"])
        encoder.batch_sizes.clear()
        encoder.tasks.clear()
        with Image.new("L", (9, 3), 7) as image:
            vectors = model.encode_images([image, image, image])
            self.assertEqual(image.getpixel((0, 0)), 7)
        expected[:, 0] = 7
        np.testing.assert_array_equal(vectors, expected)
        self.assertEqual(encoder.batch_sizes, [2, 1])
        self.assertEqual(encoder.tasks, ["retrieval", "retrieval"])


class AdditionalLibraryBatchingTests(unittest.TestCase):
    def test_new_models_batch_preserve_order_and_apply_only_text_prompts(self):
        for model_id, dimension, prefix in [
            ("google/embeddinggemma-2", 768, "Document: "),
            ("openai/clip-vit-base-patch32", 512, ""),
        ]:
            with self.subTest(model_id=model_id):
                encoder = SyntheticEncoder(dimension)
                backend = SentenceTransformer(
                    modules=[encoder], device="cpu", prompts={"Document": "Document: "}
                )
                with patch("internvl.embeddings._load_backend", return_value=backend):
                    model = EmbeddingModel(model_id, 2, device="cpu")
                texts = ["pie", "a bowl of rice", "pear"]
                vectors = model.encode_text(texts)
                np.testing.assert_array_equal(vectors[:, 0], [len(prefix + t) for t in texts])
                self.assertEqual(encoder.batch_sizes, [2, 1])
                encoder.batch_sizes.clear()
                with Image.new("L", (9, 3), 7) as image:
                    vectors = model.encode_images([image] * 3)
                np.testing.assert_array_equal(vectors[:, 0], [7, 7, 7])
                self.assertEqual(encoder.batch_sizes, [2, 1])


class EmbeddingTests(unittest.TestCase):
    model_id = "jinaai/jina-clip-v2"
    dimension = 1024
    text_options = ENCODE_OPTIONS
    image_options = ENCODE_OPTIONS

    def setUp(self):
        self.backend = MagicMock()
        self.loader = self.enterContext(
            patch("internvl.embeddings._load_backend", return_value=self.backend)
        )
        self.model = EmbeddingModel(self.model_id, 2, device="cpu")
        self.loader.assert_called_once_with(self.model_id, "cpu")

    def test_text_encoding_preserves_order_and_options(self):
        expected = np.eye(3, self.dimension, dtype=np.float32)

        def encode(texts, **kwargs):
            self.assertEqual(texts, ["0", "http://example.test/image.png", "2"])
            return expected

        self.backend.encode.side_effect = encode
        values = self.model.encode_text(
            iter(["0", "http://example.test/image.png", "2"])
        )
        np.testing.assert_array_equal(values, expected)
        self.backend.encode.assert_called_once()
        self.assertEqual(self.backend.encode.call_args.kwargs, self.text_options)

    def test_empty_inputs_return_float32_matrices_without_encoding(self):
        for encode in (self.model.encode_text, self.model.encode_images):
            with self.subTest(encode=encode.__name__):
                values = encode([])
                self.assertEqual(values.shape, (0, self.dimension))
                self.assertEqual(values.dtype, np.float32)
        self.backend.encode.assert_not_called()

    def test_encoding_failure_propagates(self):
        self.backend.encode.side_effect = RuntimeError("failed")
        with self.assertRaises(RuntimeError):
            self.model.encode_text(["food"])

    def test_invalid_outputs_raise(self):
        for values, message in [
            (np.ones((1, self.dimension + 1)), "Invalid embedding"),
            (np.ones((2, self.dimension)), "Invalid embedding"),
            (np.ones((1, 1, self.dimension)), "Invalid embedding"),
            (np.full((1, self.dimension), np.nan), "Invalid embedding"),
            (np.full((1, self.dimension), np.inf), "Invalid embedding"),
            (np.zeros((1, self.dimension)), "Model returned zero-length embeddings"),
        ]:
            with self.subTest(shape=values.shape, message=message):
                self.backend.encode.return_value = values
                with self.assertRaisesRegex(ValueError, message):
                    self.model.encode_text(["food"])

    def test_mixed_images_preserve_order_pixels_and_caller_ownership(self):
        def encode(images, **kwargs):
            if self.model_id == "google/embeddinggemma-2":
                self.assertTrue(all(set(item) == {"image"} for item in images))
                images = [item["image"] for item in images]
            self.assertTrue(all(image.mode == "RGB" for image in images))
            self.assertTrue(all(image.size == (9, 3) for image in images))
            values = np.array(
                [[image.getpixel((0, 0))[0]] + [1] * (self.dimension - 1) for image in images],
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
            self.assertEqual(values.shape, (3, self.dimension))
            self.assertEqual(values.dtype, np.float32)
            self.assertLess(values[0, 0], values[1, 0])
            np.testing.assert_array_equal(values[1], values[2])
        for args in self.backend.encode.call_args_list:
            self.assertEqual(args.kwargs, self.image_options)
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
                    self.backend.encode.return_value = np.zeros((2, self.dimension))
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


class V4EmbeddingTests(EmbeddingTests):
    model_id = "jinaai/jina-embeddings-v4"
    dimension = 2048
    text_options = dict(ENCODE_OPTIONS, task="retrieval", prompt_name="passage")
    image_options = dict(ENCODE_OPTIONS, task="retrieval")


class MockFamilyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("internvl.embeddings._check_dependency_family"))


class BackendLoadingTests(MockFamilyTests):
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


class V4BackendLoadingTests(MockFamilyTests):
    def test_v4_uses_checkpoint_dtype_sdpa_and_modality_fallback(self):
        with patch("internvl.embeddings.SentenceTransformer") as constructor:
            del constructor.return_value[0].modalities
            model = EmbeddingModel("jinaai/jina-embeddings-v4", device="cpu")
        self.assertEqual(model.dimension, 2048)
        self.assertEqual(model.model[0].modalities, ["text", "image"])
        constructor.assert_called_once_with(
            "jinaai/jina-embeddings-v4",
            device="cpu",
            trust_remote_code=True,
            truncate_dim=None,
            model_kwargs={"torch_dtype": "auto", "attn_implementation": "sdpa"},
        )

    def test_existing_modality_metadata_is_preserved(self):
        with patch("internvl.embeddings.SentenceTransformer") as constructor:
            modalities = constructor.return_value[0].modalities
            model = EmbeddingModel("jinaai/jina-embeddings-v4", device="cpu")
        self.assertIs(model.model[0].modalities, modalities)


class GemmaEmbeddingTests(EmbeddingTests):
    model_id = "google/embeddinggemma-2"
    dimension = 768
    text_options = dict(ENCODE_OPTIONS, prompt_name="Document")


class OpenAIEmbeddingTests(EmbeddingTests):
    model_id = "openai/clip-vit-base-patch32"
    dimension = 512


class NewBackendLoadingTests(MockFamilyTests):
    def test_gemma_dtype_uses_selected_device_native_support(self):
        for device, native, expected in [
            ("cpu", False, torch.float32),
            ("mps", False, torch.float32),
            ("cuda:1", True, torch.bfloat16),
            ("cuda:1", False, torch.float32),
            ("auto", True, torch.bfloat16),
        ]:
            with (
                self.subTest(device=device, native=native),
                patch("internvl.embeddings.SentenceTransformer") as constructor,
                patch("sentence_transformers.util.get_device_name", return_value="cuda:1"),
                patch("torch.cuda.device") as selected,
                patch("torch.cuda.is_bf16_supported", return_value=native) as support,
            ):
                del constructor.return_value[0].modalities
                model = EmbeddingModel("google/embeddinggemma-2", device=device)
                resolved = "cuda:1" if device == "auto" else device
                constructor.assert_called_once_with(
                    "google/embeddinggemma-2", device=resolved, truncate_dim=None,
                    config_kwargs={"audio_config": None},
                    model_kwargs={"torch_dtype": expected},
                )
                self.assertFalse(hasattr(model.model[0], "modalities"))
                if resolved.startswith("cuda"):
                    selected.assert_called_once_with(resolved)
                    support.assert_called_once_with(including_emulation=False)
                else:
                    selected.assert_not_called()
                    support.assert_not_called()

    def test_clip_loads_fp32_and_explicit_token_limit_without_remote_code(self):
        with patch("internvl.embeddings.SentenceTransformer") as constructor:
            del constructor.return_value[0].modalities
            model = EmbeddingModel("openai/clip-vit-base-patch32", device="cpu")
        constructor.assert_called_once_with(
            "openai/clip-vit-base-patch32", device="cpu", truncate_dim=None,
            model_kwargs={"torch_dtype": torch.float32},
            processor_kwargs={"model_max_length": 77},
        )
        self.assertFalse(hasattr(model.model[0], "modalities"))

    @unittest.skipUnless(version("transformers").startswith("5."), "modern stack only")
    def test_embedding_gemma2_architecture_imports_without_weights(self):
        from transformers import EmbeddingGemma2Config, EmbeddingGemma2Model

        self.assertEqual(EmbeddingGemma2Config.model_type, "embedding_gemma2")
        self.assertTrue(issubclass(EmbeddingGemma2Model, torch.nn.Module))


@unittest.skipUnless(version("transformers").startswith("5."), "modern stack only")
class ProjectedCLIPIntegrationTests(unittest.TestCase):
    def test_real_clip_projects_both_modalities_and_truncates_text(self):
        from transformers import (
            CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizer,
        )
        from internvl.embeddings import _load_backend

        # Local random weights and a tiny vocabulary: no checkpoint downloads.
        config = CLIPConfig(
            projection_dim=512,
            text_config=dict(vocab_size=4, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             max_position_embeddings=77, bos_token_id=0, eos_token_id=1),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_attention_heads=2, image_size=16, patch_size=8),
        )
        with tempfile.TemporaryDirectory() as directory:
            CLIPModel(config).save_pretrained(directory)
            tokenizer = CLIPTokenizer(
                vocab={"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2, "b</w>": 3},
                merges=[],
            )
            CLIPProcessor(
                tokenizer=tokenizer,
                image_processor=CLIPImageProcessor(size={"shortest_edge": 16},
                                                   crop_size={"height": 16, "width": 16}),
            ).save_pretrained(directory)

            def local_backend(model_id, **kwargs):
                self.assertEqual(model_id, "openai/clip-vit-base-patch32")
                return SentenceTransformer(directory, **kwargs)

            with patch("internvl.embeddings.SentenceTransformer", side_effect=local_backend):
                backend = _load_backend("openai/clip-vit-base-patch32", "cpu")
            with patch("internvl.embeddings._load_backend", return_value=backend):
                model = EmbeddingModel("openai/clip-vit-base-patch32", 2, device="cpu")
            long_text = "a " * 120
            short_text = "a " * 75
            vectors = model.encode_text([long_text, short_text, "b"])
            self.assertEqual(vectors.shape, (3, 512))
            np.testing.assert_allclose(vectors[0], vectors[1], atol=1e-6)
            self.assertFalse(np.allclose(vectors[0], vectors[2]))
            with Image.new("RGB", (20, 30), (40, 80, 120)) as image:
                images = model.encode_images([image])
                self.assertEqual(images.shape, (1, 512))
                self.assertEqual(images.dtype, np.float32)
                self.assertTrue(np.isfinite(images).all())
                # Compare against the actual projected model output, not hidden states.
                module = backend[0]
                features = module.preprocess([image])
                with torch.no_grad():
                    expected = module.auto_model.get_image_features(
                        pixel_values=features["pixel_values"]
                    ).pooler_output
                np.testing.assert_allclose(images, expected.numpy(), atol=1e-6)


class DependencyFamilyTests(unittest.TestCase):
    def test_dependency_families_before_pretrained_construction(self):
        from internvl.embeddings import _check_dependency_family

        stacks = [
            {"sentence-transformers": "5.7.0", "transformers": "4.57.6", "peft": "0.17.1"},
            {"sentence-transformers": "6.2.0", "transformers": "5.20.2", "peft": "0.21.2"},
        ]
        for model_id in ("jinaai/jina-clip-v2", "jinaai/jina-embeddings-v4",
                         "google/embeddinggemma-2", "openai/clip-vit-base-patch32"):
            for stack in stacks:
                with self.subTest(model=model_id, stack=stack), patch(
                    "internvl.embeddings.version", side_effect=stack.__getitem__
                ), patch("internvl.embeddings.SentenceTransformer") as constructor:
                    compatible = model_id.startswith("jinaai/") == stack["transformers"].startswith("4.")
                    if compatible:
                        _check_dependency_family(model_id)
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            EmbeddingModel(model_id, device="cpu")
                        message = str(caught.exception)
                        self.assertIn(model_id, message)
                        for package, value in stack.items():
                            self.assertIn(f"{package}={value}", message)
                        self.assertIn("jina-cpu or jina-cu126" if model_id.startswith("jinaai/")
                                      else "default or cu126", message)
                    constructor.assert_not_called()

    def test_missing_dependency_is_actionable(self):
        from importlib.metadata import PackageNotFoundError
        with patch("internvl.embeddings.version", side_effect=PackageNotFoundError), patch(
            "internvl.embeddings.SentenceTransformer"
        ) as constructor, self.assertRaisesRegex(RuntimeError, "peft=missing"):
            EmbeddingModel("jinaai/jina-clip-v2", device="cpu")
        constructor.assert_not_called()

    @unittest.skipUnless(version("transformers").startswith("4."), "Jina stack only")
    def test_jina_apis_and_peft_import_without_weights(self):
        from transformers.models.clip.modeling_clip import clip_loss
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        from peft import PeftModel

        self.assertTrue(callable(clip_loss))
        self.assertTrue(callable(ROPE_INIT_FUNCTIONS["default"]))
        self.assertTrue(issubclass(PeftModel, torch.nn.Module))


class V4HardwareTests(MockFamilyTests):
    def test_checks_selected_device_before_loading(self):
        for device, resolved, native, allowed in [
            ("cpu", "cpu", False, True),
            ("cuda", "cuda", True, True),
            ("cuda", "cuda", False, False),
            ("cuda:1", "cuda:1", True, True),
            ("cuda:1", "cuda:1", False, False),
            ("auto", "cuda:1", True, True),
            ("auto", "cuda:1", False, False),
            ("auto", "cpu", False, True),
            ("mps", "mps", False, False),
            ("xpu", "xpu", False, False),
        ]:
            with self.subTest(device=device, native=native), patch(
                "sentence_transformers.util.get_device_name", return_value=resolved
            ), patch("torch.cuda.device") as selected, patch(
                "torch.cuda.is_bf16_supported", return_value=native
            ) as support, patch("internvl.embeddings.SentenceTransformer") as constructor:
                support.side_effect = lambda **kwargs: (
                    selected.return_value.__enter__.assert_called_once() or native
                )
                if allowed:
                    EmbeddingModel("jinaai/jina-embeddings-v4", device=device)
                    self.assertEqual(constructor.call_args.kwargs["device"], resolved)
                else:
                    with self.assertRaisesRegex(RuntimeError, f"jinaai/jina-embeddings-v4 on {resolved}"):
                        EmbeddingModel("jinaai/jina-embeddings-v4", device=device)
                    constructor.assert_not_called()
                if resolved.startswith("cuda"):
                    selected.assert_called_once_with(resolved)
                    support.assert_called_once_with(including_emulation=False)
                else:
                    selected.assert_not_called()
                    support.assert_not_called()
