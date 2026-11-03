"""Focused contracts for dataset naming and local vector validation."""

from unittest.mock import Mock

import pytest


def test_validation_set_preserves_legacy_aliases(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("pandas")
    from classification.datasets.image_dataset import FoodX251ImageDataset

    annotation = tmp_path / "annot"
    annotation.mkdir()
    (annotation / "class_list.txt").write_text("0 apple\n", encoding="utf-8")
    for split in ("train", "val"):
        (annotation / f"{split}_info.csv").write_text("fruit.png,0\n", encoding="utf-8")

    dataset = FoodX251ImageDataset(tmp_path)

    assert dataset.validation_set() is dataset.splits["validation"]
    assert dataset.test_set() is dataset.validation_set()
    assert dataset.test_path == dataset.val_path
    assert dataset.validation_set()[0]["source_split"] == "val"
    assert str(dataset.validation_set().split) == "validation"


@pytest.mark.parametrize("modality", ["text", "images"])
@pytest.mark.parametrize("has_zero_row", [False, True])
def test_local_encoder_distinguishes_tiny_vectors_from_zero(
    monkeypatch, modality, has_zero_row,
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sentence_transformers")
    from PIL import Image

    from classification.encoders import sentence_transformers as local

    vectors = np.array([[1e-30, -1e-30], [0, 1e-30]], dtype=np.float32)
    if has_zero_row:
        vectors[1] = 0
    backend = Mock()
    backend.encode.return_value = vectors
    monkeypatch.setattr(local, "_load_backend", Mock(return_value=backend))
    encoder = local.EmbeddingModel("openai/clip-vit-base-patch32", device="cpu", truncate_dim=2)

    with Image.new("RGB", (2, 2)) as image:
        encode = encoder.encode_text if modality == "text" else encoder.encode_images
        inputs = ["apple", "pear"] if modality == "text" else [image, image]
        if has_zero_row:
            with pytest.raises(ValueError, match="zero-length embeddings"):
                encode(inputs)
        else:
            result = encode(inputs)
            np.testing.assert_array_equal(result, vectors)
            assert result.dtype == np.float32
        assert image.getpixel((0, 0)) == (0, 0, 0)
