"""Saved vocabulary loading and text-reference ordering without model services."""

import json

import pytest

from classification.experiments.descriptions import load_descriptions, text_references


def test_saved_vocabulary_is_read_only(tmp_path):
    path = tmp_path / "vocabulary.json"
    vocabulary = {"Pear": ["green pear"], "Apple": ["red apple", "apple slice"]}
    path.write_text(json.dumps({"descriptions": vocabulary}))
    assert load_descriptions(path) == vocabulary


def test_missing_vocabulary_does_not_create_file(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_descriptions(path)
    assert not path.exists()


@pytest.mark.parametrize("include_labels", [False, True])
@pytest.mark.parametrize("include_descriptions", [False, True])
def test_reference_order_and_identities(include_labels, include_descriptions):
    classes = ["Pear", "Apple"]
    vocabulary = {"Apple": ["red apple", "apple slice"], "Pear": ["green pear"]}
    records = text_references(
        classes,
        include_class_labels=include_labels,
        descriptions=vocabulary if include_descriptions else None,
    )
    expected = []
    if include_labels:
        expected += [(label, "class_name", label, label) for label in classes]
    if include_descriptions:
        expected += [
            ("Pear", "generated_description", "Pear:0", "green pear"),
            ("Apple", "generated_description", "Apple:0", "red apple"),
            ("Apple", "generated_description", "Apple:1", "apple slice"),
        ]
    assert [(r.label, r.source_kind, r.source_identity, r.value) for r in records] == expected
    assert len({r.id for r in records}) == len(records)


def test_missing_class_descriptions_fail_before_reference_encoding():
    with pytest.raises(KeyError, match="Pear"):
        text_references(["Pear"], include_class_labels=False, descriptions={})
