"""Saved vocabulary loading and text-reference ordering without model services."""

import json

import pytest

from classification.experiments.descriptions import load_descriptions, text_references


def test_saved_vocabulary_is_read_only(tmp_path):
    path = tmp_path / "vocabulary.json"
    vocabulary = {"Pear": ["green pear"], "Apple": ["red apple", "apple slice"]}
    path.write_text(json.dumps({"descriptions": vocabulary}))
    assert load_descriptions(path) == vocabulary


def test_null_mapping_rejected_even_with_other_reference_sources(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text('{"descriptions": null}')
    with pytest.raises(ValueError, match="descriptions mapping"):
        load_descriptions(path)


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
            (label, "description_prototype", label, json.dumps(vocabulary[label]))
            for label in classes
        ]
    assert [(r.label, r.source_kind, r.source_identity, r.value) for r in records] == expected
    assert len({r.id for r in records}) == len(records)


def test_missing_class_descriptions_fail_before_reference_encoding():
    with pytest.raises(KeyError, match="Pear"):
        text_references(["Pear"], include_class_labels=False, descriptions={})


@pytest.mark.parametrize("captions", [[], [""], ["apple", "  "], "apple", [None]])
def test_invalid_caption_list_names_class(captions):
    with pytest.raises(ValueError, match="Apple"):
        text_references(["Apple"], include_class_labels=True, descriptions={"Apple": captions})
