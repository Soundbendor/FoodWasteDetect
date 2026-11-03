"""Creation snapshots use synthetic configuration and real temporary LanceDB tables."""

import itertools
import json

import numpy as np
import pytest

from classification.experiments.matryoshka import prepare_table
from classification.experiments.table_metadata import build_metadata, serialize_metadata
from classification.retrieval.lancedb import LanceDBIndex
from classification.schemas import PointRecord


@pytest.mark.parametrize('flags', list(itertools.product([False, True], repeat=3))[1:])
@pytest.mark.parametrize('cap', [None, 3])
@pytest.mark.parametrize('truncate', [None, 2])
def test_selected_settings(flags, cap, truncate):
    config = dict(
        dataset={'name': 'FoodX251', 'root': '/private'},
        embedding={'model_id': 'jinaai/jina-clip-v2', 'truncate_dim': truncate,
                   'device': 'cpu', 'vertex': {'credentials': 'private'}},
        **dict(zip(('embed_images', 'embed_class_labels', 'embed_descriptions'), flags)),
        n_imgs_per_class=cap, description_count=5, k=99, wandb={'key': 'private'},
    )
    snapshot = build_metadata(config, 'reference', 2)
    encoded = serialize_metadata(snapshot)['classification.build']
    assert json.loads(encoded) == snapshot
    expected = dict(dataset={'name': 'FoodX251'},
                    embedding={'model_id': 'jinaai/jina-clip-v2', 'truncate_dim': truncate},
                    **dict(zip(('embed_images', 'embed_class_labels', 'embed_descriptions'), flags)),
                    n_imgs_per_class=cap)
    if flags[2]:
        expected['description_count'] = 5
    assert snapshot == dict(metadata_version=1, table_kind='reference',
                            vector_dimension=2, config=expected)
    validation = build_metadata(config, 'validation', 2)
    assert validation['config'] == {k: expected[k] for k in ('dataset', 'embedding')}
    assert validation['table_kind'] == 'validation'


@pytest.mark.parametrize('prompt', [None, False, True])
def test_vertex_defaults(prompt):
    config = dict(dataset={'name': 'FoodX251'}, embedding={'model_id': 'google/ge-v2'})
    if prompt is not None:
        config['embedding']['vertex'] = {'classification_prompt': prompt}
    snapshot = build_metadata(config, 'validation', 3)
    assert snapshot['config']['embedding'] == dict(
        model_id='google/ge-v2', truncate_dim=None,
        vertex={'classification_prompt': prompt if prompt is not None else False})


@pytest.mark.parametrize('with_snapshot', [False, True])
def test_persistence_and_sweep_lineage(tmp_path, with_snapshot):
    config = dict(dataset={'name': 'FoodX251'}, embedding={'model_id': 'google/ge-v2',
                                                         'truncate_dim': 3})
    metadata = {'unrelated': 'preserved café'}
    if with_snapshot:
        metadata.update(serialize_metadata(build_metadata(config, 'validation', 3)))
    index = LanceDBIndex.create(tmp_path / 'db', 'source', 3, metadata=metadata)
    expected = {key.encode(): value.encode() for key, value in metadata.items()}
    assert index.table.schema.metadata == expected
    schema = index.table.schema.remove_metadata()
    for label, vector in [('A', [1., 0., 0.]), ('B', [0., 1., 0.])]:
        index.append([PointRecord(label, 'validation_image', label, label)], np.array([vector]))
        assert index.table.schema.metadata == expected
    before = index.query(np.array([1., 0., 0.]), 2)
    index.optimize()
    reopened = LanceDBIndex.open(index.uri, 'source')
    assert reopened.table.schema.metadata == expected
    assert reopened.table.schema.remove_metadata() == schema
    assert reopened.query(np.array([1., 0., 0.]), 2) == before
    from lancedb.db import LanceDBConnection
    connection = LanceDBConnection(index.uri)
    prepare_table(connection, 'source', 'derived', 2, 1)
    destination = connection.open_table('derived')
    derived = destination.schema.metadata
    assert derived[b'unrelated'] == expected[b'unrelated']
    if with_snapshot:
        snapshot = json.loads(derived[b'classification.build'])
        original = json.loads(expected[b'classification.build'])
        assert snapshot == dict(original, vector_dimension=2,
                                derived_from={'table': 'source', 'vector_dimension': 3})
    else:
        assert derived == expected
    assert destination.to_arrow().to_pylist() == [
        dict(row, vector=row['vector'][:2]) for row in reopened.table.to_arrow().to_pylist()]
    version = destination.version
    prepare_table(connection, 'source', 'derived', 1, 1)
    destination = connection.open_table('derived')
    assert destination.version == version
    assert destination.schema.metadata == derived
    assert connection.open_table('source').schema.metadata == expected
