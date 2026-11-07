"""Synthetic cached Gemini sweeps; no model construction or cloud requests."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, Mock

import lancedb
from lancedb.db import LanceDBConnection
import numpy as np
import pytest
import yaml
from datasets import Dataset
from pydantic import ValidationError
from test_tracking import Artifact

from classification.experiments.config import ExperimentConfig
from classification import cli
from classification.experiments import matryoshka as sweep
from classification.experiments.classification import Experiment
from classification.retrieval.embedding_table import (
    EmbeddingTable,
    EmbeddingTableError,
    IncompatibleTableError,
    IncompleteTableError,
    PrefixSource,
)
from classification.schemas import PointRecord
from table_helpers import fingerprints

ROLES = {'reference': 'reference', 'validation': 'validation_cache'}


def stored(uri, name):
    return EmbeddingTable.open(uri, name, ROLES[name.split('_d')[0]])


@pytest.fixture
def sources(tmp_path, monkeypatch):
    uri = tmp_path / 'database'
    connection = lancedb.connect(uri)
    # At 768 A wins; at 1536 B wins; at 3072 A wins again.
    vectors = np.zeros((2, 3072), dtype=np.float32)
    vectors[0, [0, 1600]] = [1, 5]
    vectors[1, [1, 800]] = [1, 5]
    query = np.zeros(3072, dtype=np.float32)
    query[[0, 800, 1600]] = [1, 5, 10]
    for name, role, kind, identities, values in (
        ('reference', 'reference', 'class_name', ['A', 'B'], vectors),
        ('validation', 'validation_cache', 'validation_image', ['second', 'first'],
         np.stack([query * 2, query])),
    ):
        table = EmbeddingTable.create(uri, name, role, 3072, snapshot=f'{{"created": "{name}"}}', **fingerprints(role))
        records = [
            PointRecord(label, kind, identity, f'kept-{i}', id=f'{name}-{i}')
            for i, (identity, label) in enumerate(zip(identities, ['A', 'B'], strict=True))
        ]
        table.append(records, values)
        table.mark_complete(record.source_key for record in records)
    config = dict(
        dataset={'name': 'foodx251', 'root': str(tmp_path / 'raw')}, output_dir=str(tmp_path / 'output'), k=1,
        embed_images=True, embed_class_labels=False, embed_descriptions=False,
        n_imgs_per_class=1, cache_validation_embeddings=False, sweep_dimensions=[768, 1536, 3072],
        embedding=dict(model_id='google/ge-v2', batch_size=1, truncate_dim=None),
        lancedb=dict(uri=str(uri), table='reference', validation_table='validation'),
    )
    dataset = Mock()
    dataset.get_class_list.return_value = ['A', 'B']
    dataset.validation_set.return_value = Dataset.from_dict({'source_id': ['first', 'second']})

    def batches(split, batch_size, *, decode):
        assert (split, batch_size, decode) == ('val', 1, False)
        for identity in ['first', 'second']:
            yield dict(source_id=[identity], source_split=['val'], fname=[identity], **{'class': ['A']})

    dataset.iter_batches.side_effect = batches
    monkeypatch.setattr('classification.experiments.classification.load_dataset', Mock(return_value=dataset))
    monkeypatch.setattr(Experiment, '_create_embedder', Mock(side_effect=AssertionError('embedding forbidden')))
    return config, connection, vectors, query


@pytest.fixture
def runs(monkeypatch):
    contexts = []

    def initialize(**kwargs):
        run = MagicMock(id=f'run-{len(contexts)}', summary={})
        run.__enter__.return_value = run
        run.__exit__.return_value = False
        contexts.append((run, kwargs))
        return run

    monkeypatch.setattr(cli.wandb, 'init', Mock(side_effect=initialize))
    monkeypatch.setattr('classification.tracking.wandb.Artifact', Artifact)
    return contexts


def test_shared_cli_sweep(sources, runs, tmp_path):
    config, connection, vectors, query = sources
    original = deepcopy(config)
    uri = config['lancedb']['uri']
    snapshots = {name: stored(uri, name).rows() for name in ROLES}
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(config))
    before = path.read_bytes()
    cli.main(['experiment', 'matroyshka-sweep', '--config_file', str(path),
              '--set', 'k=2', '--set', 'k=1'])
    assert config == original and path.read_bytes() == before
    summary_path, = Path(config['output_dir']).glob('matroyshka_sweep_*/summary.json')
    summary = json.loads(summary_path.read_text())['results']
    assert len(summary) == len(runs) == 3
    for dimension, result, (run, kwargs) in zip(config['sweep_dimensions'], summary, runs, strict=True):
        effective = kwargs['config']
        assert effective['k'] == 1
        assert effective['embedding']['truncate_dim'] is None
        assert 'vector_dimension' not in effective
        assert effective['cache_validation_embeddings'] is True
        assert effective['embed_images'] is True
        assert effective['n_imgs_per_class'] == 1
        assert result['wandb_run_id'] == run.id
        assert result['dimension'] == dimension
        assert kwargs['job_type'] == 'matroyshka-sweep'
        assert kwargs['settings'].disable_git and kwargs['settings'].x_disable_meta
        run.__enter__.assert_called_once()
        run.__exit__.assert_called_once_with(None, None, None)
        config_artifact, report_artifact = [call.args[0] for call in run.log_artifact.call_args_list]
        assert yaml.safe_load(config_artifact.files['resolved_config.yaml']) == effective
        report_path, = Path(result['output_dir']).glob('evaluation_*/report.json')
        assert report_artifact.files['report.json'] == report_path.read_bytes()
        report = json.loads(report_path.read_text())
        assert report['config'] == effective
        assert report['dimension'] == dimension
        assert report['reference_source_counts'] == {'class_name': 2}
        assert run.summary['reference_source_counts'] == {'class_name': 2}
        assert report['metrics']['evaluated'] == 2
        for key, value in report['metrics'].items():
            if key != 'per_class':
                assert result[key] == value
                assert run.summary[f'evaluation/{key}'] == value
        assert [p['source_id'] for p in report['predictions']] == ['first', 'second']
        assert {p['source_split'] for p in report['predictions']} == {'val'}
        prefix = vectors[:, :dimension]
        q = query[:dimension]
        scores = (prefix @ q) / (np.linalg.norm(prefix, axis=1) * np.linalg.norm(q))
        winner = ['A', 'B'][int(np.argmax(scores))]
        for prediction in report['predictions']:
            assert prediction['prediction'] == winner
            assert prediction['neighbors'][0]['score'] == pytest.approx(float(max(scores)), abs=1e-6)
        for source, destination in zip(('reference', 'validation'), sweep.table_names(('reference', 'validation'), dimension, 3072), strict=True):
            table = stored(uri, destination)
            assert table.dimension == dimension
            assert table.is_complete
            assert table.prefix_source == (None if dimension == 3072 else PrefixSource(source, 3072))
            rows = table.rows()
            assert [record for record, _ in rows] == [record for record, _ in snapshots[source]]
            np.testing.assert_array_equal(
                [vector for _, vector in rows], [vector[:dimension] for _, vector in snapshots[source]]
            )
            assert table.creation_snapshot == f'{{"created": "{source}"}}'
            # Prefixes inherit the sources' fingerprints, not the sweep configuration's.
            source_table = stored(uri, source)
            assert table.encoder_fingerprint == source_table.encoder_fingerprint
            assert table.inputs_fingerprint == source_table.inputs_fingerprint
    assert [item['majority_vote_accuracy'] for item in summary] == [1, 0, 1]
    for name, snapshot in snapshots.items():
        after = stored(uri, name).rows()
        assert [record for record, _ in after] == [record for record, _ in snapshot]
        np.testing.assert_array_equal([v for _, v in after], [v for _, v in snapshot])


@pytest.mark.parametrize('name', ['reference', 'validation'])
def test_incomplete_source_is_refused_before_deriving_or_tracking(sources, runs, name):
    config, connection, _, _ = sources
    connection.open_table(name).delete("source_identity = 'B' OR source_identity = 'first'")
    with pytest.raises(IncompleteTableError, match=f"'{name}' is incomplete"):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert runs == []
    assert not EmbeddingTable.exists(connection.uri, 'reference_d768')
    assert not EmbeddingTable.exists(connection.uri, 'validation_d768')
    Experiment._create_embedder.assert_not_called()
    assert not list(Path(config['output_dir']).rglob('report.json'))
    assert not list(Path(config['output_dir']).rglob('summary.json'))


def test_later_failure_preserves_completed_report(sources, runs, monkeypatch):
    config, _, _, _ = sources
    original = cli.evaluate_cached

    def evaluate(effective, tracker):
        if effective.lancedb.table == 'reference_d1536':
            raise RuntimeError('later failure')
        return original(effective, tracker)

    monkeypatch.setattr(cli, 'evaluate_cached', evaluate)
    with pytest.raises(RuntimeError, match='later failure'):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert len(list(Path(config['output_dir']).rglob('report.json'))) == 1
    assert not list(Path(config['output_dir']).rglob('summary.json'))
    assert len(runs) == 2
    assert runs[1][0].__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize('validation', ['reference_d768', 'reference_d1536'])
def test_derived_name_collisions_fail_before_tracking(sources, runs, validation):
    config, connection, _, _ = sources
    stored(connection.uri, 'validation').derive_prefix(validation, 3072, batch_size=1)
    config['lancedb']['validation_table'] = validation
    with pytest.raises(ValueError, match='distinct'):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert runs == []


def test_help_does_not_load_config(monkeypatch, capsys):
    load = Mock(side_effect=AssertionError('config load forbidden'))
    monkeypatch.setattr(cli, '_load_config', load)
    with pytest.raises(SystemExit) as error:
        cli.main(['experiment', 'matroyshka-sweep', '--help'])
    assert error.value.code == 0
    assert 'matroyshka-sweep' in capsys.readouterr().out
    load.assert_not_called()


def test_sweep_variants_are_independent_validated_models(sources):
    from classification.experiments.report_types import EvaluationReport
    from classification.retrieval.metrics import metrics

    raw, _, _, _ = sources
    config = ExperimentConfig.model_validate(raw)
    variants = []

    def evaluate(effective):
        assert isinstance(effective, ExperimentConfig)
        variants.append(effective)
        return EvaluationReport(
            config=effective, table=effective.lancedb.table, dimension=0,
            reference_source_counts={'class_name': 2}, predictions=[], metrics=metrics([], ["A", "B"]),
        ), "synthetic"

    sweep.run_sweep(config, evaluate)
    assert [variant.embedding.truncate_dim for variant in variants] == [None, None, None]
    assert [variant.lancedb.table for variant in variants] == [
        'reference_d768', 'reference_d1536', 'reference']
    assert all('vector_dimension' not in variant.model_dump() for variant in variants)
    assert len({id(variant.embedding) for variant in variants}) == 3
    assert config.model_dump() == ExperimentConfig.model_validate(raw).model_dump()
    assert config.embedding.truncate_dim is None


@pytest.mark.parametrize("dimensions", [[], [1, 1], [0], [-1], [1.5], [True], ["1"]])
def test_invalid_dimensions_rejected(sources, dimensions):
    config, _, _, _ = sources
    config["sweep_dimensions"] = dimensions
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(config)


def test_missing_and_oversized_dimensions_fail_before_writes(sources, monkeypatch):
    config, _, _, _ = sources
    prepare = Mock()
    evaluate = Mock()
    monkeypatch.setattr(EmbeddingTable, "derive_prefix", prepare)
    for dimensions in (None, [3073]):
        config["sweep_dimensions"] = dimensions
        with pytest.raises(ValueError):
            sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    prepare.assert_not_called()
    evaluate.assert_not_called()


def test_custom_non_gemini_order_and_truncated_sources(sources):
    config, connection, _, _ = sources
    stored(connection.uri, 'reference').derive_prefix('short_ref', 16, batch_size=1)
    stored(connection.uri, 'validation').derive_prefix('short_val', 16, batch_size=1)
    config['embedding']['model_id'] = 'example/custom'
    config['lancedb'].update(table='short_ref', validation_table='short_val')
    config['sweep_dimensions'] = [8, 16, 4]
    variants = []
    from classification.experiments.report_types import EvaluationReport
    from classification.retrieval.metrics import metrics

    def evaluate(effective):
        variants.append(effective)
        return EvaluationReport(config=effective, table=effective.lancedb.table, dimension=0,
            reference_source_counts={'class_name': 2}, predictions=[], metrics=metrics([], ['A', 'B'])), 'run'

    result = sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    assert [item['dimension'] for item in result['results']] == [8, 16, 4]
    assert [item.lancedb.table for item in variants] == ['short_ref_d8', 'short_ref', 'short_ref_d4']
    for width in (8, 4):
        original = EmbeddingTable.open(connection.uri, 'short_ref', 'reference').rows()
        actual = EmbeddingTable.open(connection.uri, f'short_ref_d{width}', 'reference').rows()
        assert [record for record, _ in actual] == [record for record, _ in original]
        np.testing.assert_array_equal([v for _, v in actual], [v[:width] for _, v in original])
    Experiment._create_embedder.assert_not_called()


def test_mismatched_source_widths_fail_before_writes(sources, monkeypatch):
    config, connection, _, _ = sources
    stored(connection.uri, 'validation').derive_prefix('short_val', 16, batch_size=1)
    config['lancedb']['validation_table'] = 'short_val'
    prepare, evaluate = Mock(), Mock()
    monkeypatch.setattr(EmbeddingTable, 'derive_prefix', prepare)
    with pytest.raises(ValueError, match='widths must match'):
        sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    prepare.assert_not_called()
    evaluate.assert_not_called()


def test_sources_from_different_encoders_fail_before_writes(sources, monkeypatch):
    config, connection, vectors, _ = sources
    other = fingerprints('validation_cache')
    other['encoder'] = dict(other['encoder'], **{'embedding.model_id': 'other'})
    cache = EmbeddingTable.create(connection.uri, 'other_val', 'validation_cache', 3072,
                                  snapshot='', **other)
    records = [PointRecord('A', 'validation_image', identity, '') for identity in ('first', 'second')]
    cache.append(records, vectors)
    cache.mark_complete(record.source_key for record in records)
    config['lancedb']['validation_table'] = 'other_val'
    prepare, evaluate = Mock(), Mock()
    monkeypatch.setattr(EmbeddingTable, 'derive_prefix', prepare)
    with pytest.raises(IncompatibleTableError, match="embedding.model_id: stored 'synthetic', required 'other'"):
        sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    prepare.assert_not_called()
    evaluate.assert_not_called()


def test_cli_non_gemini_override_omits_source_width(sources, runs, tmp_path):
    config, connection, _, _ = sources
    config['embedding']['model_id'] = 'example/custom'
    path = tmp_path / 'config.yml'
    path.write_text(yaml.safe_dump(config))
    cli.main(['experiment', 'matroyshka-sweep', '--config_file', str(path),
              '--set', 'sweep_dimensions=[1536,768]'])
    summary_path, = Path(config['output_dir']).glob('matroyshka_sweep_*/summary.json')
    results = json.loads(summary_path.read_text())['results']
    assert [item['dimension'] for item in results] == [1536, 768]
    assert [item['table'] for item in results] == ['reference_d1536', 'reference_d768']
    assert len(runs) == 2
    assert stored(connection.uri, 'reference').dimension == 3072
    Experiment._create_embedder.assert_not_called()


def test_collision_when_source_width_omitted(sources, monkeypatch):
    config, connection, _, _ = sources
    stored(connection.uri, 'validation').derive_prefix('reference_d768', 3072, batch_size=1)
    config['lancedb']['validation_table'] = 'reference_d768'
    config['sweep_dimensions'] = [768, 1536]
    prepare, evaluate = Mock(), Mock()
    monkeypatch.setattr(EmbeddingTable, 'derive_prefix', prepare)
    with pytest.raises(ValueError, match='distinct'):
        sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    prepare.assert_not_called()
    evaluate.assert_not_called()


def test_rerun_after_a_crash_mid_copy_recreates_the_prefix_and_completes(sources, runs, monkeypatch):
    config, connection, vectors, _ = sources
    create = LanceDBConnection.create_table

    def crash(self, name, *args, **kwargs):
        table = create(self, name, *args, **kwargs)
        if name == 'validation_d1536':
            raise RuntimeError('crash mid-copy')
        return table

    monkeypatch.setattr(LanceDBConnection, 'create_table', crash)
    with pytest.raises(RuntimeError, match='crash mid-copy'):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert not stored(connection.uri, 'validation_d1536').is_complete
    monkeypatch.setattr(LanceDBConnection, 'create_table', create)

    cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')

    summaries = list(Path(config['output_dir']).glob('matroyshka_sweep_*/summary.json'))
    assert len(summaries) == 1
    results = json.loads(summaries[0].read_text())['results']
    assert [item['majority_vote_accuracy'] for item in results] == [1, 0, 1]
    recreated = stored(connection.uri, 'validation_d1536')
    assert recreated.is_complete
    assert [record.source_identity for record, _ in recreated.rows()] == ['second', 'first']
    Experiment._create_embedder.assert_not_called()


def not_a_prefix(uri):
    table = EmbeddingTable.create(uri, 'reference_d768', 'reference', 768, snapshot='kept',
                                  **fingerprints('reference'))
    table.append([PointRecord('A', 'class_name', 'A', 'A')], np.ones((1, 768)))
    table.mark_complete([('class_name', 'A')])


def legacy(uri):
    lancedb.connect(uri).create_table('reference_d768', data=[{'vector': [1.0, 2.0]}])


def prefix_of_another_source(uri):
    other = EmbeddingTable.create(uri, 'other', 'reference', 3072, snapshot='',
                                  **fingerprints('reference'))
    other.append([PointRecord('A', 'class_name', 'A', 'A')], np.ones((1, 3072)))
    other.mark_complete([('class_name', 'A')])
    other.derive_prefix('reference_d768', 768, batch_size=1)


def prefix_at_another_dimension(uri):
    stored(uri, 'reference').derive_prefix('reference_d768', 1536, batch_size=1)


@pytest.mark.parametrize('setup', [not_a_prefix, legacy, prefix_of_another_source,
                                   prefix_at_another_dimension])
def test_sweep_refuses_a_destination_it_did_not_derive_and_leaves_it_unchanged(sources, runs, setup):
    config, connection, _, _ = sources
    setup(connection.uri)
    before = connection.open_table('reference_d768').to_arrow()
    with pytest.raises(EmbeddingTableError):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert connection.open_table('reference_d768').to_arrow().equals(before)
    assert not list(Path(config['output_dir']).rglob('summary.json'))


def test_rerun_reuses_complete_prefix_tables_without_copying(sources, runs, monkeypatch):
    config, connection, _, _ = sources
    cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    for operation in ('create_table', 'drop_table'):
        monkeypatch.setattr(LanceDBConnection, operation,
                            Mock(side_effect=AssertionError(f'{operation} forbidden')))

    cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')

    summaries = sorted(Path(config['output_dir']).glob('matroyshka_sweep_*/summary.json'))
    assert len(summaries) == 2
    first, second = (json.loads(path.read_text())['results'] for path in summaries)
    assert [item['majority_vote_accuracy'] for item in second] == [
        item['majority_vote_accuracy'] for item in first] == [1, 0, 1]
