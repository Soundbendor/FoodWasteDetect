"""Synthetic cached Gemini sweeps; no model construction or cloud requests."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, Mock

import lancedb
import numpy as np
import pyarrow as pa
import pytest
import yaml
from pydantic import ValidationError
from test_tracking import Artifact

from classification.experiments.config import ExperimentConfig
from classification import cli
from classification.experiments import matroyshka as sweep
from classification.experiments.classification import Experiment
from classification.retrieval.lancedb import LanceDBIndex


def normalized_prefix(vector, dimension):
    prefix = np.asarray(vector[:dimension], dtype=np.float32)
    norm = float(np.linalg.norm(prefix))
    return (prefix / norm if norm > 0 else prefix).tolist()


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
    for name, kind, identities, values in (
        ('reference', 'class_name', ['A', 'B'], vectors),
        ('validation', 'validation_image', ['second', 'first'], [query * 2, query]),
    ):
        index = LanceDBIndex.create(uri, name, 3072)
        index.table.add([
            dict(id=f'{name}-{i}', source_kind=kind, source_identity=identity,
                 value=f'kept-{i}', vector=vector.tolist(), **{'class': label})
            for i, (identity, vector, label) in enumerate(zip(identities, values, ['A', 'B'], strict=True))
        ])
    config = dict(
        dataset={'name': 'foodx251', 'root': str(tmp_path / 'raw')}, output_dir=str(tmp_path / 'output'), k=1,
        embed_images=True, embed_class_labels=False, embed_descriptions=False,
        n_imgs_per_class=1, cache_validation_embeddings=False, sweep_dimensions=[768, 1536, 3072],
        embedding=dict(model_id='google/ge-v2', batch_size=1, truncate_dim=None),
        lancedb=dict(uri=str(uri), table='reference', validation_table='validation'),
    )
    dataset = Mock()
    dataset.get_class_list.return_value = ['A', 'B']
    dataset.validation_set.return_value = [0, 1]

    def batches(split, batch_size, *, decode):
        assert (split, batch_size, decode) == ([0, 1], 1, False)
        for identity in ['first', 'second']:
            yield dict(source_id=[identity], source_split=['val'], fname=[identity], **{'class': ['A']})

    dataset.iter_dataset_batches.side_effect = batches
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
    snapshots = {name: connection.open_table(name).to_arrow().to_pylist() for name in ('reference', 'validation')}
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
        assert effective['embedding']['truncate_dim'] == dimension
        assert effective['vector_dimension'] == dimension
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
        assert report['reference_source_counts'] is None
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
        for source, destination in zip(('reference', 'validation'), sweep.table_names(ExperimentConfig.model_validate(config), dimension, 3072), strict=True):
            table = connection.open_table(destination)
            assert table.schema.field('vector').type == pa.list_(pa.float32(), dimension)
            rows = table.to_arrow().to_pylist()
            expected = (
                snapshots[source]
                if dimension == 3072
                else [
                    dict(row, vector=pytest.approx(normalized_prefix(row['vector'], dimension), rel=1e-6))
                    for row in snapshots[source]
                ]
            )
            assert rows == expected
    assert [item['majority_vote_accuracy'] for item in summary] == [1, 0, 1]
    for name, snapshot in snapshots.items():
        assert connection.open_table(name).to_arrow().to_pylist() == snapshot


def test_reuse_has_no_writes_or_source_scan(sources, monkeypatch):
    config, connection, _, _ = sources
    source = LanceDBIndex.open(config['lancedb']['uri'], 'reference')
    source.create_prefix_table('reference_d768', 768)
    wrapper = MagicMock(wraps=connection)
    wrapper.__contains__.side_effect = lambda name: name in connection
    monkeypatch.setattr('classification.retrieval.lancedb.LanceDBConnection', Mock(return_value=wrapper))
    source.create_prefix_table('reference_d768', 768)
    wrapper.clone_table.assert_not_called()
    wrapper.open_table.assert_called_once_with('reference_d768')


@pytest.mark.parametrize('encoded', ['{broken', ''])
def test_invalid_snapshot_fails_before_destination_creation(sources, encoded, monkeypatch):
    config, connection, _, _ = sources
    source = LanceDBIndex.create(config['lancedb']['uri'], 'invalid', 3072,
                                metadata={'classification.build': encoded})
    wrapper = MagicMock(wraps=connection)
    wrapper.__contains__.side_effect = lambda name: name in connection
    monkeypatch.setattr('classification.retrieval.lancedb.LanceDBConnection', Mock(return_value=wrapper))
    with pytest.raises(json.JSONDecodeError):
        source.create_prefix_table('destination', 768)
    wrapper.clone_table.assert_not_called()
    assert 'destination' not in connection
    assert source.table.schema.metadata[b'classification.build'] == encoded.encode()

    # A readable destination remains reusable without parsing its source snapshot.
    monkeypatch.undo()
    LanceDBIndex.open(config['lancedb']['uri'], 'reference').create_prefix_table('destination', 768)
    version = connection.open_table('destination').version
    source.create_prefix_table('destination', 768)
    assert connection.open_table('destination').version == version


@pytest.mark.parametrize('exists', [True, False])
def test_open_and_create_errors_propagate(sources, exists, monkeypatch):
    config, connection, _, _ = sources
    source = LanceDBIndex.open(config['lancedb']['uri'], 'reference')
    wrapper = MagicMock(wraps=connection)
    wrapper.__contains__.return_value = exists
    failure = RuntimeError('storage failure')
    if exists:
        wrapper.open_table.side_effect = failure
    else:
        wrapper.clone_table.side_effect = failure
    monkeypatch.setattr('classification.retrieval.lancedb.LanceDBConnection', Mock(return_value=wrapper))
    with pytest.raises(RuntimeError, match='storage failure'):
        source.create_prefix_table('destination', 768)
    wrapper.drop_table.assert_not_called()


def test_missing_vector_has_no_embedding_or_summary(sources, runs):
    config, connection, _, _ = sources
    connection.open_table('validation').delete("source_identity = 'first'")
    with pytest.raises(KeyError, match='first'):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert len(runs) == 1
    Experiment._create_embedder.assert_not_called()
    assert not list(Path(config['output_dir']).rglob('report.json'))
    assert not list(Path(config['output_dir']).rglob('summary.json'))


def test_later_failure_preserves_completed_report(sources, runs, monkeypatch):
    config, _, _, _ = sources
    original = cli.evaluate_cached

    def evaluate(effective, tracker):
        if effective.embedding.truncate_dim == 1536:
            raise RuntimeError('later failure')
        return original(effective, tracker)

    monkeypatch.setattr(cli, 'evaluate_cached', evaluate)
    with pytest.raises(RuntimeError, match='later failure'):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert len(list(Path(config['output_dir']).rglob('report.json'))) == 1
    assert not list(Path(config['output_dir']).rglob('summary.json'))
    assert len(runs) == 2
    assert runs[1][0].__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize(
    'validation,sweep_dimensions',
    [
        ('reference', [768, 1536, 3072]),
        ('reference_d768', [768, 1536, 3072]),
        ('reference_d1536', [768, 1536, 3072]),
        ('reference_d768', [768, 1536]),
        ('', [768, 1536, 3072]),
    ],
)
def test_collisions_fail_before_tracking_or_writes(
    sources, runs, monkeypatch, validation, sweep_dimensions,
):
    config, _, _, _ = sources
    if validation and validation != 'reference':
        LanceDBIndex.open(config['lancedb']['uri'], 'validation').create_prefix_table(validation, 3072)
    config['lancedb']['validation_table'] = validation
    config['sweep_dimensions'] = sweep_dimensions
    prepare = Mock()
    monkeypatch.setattr(LanceDBIndex, 'create_prefix_table', prepare)
    with pytest.raises(ValueError):
        cli._run_experiment(ExperimentConfig.model_validate(config), 'matroyshka-sweep')
    assert runs == []
    prepare.assert_not_called()


def test_stream_has_no_default_result_limit(sources):
    config, connection, _, _ = sources
    source_index = LanceDBIndex.open(config['lancedb']['uri'], 'reference')
    source = source_index.table
    rows = source.to_arrow().to_pylist()
    source.add([dict(rows[i % 2], id=f'extra-{i}', source_identity=f'extra-{i}') for i in range(23)])
    snapshot = source.to_arrow().to_pylist()
    destination = source_index.create_prefix_table('many_d768', 768).table
    assert destination.to_arrow().to_pylist() == [
        dict(row, vector=pytest.approx(normalized_prefix(row['vector'], 768), rel=1e-6))
        for row in snapshot
    ]
    version = destination.version
    reused = source_index.create_prefix_table('many_d768', 768).table
    assert reused.version == version
    assert source.to_arrow().to_pylist() == snapshot


@pytest.mark.parametrize('failure', [RuntimeError('synthetic reader failure'), KeyboardInterrupt()])
def test_native_stream_creation_error(sources, monkeypatch, failure):
    config, connection, _, _ = sources
    source_index = LanceDBIndex.open(config['lancedb']['uri'], 'reference')
    import lance

    original_udf = lance.batch_udf

    def broken_udf(*args, **kwargs):
        def decorator(fn):
            @original_udf(*args, **kwargs)
            def failing(batch):
                fn(batch)
                raise failure

            return failing

        return decorator

    monkeypatch.setattr('classification.retrieval.lancedb.lance.batch_udf', broken_udf)
    with pytest.raises((Exception, KeyboardInterrupt)):
        source_index.create_prefix_table('broken_native', 768)
    # No assertion about table durability or repair after a native storage failure.
    assert connection.open_table('reference').count_rows() == 2


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
            config=effective, table=effective.lancedb.table,
            reference_source_counts=None, predictions=[], metrics=metrics([], ["A", "B"]),
        ), "synthetic"

    sweep.run_sweep(config, evaluate)
    assert [variant.embedding.truncate_dim for variant in variants] == raw['sweep_dimensions']
    assert [variant.vector_dimension for variant in variants] == raw['sweep_dimensions']
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
    monkeypatch.setattr(LanceDBIndex, "create_prefix_table", prepare)
    for dimensions in (None, [3073]):
        config["sweep_dimensions"] = dimensions
        with pytest.raises(ValueError):
            sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    prepare.assert_not_called()
    evaluate.assert_not_called()


def test_custom_non_gemini_order_and_truncated_sources(sources):
    config, connection, _, _ = sources
    uri = config['lancedb']['uri']
    LanceDBIndex.open(uri, 'reference').create_prefix_table('short_ref', 16)
    LanceDBIndex.open(uri, 'validation').create_prefix_table('short_val', 16)
    config['embedding']['model_id'] = 'example/custom'
    config['lancedb'].update(table='short_ref', validation_table='short_val')
    config['sweep_dimensions'] = [8, 16, 4]
    variants = []
    from classification.experiments.report_types import EvaluationReport
    from classification.retrieval.metrics import metrics

    def evaluate(effective):
        variants.append(effective)
        return EvaluationReport(config=effective, table=effective.lancedb.table,
            reference_source_counts=None, predictions=[], metrics=metrics([], ['A', 'B'])), 'run'

    result = sweep.run_sweep(ExperimentConfig.model_validate(config), evaluate)
    assert [item['dimension'] for item in result['results']] == [8, 16, 4]
    assert [item.lancedb.table for item in variants] == ['short_ref_d8', 'short_ref', 'short_ref_d4']
    for width in (8, 4):
        original = connection.open_table('short_ref').to_arrow().to_pylist()
        actual = connection.open_table(f'short_ref_d{width}').to_arrow().to_pylist()
        assert actual == [
            dict(row, vector=pytest.approx(normalized_prefix(row['vector'], width), rel=1e-6))
            for row in original
        ]
    Experiment._create_embedder.assert_not_called()


def test_mismatched_source_widths_fail_before_writes(sources, monkeypatch):
    config, _, _, _ = sources
    LanceDBIndex.open(config['lancedb']['uri'], 'validation').create_prefix_table('short_val', 16)
    config['lancedb']['validation_table'] = 'short_val'
    prepare, evaluate = Mock(), Mock()
    monkeypatch.setattr(LanceDBIndex, 'create_prefix_table', prepare)
    with pytest.raises(ValueError, match='widths must match'):
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
    assert connection.open_table('reference').schema.field('vector').type.list_size == 3072
    Experiment._create_embedder.assert_not_called()
