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
from test_tracking import Artifact

from internvl import cli
from internvl import matroyshka_sweep as sweep
from internvl.experiment import Experiment
from internvl.retrieval import LanceDBIndex


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
        index = LanceDBIndex(uri, name)
        index.create(3072)
        index.table.add([
            dict(id=f'{name}-{i}', source_kind=kind, source_identity=identity,
                 value=f'kept-{i}', vector=vector.tolist(), **{'class': label})
            for i, (identity, vector, label) in enumerate(zip(identities, values, ['A', 'B'], strict=True))
        ])
    config = dict(
        dataset={}, output_dir=str(tmp_path / 'output'), k=1,
        embed_images=True, embed_class_labels=False, embed_descriptions=False,
        n_imgs_per_class=1, cache_validation_embeddings=False,
        embedding=dict(model_id='google/ge-v2', batch_size=1, truncate_dim=None),
        lancedb=dict(uri=str(uri), table='reference', validation_table='validation'),
    )
    dataset = Mock()
    dataset.get_class_list.return_value = ['A', 'B']
    dataset.test_set.return_value = [0, 1]

    def batches(split, batch_size, *, decode):
        assert (split, batch_size, decode) == ('val', 1, False)
        for identity in ['first', 'second']:
            yield dict(source_id=[identity], source_split=['val'], fname=[identity], **{'class': ['A']})

    dataset.iter_batches.side_effect = batches
    monkeypatch.setattr('internvl.experiment.load_dataset', Mock(return_value=dataset))
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
    monkeypatch.setattr('internvl.tracking.wandb.Artifact', Artifact)
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
    for dimension, result, (run, kwargs) in zip(sweep.DIMENSIONS, summary, runs, strict=True):
        effective = kwargs['config']
        assert effective['k'] == 1
        assert effective['embedding']['truncate_dim'] == dimension
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
        for source, destination in zip(('reference', 'validation'), sweep.table_names(config, dimension), strict=True):
            table = connection.open_table(destination)
            assert table.schema.field('vector').type == pa.list_(pa.float32(), dimension)
            rows = table.to_arrow().to_pylist()
            expected = [dict(row, vector=row['vector'][:dimension]) for row in snapshots[source]]
            assert rows == expected
    assert [item['majority_vote_accuracy'] for item in summary] == [1, 0, 1]
    for name, snapshot in snapshots.items():
        assert connection.open_table(name).to_arrow().to_pylist() == snapshot


def test_reuse_has_no_writes_or_source_scan(sources, monkeypatch):
    config, connection, _, _ = sources
    sweep.prepare_table(connection, 'reference', 'reference_d768', 768, 1)
    wrapper = MagicMock(wraps=connection)
    wrapper.__contains__.side_effect = lambda name: name in connection
    monkeypatch.setattr(sweep, 'prefix_batches', Mock(side_effect=AssertionError('scan forbidden')))
    sweep.prepare_table(wrapper, 'reference', 'reference_d768', 768, 1)
    wrapper.create_table.assert_not_called()
    wrapper.open_table.assert_called_once_with('reference_d768')


@pytest.mark.parametrize('failure', [RuntimeError('stream failed'), KeyboardInterrupt()])
def test_streaming_failure_propagates(sources, monkeypatch, failure):
    _, connection, _, _ = sources
    original = sweep.prefix_batches
    counts = []

    def interrupted(*args):
        batches = original(*args)
        yield next(batches)
        raise failure

    monkeypatch.setattr(sweep, 'prefix_batches', interrupted)
    # Consume the same streaming reader at the create boundary to exercise interrupts
    # without relying on Arrow's translation of Python exceptions across native code.
    wrapper = MagicMock(wraps=connection)
    wrapper.__contains__.return_value = False

    def create(name, *, data, **kwargs):
        for batch in data:
            counts.append(batch.num_rows)

    wrapper.create_table.side_effect = create
    with pytest.raises(type(failure)):
        sweep.prepare_table(wrapper, 'reference', 'broken', 768, 1)
    wrapper.create_table.assert_called_once()
    assert counts == [1]
    wrapper.drop_table.assert_not_called()


@pytest.mark.parametrize('exists', [True, False])
def test_open_and_create_errors_propagate(sources, exists):
    _, connection, _, _ = sources
    wrapper = MagicMock(wraps=connection)
    wrapper.__contains__.return_value = exists
    failure = RuntimeError('storage failure')
    if exists:
        wrapper.open_table.side_effect = failure
    else:
        wrapper.create_table.side_effect = failure
    with pytest.raises(RuntimeError, match='storage failure'):
        sweep.prepare_table(wrapper, 'reference', 'destination', 768, 1)
    wrapper.drop_table.assert_not_called()


def test_missing_vector_has_no_embedding_or_summary(sources, runs):
    config, connection, _, _ = sources
    connection.open_table('validation').delete("source_identity = 'first'")
    with pytest.raises(KeyError, match='first'):
        cli._run_experiment(config, 'matroyshka-sweep')
    assert len(runs) == 1
    Experiment._create_embedder.assert_not_called()
    assert not list(Path(config['output_dir']).rglob('report.json'))
    assert not list(Path(config['output_dir']).rglob('summary.json'))


def test_later_failure_preserves_completed_report(sources, runs, monkeypatch):
    config, _, _, _ = sources
    original = cli.evaluate_cached

    def evaluate(effective, tracker):
        if effective['embedding']['truncate_dim'] == 1536:
            raise RuntimeError('later failure')
        return original(effective, tracker)

    monkeypatch.setattr(cli, 'evaluate_cached', evaluate)
    with pytest.raises(RuntimeError, match='later failure'):
        cli._run_experiment(config, 'matroyshka-sweep')
    assert len(list(Path(config['output_dir']).rglob('report.json'))) == 1
    assert not list(Path(config['output_dir']).rglob('summary.json'))
    assert len(runs) == 2
    assert runs[1][0].__exit__.call_args.args[0] is RuntimeError


@pytest.mark.parametrize('validation', ['reference', 'reference_d768', 'reference_d1536', ''])
def test_collisions_fail_before_tracking(sources, runs, validation):
    config, _, _, _ = sources
    config['lancedb']['validation_table'] = validation
    with pytest.raises(ValueError):
        cli._run_experiment(config, 'matroyshka-sweep')
    assert runs == []


def test_help_does_not_load_config(monkeypatch, capsys):
    load = Mock(side_effect=AssertionError('config load forbidden'))
    monkeypatch.setattr(cli, '_load_config', load)
    with pytest.raises(SystemExit) as error:
        cli.main(['experiment', 'matroyshka-sweep', '--help'])
    assert error.value.code == 0
    assert 'matroyshka-sweep' in capsys.readouterr().out
    load.assert_not_called()


def test_stream_has_no_default_result_limit(sources):
    _, connection, _, _ = sources
    source = connection.open_table('reference')
    rows = source.to_arrow().to_pylist()
    source.add([dict(rows[i % 2], id=f'extra-{i}', source_identity=f'extra-{i}') for i in range(23)])
    snapshot = source.to_arrow().to_pylist()
    sweep.prepare_table(connection, 'reference', 'many_d768', 768, 3)
    destination = connection.open_table('many_d768')
    assert destination.to_arrow().to_pylist() == [dict(row, vector=row['vector'][:768]) for row in snapshot]
    version = destination.version
    sweep.prepare_table(connection, 'reference', 'many_d768', 768, 3)
    assert connection.open_table('many_d768').version == version
    assert source.to_arrow().to_pylist() == snapshot


def test_native_stream_creation_error(sources, monkeypatch):
    _, connection, _, _ = sources
    original = sweep.prefix_batches

    def broken(*args):
        yield next(original(*args))
        raise RuntimeError('synthetic reader failure')

    monkeypatch.setattr(sweep, 'prefix_batches', broken)
    with pytest.raises(Exception, match='synthetic reader failure'):
        sweep.prepare_table(connection, 'reference', 'broken_native', 768, 1)
    # No assertion about table durability or repair after a native storage failure.
    assert connection.open_table('reference').count_rows() == 2
