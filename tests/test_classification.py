"""Synthetic CPU-only contracts for whole-image experiments."""
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from internvl.experiment import Experiment, _batch_rows, _prediction, validate_config
from internvl.schemas import Neighbor, PointRecord
from internvl.retrieval import QdrantIndex


def available(*modules):
    return all(importlib.util.find_spec(module) is not None for module in modules)


class Contracts(unittest.TestCase):

    def test_prediction_preserves_metadata_and_neighbor_order(self):
        batch = {
            'fname': ['fixture.png'], 'class': ['A'], 'source_id': ['image-1'],
            'source_split': ['val'], 'extra': ['retained'], 'image': [object()],
        }
        record = _batch_rows(batch)[0]
        original = dict(record)
        cases = [
            (
                [Neighbor(id='first', label='B', score=0.9), Neighbor(id='second', label='A', score=0.8),
                 Neighbor(id='third', label='A', score=0.7)],
                {
                    'prediction': 'A', 'vote_fraction': 2 / 3, 'nearest': 'B',
                    'neighbors': [
                        {'id': 'first', 'label': 'B', 'score': 0.9},
                        {'id': 'second', 'label': 'A', 'score': 0.8},
                        {'id': 'third', 'label': 'A', 'score': 0.7},
                    ],
                },
            ),
            ([], {'prediction': None, 'vote_fraction': 0.0,
                  'nearest': None, 'neighbors': []}),
        ]
        for neighbors, result in cases:
            with self.subTest(neighbors=neighbors):
                prediction = _prediction(record, neighbors)
                self.assertEqual(prediction, {
                    'fname': 'fixture.png', 'class': 'A', 'source_id': 'image-1',
                    'source_split': 'val', 'extra': 'retained', **result,
                })
                self.assertEqual(record, original)
                self.assertIsNot(prediction, record)
                self.assertEqual(json.loads(json.dumps(prediction)), prediction)

    def test_batch_rows_requires_paired_metadata_columns(self):
        with self.assertRaises(ValueError):
            _batch_rows({'fname': ['first', 'second'], 'class': ['A']})

    def test_reference_switch_combinations(self):
        keys = ('embed_images', 'embed_class_labels', 'embed_descriptions')
        for flags in itertools.product([False, True], repeat=3):
            cfg = dict(zip(keys, flags))
            with self.subTest(flags=flags):
                if any(flags):
                    self.assertIs(validate_config(cfg), cfg)
                else:
                    with self.assertRaisesRegex(ValueError, 'at least one reference source'):
                        validate_config(cfg)
                    with self.assertRaisesRegex(ValueError, 'at least one reference source'):
                        Experiment(cfg)

    def test_reference_switches_require_booleans(self):
        for key in ('embed_images', 'embed_class_labels', 'embed_descriptions'):
            for value in ('false', 0, 1, None):
                cfg = dict(embed_images=False, embed_class_labels=True, embed_descriptions=False)
                cfg[key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        validate_config(cfg)
                    with self.assertRaisesRegex(ValueError, key):
                        Experiment(cfg)

    def test_output_cannot_be_inside_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = dict(embed_images=True, embed_class_labels=False, embed_descriptions=False,
                       dataset={'root': directory}, output_dir=str(Path(directory) / 'derived'))
            with self.assertRaisesRegex(ValueError, 'separate from raw data'):
                Experiment(cfg)
            self.assertFalse((Path(directory) / 'derived').exists())



@unittest.skipUnless(available('numpy', 'PIL', 'pandas', 'tqdm', 'datasets'), 'requires datasets and CPU image/dataframe dependencies')
class DatasetTests(unittest.TestCase):
    def test_native_dataset_contract(self):
        from datasets import DatasetDict, Image as HFImage
        from PIL import Image
        from internvl.image_dataset import FoodX251ImageDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'annot').mkdir()
            (root / 'annot/class_list.txt').write_text('7 apple_PIE\n2 pear\n')
            for split, filename, color in [
                ('train', 'pear,one.png', 'green'),
                ('train', 'pie.png', 'red'),
                ('val', 'pie.png', 'blue'),
            ]:
                folder = root / split / f'{split}_set'
                folder.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (8, 4), color).save(folder / filename)
            (root / 'annot/train_info.csv').write_text('"pear,one.png",2\npie.png,7\n')
            (root / 'annot/val_info.csv').write_text('pie.png,7\n')
            # No official test files are needed, and metadata loading does not decode.
            with patch('PIL.Image.open', side_effect=AssertionError('Eager image decode')):
                ds = FoodX251ImageDataset(root)
            self.assertIsInstance(ds.splits, DatasetDict)
            self.assertEqual(set(ds.splits), {'train', 'validation'})
            self.assertEqual(ds.get_class_list(), ['Apple pie', 'Pear'])
            self.assertEqual(list(ds.train_set()['class']), ['Pear', 'Apple pie'])
            self.assertEqual(ds.train_path, root / 'train/train_set')
            self.assertEqual(ds.test_path, root / 'val/val_set')
            self.assertEqual(str(ds.test_set().split), 'validation')
            self.assertEqual(ds.train_set().features['image'], HFImage(decode=False))
            batch = next(ds.iter_batches('validation', 2))
            self.assertEqual(batch['source_split'], ['val'])
            self.assertEqual(batch['image'][0].getpixel((0, 0)), (0, 0, 255))
            batch['image'][0].close()
            (root / 'annot/val_info.csv').write_text('')
            empty = FoodX251ImageDataset(root)
            self.assertEqual(len(empty.test_set()), 0)
            self.assertEqual(empty.test_set().features, empty.train_set().features)
            self.assertEqual(list(empty.iter_batches('val', 2)), [])
            (root / 'annot/train_info.csv').write_text('pie.png,7\npie.png,7\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate source'):
                FoodX251ImageDataset(root)

    def test_dataset_and_leakage(self):
        from PIL import Image
        from internvl.image_dataset import FoodX251ImageDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'annot').mkdir()
            (root / 'annot/class_list.txt').write_text('0 apple_pie\n1 pear\n')
            for split, color in [('train', 'red'), ('val', 'green')]:
                folder = root / split / f'{split}_set'
                folder.mkdir(parents=True)
                Image.new('RGB', (8, 4), color).save(folder / 'a.png')
                (root / f'annot/{split}_info.csv').write_text('a.png,0\n')
            ds = FoodX251ImageDataset(root)
            self.assertEqual(ds.test_set()[0]['source_split'], 'val')
            self.assertEqual(ds.train_set()[0]['class'], 'Apple pie')
            self.assertTrue(Path(ds.train_set()[0]['fname']).is_absolute())
            copied = ds.train_set()[0]; copied['class'] = 'mutated'
            self.assertEqual(ds.train_set()[0]['class'], 'Apple pie')
            (root / 'val/val_set/a.png').write_bytes((root / 'train/train_set/a.png').read_bytes())
            duplicate_content = FoodX251ImageDataset(root)
            self.assertNotIn('sha256', duplicate_content.train_set().column_names)
            self.assertNotIn('sha256', duplicate_content.test_set().column_names)
            (root / 'val/val_set/a.png').unlink()
            (root / 'val/val_set/a.png').symlink_to(root / 'train/train_set/a.png')
            with self.assertRaises(ValueError): FoodX251ImageDataset(root)
            (root / 'annot/train_info.csv').write_text('a.png,99\n')
            with self.assertRaises(KeyError): FoodX251ImageDataset(root)


@unittest.skipUnless(available('qdrant_client', 'numpy', 'pandas', 'PIL', 'datasets'), 'requires datasets, qdrant-client and CPU fixture dependencies')
class ExperimentTests(unittest.TestCase):

    def test_constructor_loads_metadata_but_defers_backends(self):
        with patch.object(self.dataset, 'get_class_list', wraps=self.dataset.get_class_list) as classes:
            exp = self.experiment()
            self.dataset_constructor.assert_called_once_with({})
            classes.assert_called_once()
            self.assertFalse(exp.output.exists())
            self.assertIsNone(exp.generate_descriptions())
            exp.output.mkdir()
            descriptions = {'Apple': ['Apple stored'], 'Pear': ['Pear stored']}
            (exp.output / 'descriptions.json').write_text(json.dumps({'descriptions': descriptions}))
            exp.config['embed_descriptions'] = True
            self.assertEqual(exp.generate_descriptions(), descriptions)
            self.embedder_constructor.assert_not_called()
            self.client_constructor.assert_not_called()
            self.describer_constructor.assert_not_called()
            exp.run()
            exp.evaluate()
            classes.assert_called_once()
            self.embedder_constructor.assert_called_once_with(batch_size=2)
            self.client_constructor.assert_called_once_with()
        self.dataset_constructor.side_effect = FileNotFoundError('annotations')
        with self.assertRaisesRegex(FileNotFoundError, 'annotations'):
            Experiment(exp.config)

    def test_default_batch_size_is_stored_once(self):
        cfg = self.experiment().config
        del cfg['embedding']['batch_size']
        exp = Experiment(cfg)
        cfg['embedding']['batch_size'] = 7
        exp.build_vector_database()
        self.assertEqual(exp.batch_size, 32)
        self.embedder_constructor.assert_called_once_with(batch_size=32)

    def test_empty_index_is_rejected(self):
        exp = self.experiment((False, False, True))
        exp.output.mkdir()
        (exp.output / 'descriptions.json').write_text(json.dumps({
            'descriptions': {'Apple': [], 'Pear': []}
        }))
        with self.assertRaisesRegex(ValueError, 'No records inserted into Qdrant'):
            exp.build_vector_database()
        self.assertFalse((exp.output / 'index.json').exists())

    def test_description_errors_and_interrupts_exit_context(self):
        for error in (RuntimeError('generation failed'), KeyboardInterrupt()):
            exp = self.experiment((False, False, True))
            with patch.object(self.Describer, 'describe_class', side_effect=error) as generate, \
                 patch.object(self.Describer, '__exit__', return_value=None) as close:
                with self.assertRaises(type(error)) as raised:
                    exp.run()
                self.assertIs(raised.exception, error)
                generate.assert_called_once()
                close.assert_called_once()
            self.assertFalse((exp.output / 'descriptions.json').exists())
            self.embedder_constructor.assert_not_called()

    def test_failed_index_write_preserves_previous_summary(self):
        exp = self.experiment()
        exp.build_vector_database()
        previous = (exp.output / 'index.json').read_bytes()
        exp.config['qdrant']['collection_prefix'] += '_new'
        with patch('internvl.experiment.Path.replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                exp.build_vector_database()
        self.assertEqual((exp.output / 'index.json').read_bytes(), previous)

    def test_early_ending_evaluation_is_incomplete(self):
        exp = self.experiment()
        exp.build_vector_database()
        original = self.dataset.iter_batches
        with patch.object(self.dataset, 'iter_batches', side_effect=lambda split, size: itertools.islice(original(split, 1), 1)):
            with self.assertRaisesRegex(ValueError, 'Incomplete evaluation'):
                exp.evaluate()
        self.assertFalse(list(exp.output.glob('evaluation_*/report.json')))


    def test_successful_processing_still_fails_if_report_write_fails(self):
        from internvl.experiment import _write_json
        for stage in ('build_vector_database', 'evaluate'):
            with self.subTest(stage=stage):
                exp = self.experiment(name=stage)
                if stage == 'evaluate':
                    exp.build_vector_database()

                def write(path, value):
                    if isinstance(value, dict):
                        raise OSError('report write failed')
                    return _write_json(path, value)

                with patch('internvl.experiment._write_json', side_effect=write):
                    with self.assertRaisesRegex(OSError, 'report write failed'):
                        getattr(exp, stage)()

    def test_bad_batch_columns_and_vector_counts_fail(self):
        original = self.dataset.iter_batches
        for stage in ('build_vector_database', 'evaluate'):
            for failure in ('columns', 'vectors'):
                with self.subTest(stage=stage, failure=failure):
                    exp = self.experiment((True, False, False), f'{stage}-{failure}')
                    if stage == 'evaluate':
                        exp.build_vector_database()

                    def batches(split, size):
                        for batch in original(split, size):
                            if failure == 'columns':
                                batch['source_id'] = batch['source_id'][:-1]
                            yield batch

                    encode = exp.embedder.encode_images
                    def short_vectors(batch):
                        return encode(batch)[:-1]

                    with patch.object(self.dataset, 'iter_batches', side_effect=batches), \
                         patch.object(exp.embedder, 'encode_images', side_effect=short_vectors if failure == 'vectors' else encode):
                        with self.assertRaises(ValueError):
                            getattr(exp, stage)()
                    self.assert_no_failed_report(exp, stage)

    def test_database_failures_and_interrupts_propagate(self):
        for stage, method in [('build_vector_database', 'upsert'), ('evaluate', 'query')]:
            for error in (ConnectionError('offline'), KeyboardInterrupt()):
                with self.subTest(stage=stage, error=type(error).__name__):
                    exp = self.experiment((True, False, False), f'{stage}-{type(error).__name__}')
                    if stage == 'evaluate':
                        exp.build_vector_database()
                    with patch.object(QdrantIndex, method, side_effect=error):
                        with self.assertRaises(type(error)) as raised:
                            getattr(exp, stage)()
                    self.assertIs(raised.exception, error)
                    self.assert_no_failed_report(exp, stage)

    def test_training_partial_batches_keep_records_and_counts(self):
        from datasets import Dataset, Image as HFImage, concatenate_datasets
        extra = dict(self.rows[0], source_id='extra', image=self.rows[0]['fname'])
        self.dataset.splits['train'] = concatenate_datasets([
            self.dataset.splits['train'],
            Dataset.from_list([extra]).cast_column('image', HFImage(decode=False)),
        ])
        exp = self.experiment((True, False, False))
        with patch.object(self.dataset, 'train_set', side_effect=AssertionError('Extra metadata pass')):
            artifact = exp.build_vector_database()
        stored, _ = self.client.scroll(artifact['collection'], with_vectors=True)
        self.assertEqual(artifact['source_counts'], {'training_image': 3})
        self.assertEqual({p.payload['source_identity'] for p in stored}, {'0', '1', 'extra'})
        self.assertEqual(len({p.id for p in stored}), 3)
        for point in stored:
            self.assertEqual(point.vector, [0., 1.] if point.payload['source_identity'] == '1' else [1., 0.])

    def test_failed_rebuild_preserves_previous_summary(self):
        exp = self.experiment((True, False, False))
        exp.batch_size = 1
        exp.build_vector_database()
        previous = (exp.output / 'index.json').read_bytes()
        exp.config['qdrant']['collection_prefix'] += '_new'
        upsert = QdrantIndex.upsert
        calls = 0

        def fail_second(index, records, vectors):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ConnectionError('second upsert failed')
            return upsert(index, records, vectors)

        with patch.object(QdrantIndex, 'upsert', new=fail_second):
            with self.assertRaisesRegex(ConnectionError, 'second upsert'):
                exp.build_vector_database()
        self.assertEqual((exp.output / 'index.json').read_bytes(), previous)
        self.assertEqual(calls, 2)

    def test_image_references_still_require_every_class(self):
        self.dataset.splits['train'] = self.dataset.splits['train'].select([0])
        exp = self.experiment((True, False, False))
        with self.assertRaisesRegex(ValueError, 'Missing usable training references'):
            exp.build_vector_database()
        self.assertFalse((exp.output / 'index.json').exists())

    def test_artifact_formats_and_direct_loading_remain_compatible(self):
        exp = self.experiment((False, False, True))
        exp.output.mkdir()
        descriptions = {'Apple': ['Apple stored'], 'Pear': ['Pear stored']}
        (exp.output / 'descriptions.json').write_text(json.dumps({'descriptions': descriptions}))
        self.describer_constructor.side_effect = AssertionError('Cached descriptions regenerated')
        report = exp.run()
        artifact = json.loads((exp.output / 'index.json').read_text())
        self.assertEqual(set(artifact), {'collection', 'source_counts', 'config'})
        self.assertEqual(set(report), {'config', 'predictions',
                                      'collection', 'reference_source_counts', 'metrics'})
        folder = next(exp.output.glob('evaluation_*'))
        self.assertEqual([p.name for p in folder.iterdir()], ['report.json'])
        self.assertEqual(json.loads((folder / 'report.json').read_text()), report)
        self.assertTrue(all('value' not in p for p in report['predictions']))
        self.assertEqual(set(report['metrics']), {'evaluated', 'majority_vote_accuracy',
                                                'nearest_neighbor_accuracy', 'per_class'})
        artifact['state'] = 'incomplete'
        (exp.output / 'index.json').write_text(json.dumps(artifact))
        self.assertEqual(exp.evaluate()['metrics']['evaluated'], 2)


    def setUp(self):
        import numpy as np
        from datasets import Dataset, DatasetDict, Image as HFImage
        from internvl.image_dataset import FoodX251ImageDataset
        from PIL import Image
        from qdrant_client import QdrantClient
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = QdrantClient(':memory:'); self.addCleanup(self.client.close)
        rows = []
        for i, label in enumerate(['Apple', 'Pear']):
            path = self.root / f'{i}.png'; Image.new('RGB', (3, 4), (i*100, 0, 0)).save(path)
            rows.append(dict(fname=str(path), source_id=str(i), source_split='train', **{'class': label}))
        self.rows = rows
        self.dataset = FoodX251ImageDataset.__new__(FoodX251ImageDataset)
        self.dataset.class_map = {0: 'Apple', 1: 'Pear'}
        split = Dataset.from_list([dict(row, image=row['fname']) for row in rows]).cast_column('image', HFImage(decode=False))
        self.dataset.splits = DatasetDict(train=split, validation=split)
        class Embedder:
            dimension = 2
            def __init__(self, **kwargs): pass
            def encode_text(self, texts):
                return np.array([[1., 0.] if 'Apple' in x else [0., 1.] for x in texts])
            def encode_images(self, images):
                return np.array([[1., 0.] if image.getpixel((0, 0))[0] == 0 else [0., 1.] for image in images])
        from internvl.generate_descriptors import DescriberLLM
        class Describer:
            generate_descriptions = DescriberLLM.generate_descriptions
            def __init__(self, config): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def describe_class(self, label): return SimpleNamespace(description=label + ' description')
        self.Embedder, self.Describer = Embedder, Describer
        self.dataset_constructor = self.enterContext(patch('internvl.experiment.load_dataset', return_value=self.dataset))
        self.embedder_constructor = self.enterContext(patch('internvl.experiment.EmbeddingModel', side_effect=Embedder))
        self.describer_constructor = self.enterContext(patch('internvl.experiment.DescriberLLM', side_effect=Describer))
        self.client_constructor = self.enterContext(patch('internvl.experiment.QdrantClient', return_value=self.client))

    def experiment(self, flags=(False, True, False), name='run'):
        cfg = dict(zip(('embed_images','embed_class_labels','embed_descriptions'), flags))
        cfg.update(output_dir=str(self.root / name), embedding={'batch_size': 2},
                   dataset={}, ollama={'llm':'fake'}, qdrant={'collection_prefix':f'fixture_{name}', 'connection': {}}, k=10,
                   description_count=5)
        return Experiment(cfg)

    def test_point_records_round_trip_through_qdrant(self):
        exp = self.experiment()
        records = [PointRecord(label, 'class_name', label, label) for label in ['Apple', 'Pear']]
        self.assertTrue(all(isinstance(record, PointRecord) for record in records))
        index = QdrantIndex(self.client, 'record_round_trip', 2)
        index.create()
        vectors = exp.embedder.encode_text([record.value for record in records])
        index.upsert(records, vectors)
        stored = self.client.retrieve('record_round_trip', ids=[record.id for record in records])
        payloads = {str(point.id): point.payload for point in stored}
        for record in records:
            self.assertEqual(payloads[record.id], {
                'id': record.id, 'class': record.label, 'source_kind': record.source_kind,
                'source_identity': record.source_identity, 'value': record.value,
            })
        neighbor = index.query(vectors[0], 1)[0]
        self.assertEqual((neighbor['id'], neighbor['label']), (records[0].id, records[0].label))

    def test_combinations_and_collection_reuse(self):
        for flags in itertools.product([False, True], repeat=3):
            if not any(flags): continue
            exp = self.experiment(flags, str(flags))
            exp.batch_size = 3
            report = exp.run()
            expected = {kind: count for enabled, kind, count in zip(
                flags, ('training_image', 'class_name', 'generated_description'), (2, 2, 10)
            ) if enabled}
            self.assertEqual(report['reference_source_counts'], expected)
            self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
            self.assertEqual((exp.output / 'descriptions.json').exists(), flags[2])
            collection = exp.config['qdrant']['collection_prefix']
            self.assertEqual(report['collection'], collection)
            with self.assertLogs(level='WARNING') as logs:
                self.assertIsNone(exp.build_vector_database())
            self.assertIn(collection, logs.output[0])
            self.assertEqual(self.client.count(collection).count, sum(expected.values()))
            with self.assertLogs(level='WARNING'):
                self.assertEqual(exp.run()['metrics']['evaluated'], 2)
            self.assertEqual(self.client.count(collection).count, sum(expected.values()))

    def test_existing_collection_skips_all_build_work(self):
        original = self.experiment()
        artifact = original.build_vector_database()
        snapshot = self.client.scroll(artifact['collection'], with_vectors=True)
        previous = (original.output / 'index.json').read_bytes()
        with self.assertLogs(level='WARNING'):
            self.assertIsNone(original.build_vector_database())
        self.assertEqual((original.output / 'index.json').read_bytes(), previous)

        reused = self.experiment((True, False, True), 'reused')
        reused.config['qdrant']['collection_prefix'] = artifact['collection']
        self.embedder_constructor.reset_mock()
        with patch.object(reused, '_get_text_inputs', side_effect=AssertionError('Read text')), \
             patch.object(self.dataset, 'iter_batches', side_effect=AssertionError('Read images')), \
             patch.object(QdrantIndex, 'create', side_effect=AssertionError('Create collection')), \
             patch.object(QdrantIndex, 'upsert', side_effect=AssertionError('Insert records')), \
             self.assertLogs(level='WARNING') as logs:
            self.assertIsNone(reused.build_vector_database())
        self.assertIn(artifact['collection'], logs.output[0])
        self.embedder_constructor.assert_not_called()
        self.assertFalse(reused.output.exists())
        self.assertEqual(self.client.scroll(artifact['collection'], with_vectors=True), snapshot)
        report = reused.evaluate()
        self.assertEqual(report['collection'], artifact['collection'])
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
        self.assertNotIn('model', report)
        self.assertIsNone(report['reference_source_counts'])

    def test_evaluation_uses_configured_collection_with_stale_summary(self):
        exp = self.experiment()
        exp.build_vector_database()
        path = exp.output / 'index.json'
        summary = json.loads(path.read_text())
        summary['collection'] = 'unrelated_collection'
        path.write_text(json.dumps(summary))
        report = exp.evaluate()
        self.assertEqual(report['collection'], exp.config['qdrant']['collection_prefix'])
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
        self.assertNotIn('model', report)
        self.assertIsNone(report['reference_source_counts'])

    def test_collection_lookup_failure_propagates_without_building(self):
        exp = self.experiment()
        with patch.object(self.client, 'collection_exists', side_effect=ConnectionError('offline')) as exists:
            with self.assertRaisesRegex(ConnectionError, 'offline'):
                exp.build_vector_database()
        exists.assert_called_once_with(exp.config['qdrant']['collection_prefix'])
        self.embedder_constructor.assert_not_called()
        self.assertFalse(exp.output.exists())

    def assert_no_failed_report(self, exp, stage):
        if stage == 'build_vector_database':
            self.assertFalse((exp.output / 'index.json').exists())
        self.assertFalse(list(exp.output.glob('evaluation_*/report.json')))

    def test_missing_and_corrupt_images_abort(self):
        for corrupt in [False, True]:
            with self.subTest(corrupt=corrupt):
                path = Path(self.rows[1]['fname'])
                if corrupt:
                    path.write_bytes(b'not an image')
                else:
                    path.unlink()
                exp = self.experiment((True, False, False), f'bad-index-{corrupt}')
                with self.assertRaises(OSError): exp.build_vector_database()
                self.assertFalse((exp.output / 'index.json').exists())
                exp = self.experiment(name=f'bad-eval-{corrupt}')
                exp.build_vector_database()
                with self.assertRaises(OSError): exp.evaluate()
                self.assertFalse(list(exp.output.glob('evaluation_*/report.json')))

    def test_query_failure_and_empty_evaluation(self):
        exp = self.experiment(); exp.build_vector_database()
        with patch('internvl.retrieval.QdrantIndex.query', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError): exp.evaluate()
        self.assertFalse(list(exp.output.glob('evaluation_*/report.json')))
        self.dataset.splits['validation'] = self.dataset.splits['validation'].select([])
        with self.assertRaisesRegex(ValueError, 'No evaluation'): exp.evaluate()

    def test_empty_generation_and_missing_collection(self):
        exp = self.experiment((False, False, True))
        with patch.object(self.Describer, 'describe_class', return_value=SimpleNamespace(description='')):
            with self.assertRaises(ValueError): exp.generate_descriptions()
        self.assertFalse((exp.output/'descriptions.json').exists())
        exp = self.experiment(name='valid')
        artifact = exp.build_vector_database()
        self.client.delete_collection(artifact['collection'])
        with self.assertRaises(Exception): exp.evaluate()

    def test_embedding_failure_aborts_without_retry(self):
        for stage in ['build_vector_database', 'evaluate']:
            with self.subTest(stage=stage):
                exp = self.experiment((True, False, False), stage)
                if stage == 'evaluate': exp.build_vector_database()
                with patch.object(exp.embedder, 'encode_images', side_effect=ValueError('synthetic failure')) as encode:
                    with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                        getattr(exp, stage)()
                encode.assert_called_once()
                self.assert_no_failed_report(exp, stage)

    def test_empty_retrieval(self):
        exp = self.experiment(); exp.build_vector_database()
        with patch('internvl.retrieval.QdrantIndex.query', return_value=[]):
            report = exp.evaluate()
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 0)
        self.assertTrue(all(p['prediction'] is None for p in report['predictions']))

    def test_batch_order_partial_batch_and_payloads(self):
        from datasets import concatenate_datasets
        self.dataset.splits['validation'] = concatenate_datasets([self.dataset.splits['validation'], self.dataset.splits['validation'].select([0])])
        batches = list(self.dataset.iter_batches('val', 2))
        self.assertEqual([len(b['image']) for b in batches], [2, 1])
        self.assertEqual([sid for b in batches for sid in b['source_id']], ['0', '1', '0'])
        for batch in batches:
            for source_id, image in zip(batch['source_id'], batch['image']):
                self.assertEqual(image.size, (3, 4))
                self.assertEqual(image.getpixel((0, 0))[0], int(source_id) * 100)
                image.close()
        exp = self.experiment((True, False, False))
        artifact = exp.build_vector_database()
        stored, _ = self.client.scroll(artifact['collection'], with_vectors=True)
        self.assertEqual(len(stored), 2)
        for point in stored:
            row = next(r for r in self.rows if r['source_id'] == point.payload['source_identity'])
            self.assertEqual(point.payload, dict(id=str(point.id), source_kind='training_image',
                source_identity=row['source_id'], value=row['fname'], **{'class': row['class']}))
            self.assertEqual(point.vector, [1., 0.] if row['class'] == 'Apple' else [0., 1.])
        report = exp.evaluate()
        self.assertEqual([p['source_id'] for p in report['predictions']], ['0', '1', '0'])
        self.assertEqual(report['metrics']['evaluated'], 3)
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)

    def test_disabled_description_and_ollama_closes_before_embedder(self):
        events = []
        parent = self.Describer
        class Describer(parent):
            def __exit__(self, *args): events.append('closed')
        parent_embed = self.Embedder
        class Embedder(parent_embed):
            def __init__(self, **kwargs):
                if events != ['closed']: raise AssertionError('Ollama not closed first')
        exp = self.experiment((False, False, True))
        self.describer_constructor.side_effect = Describer
        self.embedder_constructor.side_effect = Embedder
        exp.run()
        exp = self.experiment(name='disabled')
        self.describer_constructor.side_effect = AssertionError('Disabled backend constructed')
        exp.generate_descriptions()
        self.assertIsNone(exp._embedder)
        exp.run()


if __name__ == '__main__': unittest.main()
