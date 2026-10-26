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
from internvl.retrieval import LanceDBIndex


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

    def test_image_limit_validation(self):
        for enabled in (False, True):
            cfg = dict(embed_images=enabled, embed_class_labels=True, embed_descriptions=False)
            self.assertIs(validate_config(cfg), cfg)
            for limit in (None, 1, 20):
                cfg['n_imgs_per_class'] = limit
                self.assertIs(validate_config(cfg), cfg)
            for limit in (0, -1, True, False, '2', 1.0, [], {}):
                cfg['n_imgs_per_class'] = limit
                with self.subTest(enabled=enabled, limit=limit):
                    with self.assertRaisesRegex(ValueError, 'n_imgs_per_class.*positive integer'):
                        validate_config(cfg)

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


@unittest.skipUnless(available('lancedb', 'numpy', 'pandas', 'PIL', 'datasets'), 'requires datasets, lancedb and CPU fixture dependencies')
class ExperimentTests(unittest.TestCase):

    def test_shared_descriptions_are_read_only_across_runs(self):
        path = self.root / 'shared vocabulary.json'
        descriptions = {'Apple': ['Apple stored'], 'Pear': ['Pear stored']}
        content = json.dumps({'descriptions': descriptions})
        path.write_text(content)
        for name in ('first', 'second'):
            cfg = self.experiment((False, False, True), name).config
            cfg['descriptions_file'] = str(path)
            exp = Experiment(cfg)
            report = exp.run()
            self.assertEqual(report['reference_source_counts'], {'generated_description': 2})
            self.assertEqual(report['metrics']['evaluated'], 2)
            self.assertFalse((exp.output / 'descriptions.json').exists())
            table = self.client.open_table(cfg['lancedb']['table']).to_arrow().to_pylist()
            self.assertEqual({row['value'] for row in table}, {'Apple stored', 'Pear stored'})
        self.assertEqual(path.read_text(), content)
        self.describer_constructor.assert_not_called()

    def test_explicit_missing_descriptions_never_generate(self):
        cfg = self.experiment((False, False, True)).config
        cfg['descriptions_file'] = str(self.root / 'missing.json')
        exp = Experiment(cfg)
        exp.output.mkdir()
        (exp.output / 'descriptions.json').write_text(json.dumps({
            'descriptions': {'Apple': ['local'], 'Pear': ['local']}
        }))
        with self.assertRaises(FileNotFoundError):
            exp.run()
        with self.assertRaises(FileNotFoundError):
            exp._get_text_inputs()
        self.describer_constructor.assert_not_called()
        self.embedder_constructor.assert_not_called()
        self.client_constructor.assert_not_called()

    def test_null_descriptions_file_keeps_generation_in_output(self):
        cfg = self.experiment((False, False, True)).config
        cfg['descriptions_file'] = None
        exp = Experiment(cfg)
        descriptions = exp.generate_descriptions()
        self.assertEqual(
            json.loads((exp.output / 'descriptions.json').read_text())['descriptions'],
            descriptions,
        )
        self.describer_constructor.assert_called_once()

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
        with self.assertRaisesRegex(ValueError, 'No records inserted into LanceDB'):
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
        exp.config['lancedb']['table'] += '_new'
        with patch('internvl.experiment.Path.replace', side_effect=OSError('replace failed')):
            with self.assertRaisesRegex(OSError, 'replace failed'):
                exp.build_vector_database()
        self.assertEqual((exp.output / 'index.json').read_bytes(), previous)

    def test_early_ending_evaluation_is_incomplete(self):
        exp = self.experiment()
        exp.build_vector_database()
        original = self.dataset.iter_batches
        with patch.object(self.dataset, 'iter_batches', side_effect=lambda split, size, **kwargs: itertools.islice(original(split, 1, **kwargs), 1)):
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

                    def batches(split, size, **kwargs):
                        for batch in original(split, size, **kwargs):
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
        for stage, method in [('build_vector_database', 'append'), ('evaluate', 'query')]:
            for error in (ConnectionError('offline'), KeyboardInterrupt()):
                with self.subTest(stage=stage, error=type(error).__name__):
                    exp = self.experiment((True, False, False), f'{stage}-{type(error).__name__}')
                    if stage == 'evaluate':
                        exp.build_vector_database()
                    with patch.object(LanceDBIndex, method, side_effect=error):
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
        stored = self.client.open_table(artifact['table']).to_arrow().to_pylist()
        self.assertEqual(artifact['source_counts'], {'training_image': 3})
        self.assertEqual({p['source_identity'] for p in stored}, {'0', '1', 'extra'})
        self.assertEqual(len({p['id'] for p in stored}), 3)
        for point in stored:
            self.assertEqual(point['vector'], [0., 1.] if point['source_identity'] == '1' else [1., 0.])

    def test_failed_rebuild_preserves_previous_summary(self):
        exp = self.experiment((True, False, False))
        exp.batch_size = 1
        exp.build_vector_database()
        previous = (exp.output / 'index.json').read_bytes()
        exp.config['lancedb']['table'] += '_new'
        append = LanceDBIndex.append
        calls = 0

        def fail_second(index, records, vectors):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ConnectionError('second append failed')
            return append(index, records, vectors)

        with patch.object(LanceDBIndex, 'append', new=fail_second):
            with self.assertRaisesRegex(ConnectionError, 'second append'):
                exp.build_vector_database()
        self.assertEqual((exp.output / 'index.json').read_bytes(), previous)
        self.assertEqual(calls, 2)

    def test_image_references_still_require_every_class(self):
        self.dataset.splits['train'] = self.dataset.splits['train'].select([0])
        exp = self.experiment((True, False, False))
        exp.config['n_imgs_per_class'] = 1
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
        self.assertEqual(set(artifact), {'table', 'uri', 'source_counts', 'config'})
        self.assertEqual(set(report), {'config', 'predictions',
                                      'table', 'reference_source_counts', 'metrics'})
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
        import lancedb
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db_path = self.root / 'database'
        self.client = lancedb.connect(self.db_path)
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
        self.client_constructor = self.enterContext(patch('internvl.retrieval.lancedb.connect', wraps=lancedb.connect))

    def experiment(self, flags=(False, True, False), name='run'):
        cfg = dict(zip(('embed_images','embed_class_labels','embed_descriptions'), flags))
        cfg.update(output_dir=str(self.root / name), embedding={'batch_size': 2},
                   dataset={}, ollama={'llm':'fake'}, lancedb={'table': 'fixture_' + ''.join(c if c.isalnum() else '_' for c in name), 'uri': str(self.db_path)}, k=10,
                   description_count=5)
        return Experiment(cfg)

    def interleaved_training(self):
        from datasets import Dataset, Image as HFImage
        from PIL import Image

        rows = []
        for i, label in enumerate(['Pear', 'Apple', 'Pear', 'Pear', 'Apple']):
            path = self.root / f'interleaved-{i}.png'
            Image.new('RGB', (3, 4), (0 if label == 'Apple' else 100, i, 0)).save(path)
            rows.append(dict(fname=str(path), source_id=f'ref-{i}', source_split='train',
                             image=str(path), **{'class': label}))
        self.dataset.splits['train'] = Dataset.from_list(rows).cast_column('image', HFImage(decode=False))
        return rows

    def test_image_limits_preserve_order_alignment_and_splits(self):
        rows = self.interleaved_training()
        train = self.dataset.train_set()
        validation = self.dataset.test_set()
        for limit, selected in [(None, [0, 1, 2, 3, 4]), (1, [0, 1]), (2, [0, 1, 2, 4]), (20, [0, 1, 2, 3, 4])]:
            for size in (1, 3, 8):
                with self.subTest(limit=limit, size=size):
                    batches = list(self.dataset.iter_batches('train', size, n_imgs_per_class=limit))
                    self.assertEqual([sid for b in batches for sid in b['source_id']],
                                     [rows[i]['source_id'] for i in selected])
                    self.assertEqual([len(b['image']) for b in batches],
                                     [min(size, len(selected) - start) for start in range(0, len(selected), size)])
                    for batch in batches:
                        for sid, label, image in zip(batch['source_id'], batch['class'], batch['image'], strict=True):
                            i = int(sid.split('-')[1])
                            self.assertEqual(label, rows[i]['class'])
                            self.assertEqual(image.getpixel((0, 0))[1], i)
                            image.close()
        self.assertIs(self.dataset.train_set(), train)
        self.assertIs(self.dataset.test_set(), validation)
        self.assertFalse(train.features['image'].decode)
        self.assertEqual(len(train), 5)

    def test_capped_index_vectors_counts_and_full_evaluation(self):
        rows = self.interleaved_training()
        for limit, selected in [(None, [0, 1, 2, 3, 4]), (1, [0, 1]), (2, [0, 1, 2, 4]), (20, [0, 1, 2, 3, 4])]:
            with self.subTest(limit=limit):
                exp = self.experiment((True, True, True), f'limit-{limit}')
                exp.config['n_imgs_per_class'] = limit
                exp.batch_size = 3
                exp.generate_descriptions()
                seen = []
                encode = exp.embedder.encode_images
                def record(images):
                    seen.append([image.getpixel((0, 0))[1] for image in images])
                    return encode(images)
                with patch.object(exp.embedder, 'encode_images', side_effect=record):
                    artifact = exp.build_vector_database()
                self.assertEqual(seen, [selected[start:start + 3] for start in range(0, len(selected), 3)])
                self.assertEqual(artifact['source_counts'], {
                    'training_image': len(selected), 'class_name': 2, 'generated_description': 10,
                })
                stored = self.client.open_table(artifact['table']).to_arrow().to_pylist()
                images = [r for r in stored if r['source_kind'] == 'training_image']
                self.assertEqual([r['source_identity'] for r in images], [rows[i]['source_id'] for i in selected])
                for point, i in zip(images, selected, strict=True):
                    self.assertEqual(point['value'], rows[i]['fname'])
                    self.assertEqual(point['class'], rows[i]['class'])
                    self.assertEqual(point['vector'], [1., 0.] if rows[i]['class'] == 'Apple' else [0., 1.])
                report = exp.evaluate()
                self.assertEqual(report['metrics']['evaluated'], len(self.dataset.test_set()))
                self.assertEqual([p['source_id'] for p in report['predictions']], ['0', '1'])
                self.assertEqual(report['config']['n_imgs_per_class'], limit)

    def test_all_model_dimensions_index_and_evaluate(self):
        import numpy as np
        from internvl.embeddings import EmbeddingModel

        for model_id, native, truncate_dim in [
            ("openai/clip-vit-base-patch32", 512, None),
            ("google/embeddinggemma-2", 768, None),
            ("jinaai/jina-clip-v2", 1024, None),
            ("jinaai/jina-embeddings-v4", 2048, None),
            ("Qwen/Qwen3-VL-Embedding-8B", 4096, None),
            ("Qwen/Qwen3-VL-Embedding-8B", 4096, 128),
        ]:
            with self.subTest(model_id=model_id, truncate_dim=truncate_dim):
                dimension = truncate_dim or native
                def encode(inputs, **kwargs):
                    vectors = np.zeros((len(inputs), native), dtype=np.float32)
                    for i, item in enumerate(inputs):
                        if isinstance(item, dict):
                            item = item['image']
                        apple = 'Apple' in item if isinstance(item, str) else item.getpixel((0, 0))[0] == 0
                        vectors[i, 0 if apple else 1] = 1
                    return vectors[:, :kwargs.get("truncate_dim", native)]

                with patch('internvl.embeddings._load_backend') as loader:
                    loader.return_value.encode.side_effect = encode
                    model = EmbeddingModel(model_id, device='cpu', truncate_dim=truncate_dim)
                self.embedder_constructor.side_effect = None
                self.embedder_constructor.return_value = model
                exp = self.experiment((True, True, False), f'dim-{dimension}')
                exp.config['embedding']['model_id'] = model_id
                exp.config['embedding']['truncate_dim'] = truncate_dim
                artifact = exp.build_vector_database()
                table = self.client.open_table(artifact['table'])
                self.assertEqual(table.schema.field('vector').type.list_size, dimension)
                self.assertEqual(table.count_rows(), 4)
                report = exp.evaluate()
                self.assertEqual(report['metrics']['evaluated'], 2)
                self.assertEqual(report['metrics']['majority_vote_accuracy'], 1.0)
                self.assertEqual(report['metrics']['nearest_neighbor_accuracy'], 1.0)
                self.assertEqual([p['prediction'] for p in report['predictions']], ['Apple', 'Pear'])

    def test_excluded_unreadable_images_are_not_decoded(self):
        rows = self.interleaved_training()
        Path(rows[2]['fname']).unlink()
        Path(rows[3]['fname']).write_bytes(b'corrupt')
        exp = self.experiment((True, False, False), 'excluded')
        exp.config['n_imgs_per_class'] = 1
        self.assertEqual(exp.build_vector_database()['source_counts'], {'training_image': 2})
        for i, corrupt in ((0, False), (1, True)):
            path = Path(rows[i]['fname'])
            original = path.read_bytes()
            if corrupt:
                path.write_bytes(b'corrupt')
            else:
                path.unlink()
            exp = self.experiment((True, False, False), f'selected-{i}')
            exp.config['n_imgs_per_class'] = 1
            with self.assertRaises(OSError):
                exp.build_vector_database()
            self.assertFalse((exp.output / 'index.json').exists())
            path.write_bytes(original)

    def test_disabled_images_ignore_limit_and_preserve_text_counts(self):
        exp = self.experiment((False, True, True))
        exp.config['n_imgs_per_class'] = 1
        exp.generate_descriptions()
        with patch.object(self.dataset, 'iter_batches', side_effect=AssertionError('Read images')):
            artifact = exp.build_vector_database()
        self.assertEqual(artifact['source_counts'], {'class_name': 2, 'generated_description': 10})

    def test_point_records_round_trip_through_lancedb(self):
        exp = self.experiment()
        records = [PointRecord(label, 'class_name', label, label) for label in ['Apple', 'Pear']]
        self.assertTrue(all(isinstance(record, PointRecord) for record in records))
        index = LanceDBIndex(self.db_path, 'record_round_trip')
        index.create(2)
        vectors = exp.embedder.encode_text([record.value for record in records])
        index.append(records, vectors)
        stored = self.client.open_table('record_round_trip').to_arrow().to_pylist()
        payloads = {point['id']: {k: v for k, v in point.items() if k != 'vector'} for point in stored}
        for record in records:
            self.assertEqual(payloads[record.id], {
                'id': record.id, 'class': record.label, 'source_kind': record.source_kind,
                'source_identity': record.source_identity, 'value': record.value,
            })
        neighbor = index.query(vectors[0], 1)[0]
        self.assertEqual((neighbor['id'], neighbor['label']), (records[0].id, records[0].label))

    def test_combinations_and_table_reuse(self):
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
            table = exp.config['lancedb']['table']
            self.assertEqual(report['table'], table)
            with self.assertLogs(level='WARNING') as logs:
                self.assertIsNone(exp.build_vector_database())
            self.assertIn(table, logs.output[0])
            self.assertEqual(self.client.open_table(table).count_rows(), sum(expected.values()))
            with self.assertLogs(level='WARNING'):
                self.assertEqual(exp.run()['metrics']['evaluated'], 2)
            self.assertEqual(self.client.open_table(table).count_rows(), sum(expected.values()))

    def test_existing_table_skips_all_build_work(self):
        original = self.experiment()
        artifact = original.build_vector_database()
        snapshot = self.client.open_table(artifact['table']).to_arrow().to_pylist()
        previous = (original.output / 'index.json').read_bytes()
        with self.assertLogs(level='WARNING'):
            self.assertIsNone(original.build_vector_database())
        self.assertEqual((original.output / 'index.json').read_bytes(), previous)

        reused = self.experiment((True, False, True), 'reused')
        reused.config['lancedb']['table'] = artifact['table']
        reused.config['n_imgs_per_class'] = 1
        self.embedder_constructor.reset_mock()
        with patch.object(reused, '_get_text_inputs', side_effect=AssertionError('Read text')), \
             patch.object(self.dataset, 'iter_batches', side_effect=AssertionError('Read images')), \
             patch.object(LanceDBIndex, 'create', side_effect=AssertionError('Create table')), \
             patch.object(LanceDBIndex, 'append', side_effect=AssertionError('Insert records')), \
             self.assertLogs(level='WARNING') as logs:
            self.assertIsNone(reused.build_vector_database())
        self.assertIn(artifact['table'], logs.output[0])
        self.embedder_constructor.assert_not_called()
        self.assertFalse(reused.output.exists())
        self.assertEqual(self.client.open_table(artifact['table']).to_arrow().to_pylist(), snapshot)
        report = reused.evaluate()
        self.assertEqual(report['table'], artifact['table'])
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
        self.assertNotIn('model', report)
        self.assertIsNone(report['reference_source_counts'])

    def test_evaluation_uses_configured_table_with_stale_summary(self):
        exp = self.experiment()
        exp.build_vector_database()
        path = exp.output / 'index.json'
        summary = json.loads(path.read_text())
        summary['table'] = 'unrelated_table'
        path.write_text(json.dumps(summary))
        report = exp.evaluate()
        self.assertEqual(report['table'], exp.config['lancedb']['table'])
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
        self.assertNotIn('model', report)
        self.assertIsNone(report['reference_source_counts'])

    def test_summary_for_same_table_in_different_database_is_ignored(self):
        exp = self.experiment()
        exp.build_vector_database()
        path = exp.output / 'index.json'
        summary = json.loads(path.read_text())
        summary['uri'] = str(self.root / 'other_database')
        path.write_text(json.dumps(summary))
        self.assertIsNone(exp.evaluate()['reference_source_counts'])

    def test_evaluation_reads_read_only_database(self):
        exp = self.experiment()
        artifact = exp.build_vector_database()
        table = self.client.open_table(artifact['table'])
        version, rows = table.version, table.to_arrow()
        paths = [self.db_path, *self.db_path.rglob('*')]
        modes = {path: path.stat().st_mode for path in paths}
        try:
            for path in paths:
                path.chmod(0o555 if path.is_dir() else 0o444)
            with patch.object(LanceDBIndex, 'create', side_effect=AssertionError('write')), \
                 patch.object(LanceDBIndex, 'append', side_effect=AssertionError('write')):
                self.assertEqual(self.experiment().evaluate()['metrics']['evaluated'], 2)
            reopened = self.client.open_table(artifact['table'])
            self.assertEqual(reopened.version, version)
            self.assertEqual(reopened.to_arrow(), rows)
        finally:
            for path, mode in modes.items():
                path.chmod(mode)

    def test_evaluation_missing_database_or_table_creates_nothing(self):
        exp = self.experiment()
        for uri in (self.root / 'absent', self.db_path):
            exp.config['lancedb']['uri'] = str(uri)
            before = sorted(self.root.rglob('*'))
            with self.assertRaises((FileNotFoundError, ValueError)):
                exp.evaluate()
            self.assertEqual(sorted(self.root.rglob('*')), before)
            self.embedder_constructor.assert_not_called()

    def test_table_lookup_failure_propagates_without_building(self):
        exp = self.experiment()
        with patch.object(LanceDBIndex, 'exists', side_effect=ConnectionError('offline')) as exists:
            with self.assertRaisesRegex(ConnectionError, 'offline'):
                exp.build_vector_database()
        exists.assert_called_once_with()
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
        with patch('internvl.retrieval.LanceDBIndex.query', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError): exp.evaluate()
        self.assertFalse(list(exp.output.glob('evaluation_*/report.json')))
        self.dataset.splits['validation'] = self.dataset.splits['validation'].select([])
        with self.assertRaisesRegex(ValueError, 'No evaluation'): exp.evaluate()

    def test_empty_generation_and_missing_table(self):
        exp = self.experiment((False, False, True))
        with patch.object(self.Describer, 'describe_class', return_value=SimpleNamespace(description='')):
            with self.assertRaises(ValueError): exp.generate_descriptions()
        self.assertFalse((exp.output/'descriptions.json').exists())
        exp = self.experiment(name='valid')
        artifact = exp.build_vector_database()
        self.client.drop_table(artifact['table'])
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
        with patch('internvl.retrieval.LanceDBIndex.query', return_value=[]):
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
        stored = self.client.open_table(artifact['table']).to_arrow().to_pylist()
        self.assertEqual(len(stored), 2)
        for point in stored:
            row = next(r for r in self.rows if r['source_id'] == point['source_identity'])
            self.assertEqual({k: v for k, v in point.items() if k != 'vector'}, dict(id=point['id'], source_kind='training_image',
                source_identity=row['source_id'], value=row['fname'], **{'class': row['class']}))
            self.assertEqual(point['vector'], [1., 0.] if row['class'] == 'Apple' else [0., 1.])
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
