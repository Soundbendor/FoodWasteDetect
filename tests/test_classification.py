"""Synthetic CPU-only contracts for whole-image experiments."""
import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from internvl.experiment import Experiment, validate_config
from internvl.records import ReferenceRecord
from internvl.retrieval import Neighbor, QdrantIndex, classify, metrics
from test_internvl import run_python, IMPORT_GUARD, available


class Contracts(unittest.TestCase):
    def test_switches(self):
        for flags in itertools.product([False, True], repeat=3):
            cfg = dict(zip(('embed_images', 'embed_class_labels', 'embed_descriptions'), flags))
            if any(flags):
                self.assertEqual(validate_config(cfg), cfg)
            else:
                with self.assertRaises(ValueError): validate_config(cfg)

    def test_vote_and_denominator(self):
        self.assertEqual(classify([]), (None, 0))
        neighbors = [Neighbor('1', 'B', .9), Neighbor('2', 'A', .8)]
        self.assertEqual(classify(neighbors), ('B', .5))
        self.assertEqual(classify([Neighbor('1', 'B', .8), Neighbor('2', 'A', .8)]), ('A', .5))
        predictions = [dict(prediction='A' if i < 63 else None, nearest='A', **{'class': 'A'}) for i in range(90)]
        result = metrics(predictions, 100, 92, 90, ['A', 'B'])
        self.assertEqual(result['majority_vote_accuracy'], .7)
        self.assertEqual(result['coverage'], .9)
        self.assertIsNone(result['per_class']['B']['recall'])

    def test_import_help_and_retirement(self):
        result = run_python("forbidden = {'numpy','pandas','torch','transformers','PIL','yaml','ollama','qdrant_client','datasets'}\n" + IMPORT_GUARD + '''
import internvl.experiment
import internvl.embeddings
import internvl.retrieval
try:
    internvl.experiment.main(['--help'])
except SystemExit as e:
    assert e.code == 0
''')
        self.assertEqual(result.returncode, 0, result.stderr)
        from internvl.cli import main
        for command in ['build-db', 'evaluate']:
            with contextlib.redirect_stderr(io.StringIO()) as stderr, self.assertRaises(SystemExit):
                main([command, '--config_file', 'unused'])
            self.assertIn('retired', stderr.getvalue())


@unittest.skipUnless(available('numpy', 'PIL', 'pandas', 'tqdm', 'datasets'), 'requires datasets and CPU image/dataframe dependencies')
class DatasetAndEmbedding(unittest.TestCase):
    def test_native_dataset_contract_without_legacy_imports(self):
        import builtins
        from datasets import DatasetDict, Image as HFImage
        from PIL import Image
        from internvl.image_dataset import FoodX251ImageDataset

        original_import = builtins.__import__

        def import_without_legacy(name, *args, **kwargs):
            if name == 'data_wrappers' or name.startswith('data_wrappers.'):
                raise AssertionError(f'Legacy dataset import: {name}')
            return original_import(name, *args, **kwargs)

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
            with patch('builtins.__import__', side_effect=import_without_legacy), \
                 patch('PIL.Image.open', side_effect=AssertionError('Eager image decode')):
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

    def test_embeddings(self):
        import numpy as np
        from PIL import Image
        from internvl.embeddings import EmbeddingModel
        captured = []
        class Backend:
            def encode_text(self, texts, **kwargs):
                return np.array([[i + 1] + [1] * 1023 for i in map(int, texts)])
            def encode_image(self, images, **kwargs):
                captured.extend(images)
                assert all(image.mode == 'RGB' for image in images)
                return np.ones((len(images), 1024))
        model = EmbeddingModel('jinaai/jina-clip-v2', 'cpu', 2, backend=Backend())
        values = model.encode_text(['0', '1', '2'])
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)
        self.assertLess(values[0, 0], values[2, 0])
        self.assertEqual(model.encode_images([]).shape, (0, 1024))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'; Image.new('L', (9, 3)).save(path)
            self.assertEqual(model.encode_images([path]).shape, (1, 1024))
            with self.assertRaises(ValueError): captured[0].getpixel((0, 0))


@unittest.skipUnless(available('qdrant_client', 'numpy', 'pandas', 'PIL', 'datasets'), 'requires datasets, qdrant-client and CPU fixture dependencies')
class ExperimentTests(unittest.TestCase):
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
        self.captured_images = []
        captured = self.captured_images
        class Embedder:
            dimension = 2
            metadata = {'dimension': 2, 'fixture': True}
            def __init__(self, **kwargs): pass
            def encode_text(self, texts):
                return np.array([[1., 0.] if 'Apple' in x else [0., 1.] for x in texts])
            def encode_images(self, images):
                captured.extend(images)
                return np.array([[1., 0.] if image.getpixel((0, 0))[0] == 0 else [0., 1.] for image in images])
        class Describer:
            def __init__(self, config): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def _describe_class(self, label, prompt): return SimpleNamespace(description=label + ' description')
        self.Embedder, self.Describer = Embedder, Describer

    def experiment(self, flags=(False, True, False), name='run'):
        cfg = dict(zip(('embed_images','embed_class_labels','embed_descriptions'), flags))
        cfg.update(output_dir=str(self.root / name), embedding={'batch_size': 2},
                   ollama={'llm':'fake'}, qdrant={'collection_prefix':'fixture'}, k=10,
                   description_count=5)
        return Experiment(cfg, dataset=self.dataset, embedder_factory=self.Embedder,
                          describer_factory=self.Describer, client=self.client)

    def test_reference_records_round_trip_through_qdrant(self):
        exp = self.experiment()
        records = exp.references()
        self.assertTrue(all(isinstance(record, ReferenceRecord) for record in records))
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
        self.assertEqual((neighbor.id, neighbor.label), (records[0].id, records[0].label))

    def test_combinations_and_fresh_builds(self):
        for flags in itertools.product([False, True], repeat=3):
            if not any(flags): continue
            exp = self.experiment(flags, str(flags))
            report = exp.run()
            self.assertEqual(report['state'], 'complete')
            self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
            self.assertEqual((exp.output / 'descriptions.json').exists(), flags[2])
            collection = exp.index()['collection']
            self.assertNotEqual(exp.index()['collection'], collection)
            self.assertEqual(exp.evaluate()['metrics']['evaluated'], 2)

    def assert_images_closed(self, images):
        self.assertTrue(images)
        for image in images:
            with self.assertRaises(ValueError): image.getpixel((0, 0))

    def test_missing_and_corrupt_images_abort(self):
        for corrupt in [False, True]:
            with self.subTest(corrupt=corrupt):
                path = Path(self.rows[1]['fname'])
                if corrupt:
                    path.write_bytes(b'not an image')
                else:
                    path.unlink()
                exp = self.experiment((True, False, False), f'bad-index-{corrupt}')
                with self.assertRaises(OSError): exp.index()
                artifact = json.loads((exp.output/'index.json').read_text())
                self.assertEqual(artifact['state'], 'incomplete')
                self.assertIn('error', artifact)
                self.assertNotIn('exclusions', artifact)
                exp = self.experiment(name=f'bad-eval-{corrupt}')
                exp.index()
                with self.assertRaises(OSError): exp.evaluate()
                report = json.loads(next(exp.output.glob('evaluation_*/report.json')).read_text())
                self.assertEqual(report['state'], 'incomplete')
                self.assertEqual(report['metrics']['selected'], 2)
                self.assertEqual(report['metrics']['evaluated'], 0)
                self.assertFalse(list(exp.output.glob('evaluation_*/exclusions.json')))

    def test_query_failure_and_empty_evaluation(self):
        exp = self.experiment(); exp.index()
        with patch('internvl.retrieval.QdrantIndex.query', side_effect=ConnectionError('offline')):
            with self.assertRaises(ConnectionError): exp.evaluate()
        report = json.loads(next(exp.output.glob('evaluation_*/report.json')).read_text())
        self.assertEqual(report['state'], 'incomplete')
        self.assert_images_closed(self.captured_images)
        self.dataset.splits['validation'] = self.dataset.splits['validation'].select([])
        with self.assertRaisesRegex(ValueError, 'No evaluation'): exp.evaluate()

    def test_empty_generation_and_missing_collection(self):
        exp = self.experiment((False, False, True))
        with patch.object(self.Describer, '_describe_class', return_value=SimpleNamespace(description='')):
            with self.assertRaises(ValueError): exp.describe()
        self.assertFalse((exp.output/'descriptions.json').exists())
        exp = self.experiment(name='valid'); exp.index()
        self.client.delete_collection(exp.index()['collection'])
        with self.assertRaises(Exception): exp.evaluate()

    def test_embedding_failure_aborts_without_retry(self):
        for stage in ['index', 'evaluate']:
            with self.subTest(stage=stage):
                exp = self.experiment((True, False, False), stage)
                if stage == 'evaluate': exp.index()
                with patch.object(exp.embedder, 'encode_images', side_effect=ValueError('synthetic failure')) as encode:
                    with self.assertRaisesRegex(ValueError, 'synthetic failure'):
                        getattr(exp, stage)()
                encode.assert_called_once()
                self.assert_images_closed(encode.call_args.args[0])
                path = exp.output/'index.json' if stage == 'index' else next(exp.output.glob('evaluation_*/report.json'))
                report = json.loads(path.read_text())
                self.assertEqual(report['state'], 'incomplete')
                if stage == 'evaluate':
                    self.assertEqual(report['metrics']['available'], 2)
                    self.assertEqual(report['metrics']['successfully_embedded'], 0)

    def test_empty_retrieval(self):
        exp = self.experiment(); exp.index()
        with patch('internvl.retrieval.QdrantIndex.query', return_value=[]):
            report = exp.evaluate()
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 0)
        self.assertTrue(all(p['prediction'] is None for p in report['predictions']))
        self.assert_images_closed(self.captured_images)

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
        references = exp.references()
        with patch.object(exp, 'references', return_value=references):
            artifact = exp.index()
        stored = self.client.retrieve(artifact['collection'], ids=[r.id for r in references], with_vectors=True)
        for point in stored:
            record = next(r for r in references if r.id == str(point.id))
            self.assertEqual(point.payload, record.to_payload())
            self.assertEqual(point.vector, [1., 0.] if record.label == 'Apple' else [0., 1.])
        report = exp.evaluate()
        self.assertEqual([p['source_id'] for p in report['predictions']], ['0', '1', '0'])
        self.assertEqual(report['metrics']['coverage'], 1)
        self.assertEqual(report['metrics']['majority_vote_accuracy'], 1)
        self.assert_images_closed(self.captured_images)

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
        exp.describer_factory, exp.embedder_factory = Describer, Embedder
        exp.run()
        exp = self.experiment(name='disabled')
        exp.describer_factory = lambda **kw: self.fail('Disabled backend constructed')
        exp.describe()
        self.assertIsNone(exp._embedder)
        exp.run()


if __name__ == '__main__': unittest.main()
