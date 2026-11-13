# Add classification datasets

Evaluate the embedding-retrieval classifier in `src/classification` on six
food datasets: `foodx251`, `food101`, `uecfood256`, `vegfru`, `isia500`, and
`food2k`. Each step below is one reviewable change with its own tests. Caption
datasets are out of scope; see [TO_INVESTIGATE](../TO_INVESTIGATE.md).

Status: proposed; nothing implemented, no dataset downloaded.

## Datasets

| Name | Classes / images | Train split | Evaluation split | Host |
| --- | --- | --- | --- | --- |
| `foodx251` | 251 / 158k | `train` | `val` (test is unlabeled) | `food-x.s3.amazonaws.com/{annot,train,val}.tar` |
| `food101` | 101 / 101k | `train` 75,750 | `test` 25,250 | `data.vision.ee.ethz.ch/cvl/food-101.tar.gz` (5.0 GB) |
| `uecfood256` | 256 / ~31k | defined by us | defined by us | HF `kng-research/uecfood256`, `dataset256.zip` |
| `vegfru` | 292 / ~160k | official list | official list | HF `xedwin23x/VegFru`, `vegfru.tar.gz` |
| `isia500` | 500 / 399,726 | `train_finetune.txt` 239k | `test_finetune.txt` 120k | `123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/` (~43.6 GB) |
| `food2k` | 2,000 / ~1M | official list | official list | none; access by agreement |

Hosts for `foodx251`, `food101`, and `isia500` answered on 2026-11-10. The
`uecfood256` and `vegfru` archive layouts are unverified until their step.

## Step 1: `FoodImageDataset` base class

Move everything dataset-independent out of `FoodX251ImageDataset`. A subclass
supplies a class list and, per split, a table of relative image paths and
labels. The Hugging Face schema, `select_images()`, and
`iter_dataset_batches()` do not change.

```python
class FoodImageDataset(ABC):
    name: ClassVar[str]

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.classes = self._read_classes()
        self.splits = DatasetDict(
            {s: self._build_split(*self._read_split(s)) for s in ("train", "validation")}
        )

    @abstractmethod
    def _read_classes(self) -> list[str]:
        """Return unique canonical labels in taxonomy order."""

    @abstractmethod
    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        """Return (source_split, image_dir, table[fname, class])."""

    @classmethod
    def download(cls, root: str | Path) -> None:
        raise NotImplementedError(f"{cls.name} has no public download host")
```

`_build_split()` is the current `_load_split()` body after the CSV read. It
resolves paths under `image_dir`, builds `f"{self.name}:{source_split}:{relative}"`
identities, rejects duplicates, and returns the typed `Dataset`. FoodX-251
identities stay `foodx251:<train|val>:...`, so existing LanceDB tables remain
resumable.

`validation_set()` keeps its name and means the labeled evaluation split.
`source_split` records the dataset's own split name, such as `test` for
Food-101.

Registry and configuration:

```python
# experiments/config.py
DatasetName = Literal["foodx251"]          # one name added per later step

class DatasetConfig(ConfigModel):
    name: DatasetName
    root: NonemptyString

# datasets/image_dataset.py
DATASETS: dict[str, type[FoodImageDataset]] = {
    c.name: c for c in (FoodX251ImageDataset,)
}

def load_dataset(config: DatasetConfig) -> FoodImageDataset:
    return DATASETS[config.name](config.root)
```

The `Literal` lives in `config.py` because `image_dataset.py` imports from it.
Names become lowercase only; the current validator accepts any case.

Rewire `experiments/classification.py`, `index_builder.py`, and
`matroyshka.py` to annotate `FoodImageDataset`.

Checks: existing `tests/test_classification.py` and `tests/test_config.py`
pass unchanged apart from the type name; add a test that `DatasetName` values
equal `DATASETS` keys, and one that the base `download()` raises. Update
`docs/classification/data.md` and `src/classification/AGENTS.md`.

## Step 2: `FoodX251ImageDataset.download()`

Add two module helpers that later steps reuse. Neither adds a dependency;
`httpx` and `huggingface_hub` are already in the environment.

```python
def _fetch_archive(url: str, root: Path) -> Path:
    """Stream to root/.downloads/<name>.part, rename on completion, return the path."""

def _extract(archive: Path, destination: Path) -> None:
    """tarfile.extractall(filter="data") or zipfile.extractall by suffix."""
```

```python
class FoodX251ImageDataset(FoodImageDataset):
    @classmethod
    def download(cls, root: str | Path) -> None:
        root = Path(root).expanduser().resolve()
        if (root / "annot/class_list.txt").exists():
            return
        for part in ("annot", "train", "val"):
            _extract(_fetch_archive(f"https://food-x.s3.amazonaws.com/{part}.tar", root), root / part)
```

Confirm on the first authorized run that the tars unpack to the existing
`annot/`, `train/train_set/`, `val/val_set/` layout, and adjust destinations
if not.

CLI, next to `download_model`:

```sh
classification download_dataset --config_file cfg/classification.yml
```

It validates only the `dataset` section, calls
`DATASETS[name].download(root)`, and opens no W&B run.

Checks: build a small tar in `tmp_path`, serve it through
`httpx.MockTransport`, and assert the extracted layout loads; assert a second
call makes no request; CLI dispatch test with a fake dataset class. Update
`docs/classification/cli.md` and `data.md`.

## Steps 3 to 7: one dataset per change

Each step adds one subclass, its `DatasetName` entry, a `download()`, a
miniature synthetic tree test, a `cfg/classification_<name>.yml`, and a
paragraph in `data.md`. Each config sets its own `output_dir`,
`lancedb.table`, and `descriptions_file`, because
`assets/classification/descriptions_v1` only covers FoodX-251 labels.

### Step 3: `food101`

```python
class Food101ImageDataset(FoodImageDataset):
    name = "food101"

    def _read_classes(self) -> list[str]:
        return [_canonical(n) for n in (self.root / "meta/classes.txt").read_text().split()]

    def _read_split(self, split: str) -> tuple[str, Path, pd.DataFrame]:
        source = {"train": "train", "validation": "test"}[split]
        stems = (self.root / f"meta/{source}.txt").read_text().split()
        table = pd.DataFrame({
            "fname": [f"{s}.jpg" for s in stems],
            "class": [_canonical(s.split("/")[0]) for s in stems],
        })
        return source, self.root / "images", table
```

`download()` fetches the ETHZ tarball and moves the inner `food-101/`
directory's contents up into `root`.

### Step 4: `uecfood256`

The archive has one directory per class ID, a `category.txt` ID/name table,
and per-class `bb_info.txt` boxes. It has no official train/test split, so the
loader derives one: sort files per class, shuffle with a fixed seed, and hold
out 20% for evaluation. The seed and ratio are class constants and the split
is not written to disk. Results are not comparable with published 5-fold
numbers.

An image listed under several classes contains several foods. Drop those
images from both splits so each record has one label.

`download()` uses `hf_hub_download("kng-research/uecfood256", "dataset256.zip", repo_type="dataset")`
then `_extract()`. Verify the archive layout before writing the parser.

### Step 5: `vegfru`

Read the official train and test image lists (`<relative path> <label id>`)
and take class names from the directory component. Validation and test list
choice: use the official test list for evaluation and the train list for
references; leave the val list unread. Verify list filenames and layout from
the archive first.

`download()` uses `hf_hub_download("xedwin23x/VegFru", "vegfru.tar.gz", repo_type="dataset")`.

### Step 6: `isia500`

Layout confirmed from `metadata_ISIAFood_500.zip`:
`ISIAFood500_classLabel.txt` holds `name<TAB>id` for 500 classes, and
`train_finetune.txt` / `test_finetune.txt` hold `Class/Class_0001.jpg id`
lines. Class names use underscores and include non-ASCII characters and
parentheses; canonicalize by replacing underscores only.

Images ship as a split zip: `ISIA_Food500.z01` to `.z10` at 4 GiB each plus
`ISIA_Food500.zip`. Python's `zipfile` cannot read split archives, so
`download()` fetches all eleven parts and the metadata zip, then runs `7z x`
on `ISIA_Food500.zip`, failing with an install hint when `7z` is not on PATH.
The server is a plain HTTP IP address with no checksums; compare each part's
size with its `Content-Length`.

### Step 7: `food2k`

Loader only; `download()` inherits the stub, with a message pointing to the
access agreement. The parser is blocked on the on-disk layout (class-name
file, split list filenames and line format, image directory structure).

## Verification and limits

All tests use synthetic trees and mocked transports. Real downloads total
roughly 60 GB before `food2k` and need explicit authorization per dataset.
README changes need separate approval.

## Open decisions

1. `uecfood256`: accept the seeded 80/20 split, or pick another policy.
2. `food2k`: provide the file layout.
3. `isia500`: accept the `7z` system requirement, or add `p7zip` to the Pixi
   manifest.
