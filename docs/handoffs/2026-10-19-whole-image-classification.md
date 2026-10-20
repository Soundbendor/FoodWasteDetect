# Whole image classification handoff

Conversation date: October 19, 2026.

This is a historical handoff. The subsequent orchestration refactor removed
`Experiment.references()`: text references now use a private helper, and image
references are constructed directly from decoded batches. Dataset loading is
lazy, description generation uses `DescriberLLM.describe_class(label)`, and stage
failure reporting includes index setup errors. See README.md and AGENTS.md for
the current behavior; the snapshot below also predates removal of exclusions.

The configurable whole-image classification runner is implemented in the working
tree. The user then explicitly requested simplification: remove revision pins,
artifact validation, provenance bookkeeping, and the retired BGE fine-tuning
experiment. Those later requests supersede the original implementation plan.
Do not restore that machinery without a new request.

The latest code change replaces reference dictionaries with a shared
`ReferenceRecord` dataclass. The user's latest question was about
`source_identity`; it was explained, and no removal was requested.

## Current experiment

Run from the repository root:

```sh
PYTHONPATH=src python -m internvl.experiment run --config_file cfg/classification.yml
```

The stages are `describe`, `index`, `evaluate`, and `run`. Three YAML booleans
select reference sources: `embed_images`, `embed_class_labels`, and
`embed_descriptions`. All seven nonempty combinations are supported; all-false
is rejected. Defaults are descriptions only, five descriptions per class,
batch size 32, explicit CPU device, and k=10.

The runner embeds whole-image pixels with pretrained Jina CLIP v2. It does not
caption evaluation images, run YOLO, or produce segmentation masks. The composed
FoodX251 adapter uses the complete official training table for image references
and the complete labeled validation table for evaluation, marked
`source_split="val"`. It does not use the unlabeled official test split or
subsample evaluation images. Dataset path/content overlap checks remain in the
dataset adapter; the artifact-validation removal did not remove these checks.

Retrieval uses exact cosine search. Each neighbor contributes one vote. Ties
resolve by the highest individual similarity, then class name. `vote_fraction`
is winning votes divided by actual neighbor count, not calibrated confidence.
Empty results abstain and count as incorrect. Accuracy uses successfully embedded
evaluation images; reports retain coverage, exclusions, sample counts, per-class
support/correct/recall, and nearest-neighbor accuracy. There is no top-five metric.
Missing or failed image embeddings retain their identities in exclusions.
Infrastructure failures abort evaluation and save incomplete reports.

## Decisions that supersede the original plan

- Removed YAML `revision` and `code_revision`, hard-coded nested model/code pins,
  revision metadata, and the related tests. The embedder uses default cached
  versions. Local-cache-only loading, explicit device selection, RGB image
  conversion, normalized float32 vectors, and full 1,024 dimensions remain.
- Removed `digest()`, artifact compatibility checks, expected/indexed manifest
  comparisons, collection-content verification, and `QdrantIndex.validate()`.
  Artifact validation is deferred, not a pending task to implement now.
- Removed `provenance()` and its report fields, Git subprocess calls, dependency
  version collection, timestamps, and unused imports.
- Removed `finetune_embeddings()` from `workflows.py` and the
  `finetune-embeddings` CLI command. The old BGE training/triplet implementation
  was already removed when the inference-only embedder replaced it. Preserve
  the separate `src/clip_finetune.py` experiment.
- Kept basic configuration checks, empty-description checks, per-class training
  image coverage checks, exclusion logging, and failure reporting.

The user repeatedly objected to excessive infrastructure. Keep future changes
small and directly tied to the requested behavior. Do not introduce replacement
validation frameworks or reproducibility tracking under new names.

## Current artifact behavior

`describe` loads an existing `descriptions.json` directly. Otherwise it generates
and saves a dictionary under the `descriptions` key. It does not validate an
existing file against the current configuration. Remove the file to regenerate.
When descriptions are disabled, the stage does nothing and does not load Ollama.
Owned Ollama cleanup completes before `run` constructs the embedder.

Every `index` invocation, including through `run`, creates a fresh uniquely named
Qdrant collection. It replaces `index.json` with the latest collection name,
model settings, source counts, exclusions, configuration, and status. Historical
collections remain in Qdrant. There is no index reuse or embedding resume logic.

`evaluate` directly reads the collection name and summary from `index.json`.
It does not check compatibility or gate loading on the saved status field.
Status fields report outcomes only. Each evaluation recomputes query embeddings
and writes a separate `evaluation_<UUID>` directory with report, predictions,
metrics, and exclusions JSON files.

## ReferenceRecord

`src/internvl/records.py` defines a frozen dataclass:

```python
@dataclass(frozen=True)
class ReferenceRecord:
    label: str
    source_kind: str
    source_identity: str
    value: str
    id: str = field(default_factory=lambda: str(uuid4()))
```

This is the application's record schema, not a universal Qdrant schema.
`to_payload()` serializes `label` as `"class"` to preserve the existing stored
payload key; `class` cannot be a Python attribute declaration because it is a
reserved word. The other fields retain their names.

`Experiment.references()` constructs these objects for enabled sources.
`QdrantIndex.upsert()` accepts a sequence of them, uses `record.id` as the point
ID, and serializes each record as the payload alongside its embedding. Query
results remain lightweight `Neighbor` dataclasses. Shared image batching retains
original reference objects while using serialized payloads for exclusion reports;
evaluation dataset rows remain dictionaries.

| Field | Meaning |
| --- | --- |
| `label` / stored `class` | Category used for voting, such as `Apple`. |
| `source_kind` | `class_name`, `generated_description`, or `training_image`. |
| `source_identity` | Identifies the source reference: `Apple`, `Apple:2`, or `foodx251:train:123.jpg`, for example. |
| `value` | Text to embed or a local image path. |
| `id` | Random UUID assigned to this record for Qdrant insertion. |

Many references can share a class. `source_identity` distinguishes their origins
and is currently only stored and used in exclusion logs; it does not affect
embeddings, retrieval, or voting. The user asked for this distinction and has
not requested changing the field.

## Files and migration boundaries

| File | Responsibility |
| --- | --- |
| `src/internvl/experiment.py` | Stage orchestration, reference construction, image batching, reports, CLI. |
| `src/internvl/records.py` | Shared reference dataclass and payload conversion. |
| `src/internvl/retrieval.py` | Separate Qdrant adapter, neighbors, voting, and metrics. |
| `src/internvl/image_dataset.py` | Independent dataset interface and composed FoodX251 adapter. |
| `src/internvl/embeddings.py` | Inference-only multimodal embedding wrapper. |
| `cfg/classification.yml` | Dedicated whole-image experiment configuration. |
| `tests/test_classification.py` | Synthetic contracts and in-memory Qdrant tests. |
| `src/internvl/cli.py` | Preserved caption utilities and migration errors for retired retrieval commands. |
| `src/internvl/workflows.py` | Legacy workflows; the BGE fine-tuning function was deleted. |
| `README.md`, `AGENTS.md`, `docs/experiments.md` | Updated usage, scope, and retirement guidance. |

The legacy `internvl build-db` and `internvl evaluate` commands still exit with
migration guidance. Their old workflow implementations remain for historical
inspection and are incompatible with the new embedder. They were not repaired.
Caption generation, description generation, caption merge/conversion, and model
download utilities remain. Legacy image-facing `VectorDB`, `EvalMetric`, the
FoodX251 wrapper's filename behavior, and separate CLIP fine-tuning remain intact.

## Verification and workspace state

Latest code verification ran 41 tests: 40 passed and one pre-existing test was
skipped. The skip is
`test_remote_endpoint_failure_does_not_launch_local_server`, documented in the
test as an existing remote-endpoint guard mismatch outside its scope. No new
dependency skips occurred in the temporary test environment.

```sh
PYTHONPATH=src /tmp/food-classification-tests/bin/python -m unittest discover -s tests -v
```

The temporary environment was created with system site packages and had
`qdrant-client` and `PyYAML` installed with approval. It may not survive a future
session. The ordinary repository command is:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover source combinations, dataset split handling and leakage rejection,
embedding shapes/order/normalization and image cleanup, exclusions, voting,
coverage, service-error reporting, import/help isolation, and a dataclass payload
round trip through in-memory Qdrant. `git diff --check` passed after the latest
code change. No real dataset, model, GPU, Ollama, remote-Qdrant, or HPC validation
was performed. No large weights or datasets were downloaded, and no live server
was started. Fixture success does not establish real-model compatibility.

The working tree contains the implementation and follow-up edits, including
untracked new modules/configuration/tests. No commits or pull requests were made
in this conversation. Inspect `git status` before further work and preserve these
changes. There is no outstanding implementation request beyond saving this
handoff; await the user's next direction.
