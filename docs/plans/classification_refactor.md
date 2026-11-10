# Classification refactor plan

Follow-up to the `src/classification` simplification pass. Nine steps, one
commit each, ordered from deletions to changes that need verification first.
After each step run the `test`, `lint`, and `typecheck` Pixi tasks in the
`default` environment and update the docs named in the step.

Decisions already made: steps 1 and 2 remove documented, tested contracts on
purpose. Step 8 is verified with mocked transports only; no live Vertex run.

## 1. Drop validation that repeats the config

Constructors keep only the checks `ExperimentConfig` cannot express. Direct
construction with bad types then fails later and less clearly.

```python
# encoders/vertex.py, inline in __init__, replacing _validate_config
if not project:
    raise ValueError("vertex.project must be a nonempty string")
if truncate_dim is not None and not 128 <= truncate_dim <= 3072:
    raise ValueError("truncate_dim must be None or an integer between 128 and 3072")

# encoders/sentence_transformers.py
if truncate_dim is not None and not 1 <= truncate_dim <= native_dimension:

# experiments/evaluation.py: delete the k check; use the batch as given
return [
    cast(ImageMetadata, dict(zip(batch, values, strict=True)))
    for values in zip(*batch.values(), strict=True)
]
```

Also delete the blank `binary_path` check in `llm/ollama.py`.

- Tests: prune the matching cases in `test_vertex_embeddings.py` (direct
  construction parametrization), `test_embeddings.py` (`truncate_dim` types),
  `test_evaluation.py` (`k` and image-column cases), and the blank and `None`
  path subtests in `test_descriptor_lifecycle.py`.
- Docs: Vertex settings paragraph in `docs/classification/encoders.md`.

## 2. Remove dataset aliases

Delete `test_set`, `iter_batches`, and the `train_path`, `val_path`, and
`test_path` attributes (use locals in `__init__`). `SPLIT_NAMES` stays for
`_load_split`.

```python
# experiments/classification.py, both call sites
self.dataset.iter_dataset_batches(self.dataset.validation_set(), batch_size, decode=False)
```

- Tests: the alias and `iter_batches` cases in `test_classification.py` switch
  to `select_images` plus `iter_dataset_batches`.
- Docs: split and alias text in `docs/classification/data.md`.

## 3. Move the embedder factory into `encoders/`

```python
# encoders/factory.py
def create_embedder(options: EmbeddingConfig) -> Embedder:
    if options.model_id == "google/ge-v2":
        return VertexEmbeddingModel(
            **options.vertex.model_dump(), truncate_dim=options.truncate_dim
        )
    return EmbeddingModel(
        options.model_id,
        options.batch_size,
        device=options.device,
        truncate_dim=options.truncate_dim,
    )
```

`Experiment.embedder` calls it and `_create_embedder` goes away.

- Tests: `test_matroyshka_sweep.py` patches
  `classification.experiments.classification.create_embedder` instead of
  `Experiment._create_embedder`.
- Docs: backend selection in `docs/classification/encoders.md` and the module
  table in `src/classification/encoders/AGENTS.md`.

## 4. Stop `LanceDBIndex.create` from mutating the config

`create` loses `config=`. The builder owns the mutation, and `retrieval` no
longer imports `experiments.config`. Persisted metadata is unchanged.

```python
# experiments/index_builder.py, one helper for both call sites
def _create_table(self, name: str) -> LanceDBIndex:
    self.config.vector_dimension = self.embedder.dimension
    return LanceDBIndex.create(
        self.uri,
        name,
        self.embedder.dimension,
        metadata={"classification.build": self.config.model_dump_json()},
    )

# retrieval/lancedb.py, create_prefix_table
if encoded is not None:
    snapshot = json.loads(encoded) | {"vector_dimension": dimension}
```

- Tests: the `config=` case in `test_retrieval.py`.
- Docs: build-metadata section in `docs/classification/retrieval.md`.

## 5. Load the dataset once per sweep

```python
# experiments/classification.py
def __init__(self, config, *, tracker=None, dataset: FoodX251ImageDataset | None = None):
    self.dataset = dataset or load_dataset(config.dataset)

# cli.py
dataset = load_dataset(config.dataset)
run_sweep(
    config,
    lambda effective: _tracked_run(
        effective,
        stage,
        lambda c, t: Experiment(c, tracker=t, dataset=dataset).evaluate_cached(),
    ),
)
```

This removes `matroyshka.evaluate_cached`.

- Tests: `test_matroyshka_sweep.py` patches `cli.Experiment` instead of
  `cli.evaluate_cached`.

## 6. Arrow-based `append`

```python
vectors = np.ascontiguousarray(vectors, dtype=np.float32)
if vectors.shape != (len(records), self.dimension):
    raise ValueError("Records and vectors must pair with the table dimension")
payloads = [r.to_payload() for r in records]
columns = {name: [p[name] for p in payloads] for name in payloads[0]}
columns["vector"] = pa.FixedSizeListArray.from_arrays(
    pa.array(vectors.reshape(-1)), self.dimension
)
self.table.add(pa.table(columns, schema=self.table.schema), on_bad_vectors="error")
```

Check first: run the malformed-vector cases in `test_retrieval.py` against this
and confirm `on_bad_vectors="error"` still rejects NaN for an Arrow table. If it
does not, add an `np.isfinite` check. Empty batches need an early return.

## 7. Batched validation queries

Verify on a synthetic table before writing code: `limit` applies per query
vector, and rows carry a `query_index` column.

```python
# retrieval/lancedb.py
def query_batch(self, vectors, limits: Sequence[int]) -> list[list[Neighbor]]:
    rows = (
        self.table.search(vectors.tolist())
        .distance_type("cosine")
        .bypass_vector_index()
        .select(["id", "class"])
        .limit(max(limits))
        .to_list()
    )
    rows.sort(key=lambda r: (r["query_index"], r["_distance"]))
    out: list[list[Neighbor]] = [[] for _ in limits]
    for r in rows:
        neighbors = out[r["query_index"]]
        if len(neighbors) < limits[r["query_index"]]:
            neighbors.append(Neighbor(id=r["id"], label=r["class"], score=1 - r["_distance"]))
    return out

# experiments/evaluation.py
counts = [class_counts.get(r["class"], 0) for r in rows]
ranked = query(vectors, [max(k, 5, c) for c in counts])
```

`evaluate_batches` takes the batch callable. Delete `query` once nothing calls
it.

- Acceptance: predictions on the synthetic fixture match the single-query path
  exactly. Order among equal distances can differ, so include a tie case.
- Tests: fake `query` callables in `test_evaluation.py` change shape; add
  `query_batch` cases to `test_retrieval.py`.
- Docs: exact retrieval section in `docs/classification/retrieval.md`,
  including tie order.

## 8. Reuse the Vertex client

```python
# encoders/vertex.py
@functools.cached_property
def _client(self) -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=self.project,
        location=self.location,
        http_options=self.http_options,
    )

def close(self) -> None:
    if "_client" in self.__dict__:
        self.__dict__.pop("_client").close()

# encoders/base.py: Embedder.close() is a no-op default
# experiments/classification.py: Experiment.close() closes a constructed embedder
# cli.py
with contextlib.closing(Experiment(effective, tracker=tracker)) as experiment:
```

The executor stays per call.

- Tests: client lifecycle cases in `test_vertex_embeddings.py`, mocked
  transport only. Token refresh over a long build stays unverified.
- Docs: replace the "each encode call owns its client" contract in
  `docs/classification/encoders.md`.

## 9. Minor

```python
# cli.py: method names, so tests that replace cli.Experiment keep working
STAGES = {
    "describe": "generate_descriptions",
    "index": "build_vector_database",
    "evaluate": "evaluate",
    "run": "run",
}
choices=(*STAGES, "matroyshka-sweep")
getattr(experiment, STAGES[stage])()

# encoders/sentence_transformers.py: flags in _MODEL_SPECS replace the name checks
if spec.get("native_bf16"): ...
if spec.get("patch_modalities"): ...

# tracking.py: a Mapping annotation satisfies ty
per_class: Mapping[str, Mapping[str, float]] = metrics["per_class"]
columns = ["support", "correct", "recall", "precision", "f1"]
data = [[label, *(values[c] for c in columns)] for label, values in per_class.items()]

# scripts/smoke_jina.py: _worker keeps only the print; _encode already validates
```

Read `docs/classification/scripts.md` before the smoke change; keep the checks
if that document lists them as the script's acceptance criteria.
