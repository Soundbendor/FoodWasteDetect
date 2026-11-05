# Consolidate validation-batch preparation inside `Experiment`

Status: agreed implementation plan; not yet implemented. Current behavior remains
documented in [experiment contracts](../classification/experiments.md).

## 1. Summary and agreed scope

Make `Experiment` the sole owner of selecting and preparing validation batches for
ordinary evaluation and cached sweeps.

The implementation will:

- Preserve the configuration-driven `evaluate()` operation.
- Add `evaluate_cached()` for evaluation using existing cached vectors.
- Remove `evaluate_from_batches()` without a compatibility shim.
- Keep cache construction and resumption in `IndexBuilder`.
- Preserve prediction, reporting, tracking, and failure ordering.
- Include the approved, targeted repair of the undefined `sources` variable in
  shorter-dimension sweeps.

This change will not redesign encoders, alter image cleanup, fix Ollama client
cleanup, add storage compatibility checks, or change configuration schemas, table
formats, metrics, dependencies, or README files.

## 2. Public interface and behavior

Both public evaluation methods return the existing `EvaluationReport`.

```python
def evaluate(self) -> EvaluationReport: ...

def evaluate_cached(self) -> EvaluationReport: ...
```

| Operation | Validation preparation | Encoding allowed? |
| --- | --- | --- |
| `evaluate()`, caching disabled | Encode validation images in existing bounded batches | Yes |
| `evaluate()`, caching enabled | Complete missing cache entries through `IndexBuilder`, then read ordered vectors | Only for missing entries |
| `evaluate_cached()` | Open the existing validation table and read ordered vectors | Never |

`evaluate_cached()` operates independently of `cache_validation_embeddings`. It
must work when that flag is false and must not change the flag or otherwise mutate
configuration.

When a cache is required, reject a missing, empty, whitespace-only, or
reference-equal validation-table name with `ValueError`, using the existing
configuration-validation message. Ordinary uncached evaluation continues to
ignore unused validation-table settings.

Missing databases, tables, and vectors propagate their existing storage errors.
Cached-only evaluation performs no regeneration, identity scan, completeness
preflight, table creation, append, or optimization. Existing evaluation
completeness checks remain intact.

Removing `evaluate_from_batches()` deliberately withdraws supplied-batch
evaluation from the `Experiment` interface. The lower-level evaluation module
continues accepting supplied batches.

## 3. Implementation sequence

### A. Repair the sweep prerequisite

In `experiments/matryoshka.py`, define the ordered source-table pair from the
original configuration:

```python
sources = (config.lancedb.table, validation_source)
```

Create it after validating the validation-source name and before processing
dimensions. Preserve the pairing of reference source → reference destination and
validation source → validation destination.

Keep this repair distinguishable from the consolidation in review. Do not change
dimension validation, derived-table creation, reuse behavior, or summary
publication.

### B. Consolidate preparation inside `Experiment`

In `experiments/classification.py`, introduce:

```python
def _prepare_validation_batches(
    self, *, cached_only: bool
) -> Iterator[EmbeddingBatch]: ...
```

Implement it as an **ordinary method returning an iterator**, not a generator
function containing a top-level `yield`.

Its behavior is:

1. When `cached_only` is false and caching is disabled, return the existing
   `_evaluation_batches()` iterator.
2. Otherwise, validate the validation-table name.
3. For cached-only operation, open the existing table directly through
   `LanceDBIndex.open()`.
4. For ordinary cached evaluation, call the existing
   `IndexBuilder.build_validation_table()`.
5. Obtain the undecoded validation-metadata iterator using the experiment's
   dataset and batch size.
6. Return a generator expression pairing each metadata batch with
   `fetch_vectors()` requests keyed by `("validation_image", source_id)` in
   metadata order.

Cache opening/completion and metadata-iterator construction must happen
immediately. Vector fetching must remain lazy until evaluation consumes each
batch. Preserve repeated identities, final partial batches, and the existing
`EmbeddingBatch` representation.

Move the ordered-cache-fetch implementation out of `experiments/batches.py`. Keep
that module as the owner of the unchanged `EmbeddingBatch` type alias, and remove
its obsolete storage and iterator imports.

Do not add a new preparation class, public policy type, configuration field, or
dependency-injection interface.

### C. Preserve ordering in both public operations

Wire the public methods as follows:

```text
evaluate()
  → prepare references, matching summary, and reference-count tracking
  → prepare validation batches with cached_only=False
  → execute evaluation and publish the completed report

evaluate_cached()
  → prepare validation batches with cached_only=True
  → prepare references, matching summary, and reference-count tracking
  → execute evaluation and publish the completed report
```

This preserves the existing difference between ordinary evaluation and the sweep:

- Ordinary evaluation discovers reference failures before completing a validation
  cache.
- Cached-only evaluation discovers cache-open or metadata-iterator-construction
  failures before preparing references.
- Missing individual vectors still fail during batch consumption, after reference
  preparation and class-count loading.
- Ordinary cache completion finishes before prediction starts.

Retain `_execute_evaluation()` as the shared prediction/reporting path and
`_evaluation_batches()` as the uncached encoding iterator. Remove
`evaluate_from_batches()`.

### D. Migrate the sweep caller

Retain the existing module-level `matryoshka.evaluate_cached(config, tracker)`
callback and its signature because the CLI supplies it to the tracked-run
mechanism.

Reduce its implementation to:

```python
return Experiment(config, tracker=tracker).evaluate_cached()
```

Remove the callback's knowledge of dataset iteration, batch size, cache opening,
and cached-vector assembly. Remove imports made obsolete by this change.

Preserve the CLI command spelling, one W&B context per dimension, derived-table
preparation, report locations, and sweep-summary behavior.

### E. Update the owning documentation

Update the existing experiment contract document and scoped experiment guide to
describe:

- `Experiment` ownership of validation-batch preparation.
- The two public evaluation methods and removal of supplied-batch evaluation.
- Cached-only behavior independently of the normal cache flag.
- Eager preparation, lazy vector fetching, and preserved failure ordering.
- `batches.py` retaining only the batch type.

Update existing documents rather than adding a new documentation file. Leave
historical verification records unchanged. Do not edit READMEs.

## 4. Regression and acceptance tests

Use existing synthetic dataset/encoder fixtures, mocked external transports, and
temporary real LanceDB databases. Exercise the public methods rather than testing
the new private helper directly.

| Scenario | Required assertions |
| --- | --- |
| Cached-only success | Predictions and source IDs follow validation-metadata order; both database tables remain unchanged; a complete report is produced. |
| Vector/metadata pairing | Store distinct orthogonal vectors in reverse order and evaluate with `k=1`; each identity receives its correct prediction. |
| Cached-only with caching disabled | Evaluation succeeds using existing vectors; the false cache flag and complete configuration remain unchanged. |
| No encoding or database mutation | Cached-only evaluation succeeds with image files unavailable and encoder construction, cache building, identity scans, and database mutations configured to fail if invoked. |
| Ordinary partial-cache evaluation | Only missing images are encoded; stored rows and creation metadata survive; cache completion precedes every prediction query. |
| Complete-cache reuse | A subsequent ordinary cached evaluation succeeds without decoding images or constructing an encoder. |
| Missing cache or vector | Existing errors propagate; no fallback encoding, completed report, or tracker evaluation publication occurs. |
| Invalid validation-table name | Parameterize `None`, empty, whitespace-only, and reference-equal names with caching disabled; cached-only evaluation rejects them before storage access. Ordinary uncached evaluation ignores unused settings. |
| Failure ordering | Distinct reference/cache failures demonstrate reference-first ordinary evaluation and cache-first cached-only evaluation; vector fetching remains deferred. |
| Batch edges and publication | Preserve final partial batches, repeated identity ordering, empty-validation rejection, and successful local report writing before tracking. |
| CLI sweep integration | Full-width and shorter-prefix dimensions complete through the existing tracked callback; earlier reports survive a later failure; failed sweeps produce no summary. |

Place direct evaluation coverage in the existing classification tests, using the
`ExperimentTests` synthetic fixtures. Extend the sweep tests for integration and
retain the existing Vertex cache-completion and tracking regressions.

The current sweep fixture uses query vectors with identical cosine directions, so
its existing assertions cannot establish correct vector pairing. Use the
distinct-vector fixture for that guarantee.

Retain meaningful `IndexBuilder` tests for persistence and resumption. Do not
replace them wholesale or add tests that merely assert private method calls or
method absence.

## 5. Verification and completion criteria

At planning time, the working tree was clean at `1343b8c`. The current
classification Pixi environments were absent; earlier lint, type-check, and test
attempts could not find their executables.

Before implementation verification, prepare the locked CPU environment if still
missing:

```sh
pixi install --manifest-path src/classification/pyproject.toml --locked -e default
```

Use the repository's prescribed Pixi environment. Do not use the legacy
environment as a substitute or update the manifest/lockfile. Installation may
require environment/network permission; model downloads and live external
execution remain outside scope.

Run the focused classification, sweep, Vertex-cache, and tracking tests during
development. After the final edits, run the standard checks once:

```sh
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default lint
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default format-check
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default typecheck
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default test
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default classification --help
```

Completion requires:

- The sweep accesses validation batches only through
  `Experiment.evaluate_cached()`.
- No active code or current contract documentation references
  `evaluate_from_batches()` or the removed standalone `cached_batches()` function.
- Both evaluation operations retain their specified ordering and publication
  behavior.
- The targeted source-table repair restores shorter-dimension sweep execution.
- Relevant checks pass, with any environment limitations or unrelated baseline
  failures explicitly reported.
- No unrelated changes, README edits, data migrations, model runs, or live cloud
  requests are included.
