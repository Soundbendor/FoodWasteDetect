# Simplify IndexBuilder while preserving its storage boundary

Status: historical plan; the helper simplifications and subpackage move are
present in the current source. The paths below record the pre-subpackage layout.
Use the [experiment guide](../../src/internvl/experiments/AGENTS.md) and
[current contracts](../internvl/experiments.md) for further work.

## Summary

Keep `IndexBuilder` in its own module. Apply the python-design-patterns skill’s KISS, separation of concerns, and composition principles by removing unnecessary delegation while retaining the existing division:

- `IndexBuilder` selects inputs, orchestrates encoding, resumes batches, and reports progress.
- `LanceDBIndex` owns database operations.
- `Experiment` owns summaries, tracking, and evaluation.

This is a behavior-preserving refactor.

## Implementation changes

In `src/internvl/index_builder.py`:

- Inline `_save_text_batch()` and `_save_image_batch()` into their respective insertion loops. Assign encoded vectors to a local variable, then call `index.append()`. Advance counts and progress only after append succeeds.
- Inline `_lancedb_adapter()` into `build_reference_index()`.
- Retain `_insert_references()`: it expresses the shared insertion phase for new and resumed reference builds and already supports focused testing.
- Retain separate preparation methods for new and resumed builds, the shared image insertion loop, and text-input preparation. These express meaningful workflow steps.
- Correct `build_reference_index()`’s docstring to describe building/resuming and optimizing the table, then returning the index and counts. Publication remains in `Experiment`.
- Remove imports made unused by deleting helpers.

Preserve the constructor, lazy embedder callback, public build methods, return types, and `LanceDBIndex` API. Do not introduce protocols, factories, configuration wrapper classes, inheritance, or additional services. Retain the existing constructor dependencies for this scoped change.

## Behavior to preserve

- New reference builds validate inputs before table creation or encoder loading.
- Resumed builds apply training caps before excluding saved identities; completed tables require no encoder.
- Stored rows and UUIDs remain unchanged, and initial counts include every stored row.
- Successful batches persist immediately; failures propagate, preserving earlier batches without retries or rollback.
- Reference optimization completes before summary publication, including no-work resumes. Validation-cache behavior remains unchanged.
- Record/vector ordering, strict metadata pairing, image ownership, progress-bar settings, and final partial batches remain intact.

## Verification

Update existing tests that patch removed helpers to observe the fake encoder and `LanceDBIndex.append()` instead. Preserve assertions about image/record pairing and failure ordering; avoid adding tests that merely reproduce the new implementation.

Run the existing CPU-only test task with mocked models/services and temporary LanceDB tables. Verify coverage for:

- Reference-source combinations, capped selection, and missing class coverage.
- Partial and complete resumes, including filtering before decoding.
- Encoding and append failures, committed-batch persistence, and progress advancement.
- Optimization failure preventing summary publication.
- Validation-cache resumption and ordered vector reuse.

Run the configured lint and type checks. Compare any failures with the pre-change baseline and report unrelated failures separately. Do not download dependencies or models or invoke real services for these checks.

## Assumptions and boundaries

Use the current working tree as the baseline and preserve its existing uncommitted changes. Update tests only where helper removal requires it or a listed behavior lacks coverage. No README edits, configuration changes, migrations, artifact-validation additions, or changes to retrieval metrics are included.
