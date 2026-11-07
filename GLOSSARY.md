# Food Waste Detection

Shared vocabulary for food-recognition experiments. The whole-image classification
pipeline compares validation images with labeled references stored as embeddings.

## Language

### Embedding tables

**Embedding table**:
A stored collection of records, each paired with one embedding vector of a fixed
dimension, with a role and a creation snapshot.
_Avoid_: index, vector database, LanceDB index

**Reference table**:
An embedding table of labeled references (class names, generated descriptions, or
training images) that validation images are searched against.
_Avoid_: index, reference index, training table

**Validation cache**:
An embedding table holding one embedding per validation image, reused so that
evaluation does not embed the validation split again.
_Avoid_: validation index, cache table

**Prefix table**:
An embedding table derived from another by keeping every record and shortening
each vector to its leading values, used to evaluate smaller dimensions. It is
derived only from a complete table and is complete once its row count matches
its source's.
_Avoid_: truncated index, derived index, sweep table

**Creation snapshot**:
The experiment configuration recorded when an embedding table was created, kept
exactly as written for provenance; a prefix table carries its source's snapshot.
_Avoid_: build metadata, table config

**Encoder fingerprint**:
The encoder settings recorded in an embedding table's header when it is created
(model ID, truncation dimension, and the Vertex classification-prompt flag).
Vectors are comparable only between tables with equal encoder fingerprints and
dimensions.
_Avoid_: model metadata, encoder config

**Inputs fingerprint**:
The reference-input settings recorded in a reference table's header when it is
created: dataset name, reference-source switches, per-class image cap, and the
vocabulary digest. Validation caches have none.
_Avoid_: source config, reference metadata

**Incompatible table**:
An embedding table whose fingerprints or dimension differ from what a resume or
evaluation requires. It is refused with each differing setting listed; the fix
is a new table name, never an override.
_Avoid_: mismatched index, stale table

**Complete table**:
An embedding table whose build has finished, so it holds every source key it is
expected to hold. It carries a completion marker at its current version; any
later write makes it incomplete again.
_Avoid_: finished index, optimized table

**Role guard**:
The append rule that keeps validation vectors out of reference searches: a
reference table refuses validation-image records, and a validation cache accepts
only validation-image records. A refused batch writes nothing.
_Avoid_: kind filter, leak check

### Records

**Source key**:
The pair of a source kind and a source identity that names what a stored record
was embedded from.
_Avoid_: record ID, identity, source identity (for the pair)

**Source kind**:
The type of input a record was embedded from: class name, generated description,
training image, or validation image.
_Avoid_: reference type, source type

**Source identity**:
The identifier of one input within its source kind, such as a class name, a
description position, or a dataset image ID.
_Avoid_: source ID (for the stored value), key

**Duplicate source key**:
A source key stored in more than one row of an embedding table. Duplicates are
tolerated and reported; the duplicate count is the stored row count minus the
number of distinct source keys, and every duplicate row still counts in source
and class counts.
_Avoid_: duplicate record, repeated ID
