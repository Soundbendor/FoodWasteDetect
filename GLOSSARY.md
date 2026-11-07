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
each vector to its leading values, used to evaluate smaller dimensions.
_Avoid_: truncated index, derived index, sweep table

**Creation snapshot**:
The experiment configuration recorded when an embedding table was created, kept
exactly as written for provenance; a prefix table carries its source's snapshot.
_Avoid_: build metadata, table config

**Complete table**:
An embedding table whose build has finished, so it holds every source key it is
expected to hold.
_Avoid_: finished index, optimized table

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
