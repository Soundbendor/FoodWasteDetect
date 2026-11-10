# Encoder guide

Encoders translate ordered text or image batches into vectors. Keep dataset
selection, persistence, retrieval, and tracking in their owning modules.

| Module | Responsibility |
| --- | --- |
| `base.py` | Backend-independent `Embedder` interface, `ImageInput` alias, and RGB preparation. |
| `sentence_transformers.py` | Local model specifications, loading, library encoding, and vector validation. |
| `vertex.py` | Vertex request construction, concurrency, SDK retries, response validation, and the reused client closed by `close()`. |

Import concrete modules directly; `__init__.py` stays free of backend imports.
Read the applicable contract before changing a backend:

| Task | Required reference |
| --- | --- |
| Interface, output validation, batching, or image ownership | [Shared contract](../../../docs/classification/encoders.md#shared-interface) and the affected backend section. |
| Local model, prompt, dtype, device, or dimension | [Local models](../../../docs/classification/encoders.md#local-models) and [environment selection](../../../docs/classification/encoders.md#environment-and-hardware-selection). |
| Vertex request, authentication, concurrency, retry, or prompt | [Vertex](../../../docs/classification/encoders.md#vertex). |
| Reference/cache persistence or restart behavior | [Experiment contracts](../../../docs/classification/experiments.md). |
| Backend verification or real-model acceptance | [Verification](../../../docs/classification/encoders.md#verification) and the dated [experiment notes](../../../docs/experiments.md). |

For an encoder change, check every affected modality, selected dimension, empty
input, failure path, and caller-owned image against the contract. Use synthetic
CPU models or mocked transports; real models, services, and hardware require the
authorization described in the root guide.
