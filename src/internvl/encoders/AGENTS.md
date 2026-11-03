# Encoder guide

Encoders translate ordered text or image batches into vectors. Keep dataset
selection, persistence, retrieval, and tracking in their owning modules.

| Module | Responsibility |
| --- | --- |
| `base.py` | Backend-independent `Embedder` interface and `ImageInput` alias. |
| `sentence_transformers.py` | Local model specifications, loading, RGB preparation, library encoding, and vector validation. |
| `vertex.py` | Vertex request construction, concurrency, SDK retries, response validation, and client cleanup. |

Import concrete modules directly; `__init__.py` stays free of backend imports.
Read the applicable contract before changing a backend:

| Task | Required reference |
| --- | --- |
| Interface, output validation, batching, or image ownership | [Shared contract](../../../docs/internvl/encoders.md#shared-interface) and the affected backend section. |
| Local model, prompt, dtype, device, or dimension | [Local models](../../../docs/internvl/encoders.md#local-models) and [environment selection](../../../docs/internvl/encoders.md#environment-and-hardware-selection). |
| Vertex request, authentication, concurrency, retry, or prompt | [Vertex](../../../docs/internvl/encoders.md#vertex). |
| Reference/cache persistence or restart behavior | [Experiment contracts](../../../docs/internvl/experiments.md). |
| Backend verification or real-model acceptance | [Verification](../../../docs/internvl/encoders.md#verification) and the dated [experiment notes](../../../docs/experiments.md). |

For an encoder change, check every affected modality, selected dimension, empty
input, failure path, and caller-owned image against the contract. Use synthetic
CPU models or mocked transports; real models, services, and hardware require the
authorization described in the root guide.
