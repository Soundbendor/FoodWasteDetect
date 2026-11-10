# Classification encoders

These backends serve the whole-image classification baseline. Source modules live
in [`src/classification/encoders`](../../src/classification/encoders); dataset selection,
resumption, and persistence belong to [experiments](experiments.md).

## Shared interface

`base.Embedder` declares `dimension: int` and abstract `encode_text()` /
`encode_images()` methods, with no shared constructor. `ImageInput` accepts a
string path, `Path`, or PIL image and is re-exported by
`encoders.sentence_transformers`. Importing the interface loads neither concrete
backends nor experiment modules. Package initializers remain lightweight.

Both implementations return NumPy float32 matrices in input order, with one finite,
nonzero vector of the selected dimension per input. Empty calls return shape
`(0, dimension)`. The application adds no normalization. Caller-owned PIL images
stay open. Errors propagate; callers receive no partial result matrix.

`Experiment.embedder` constructs its backend on first use and reuses it. The
`google/ge-v2` alias selects `VertexEmbeddingModel`; other supported IDs select
`EmbeddingModel`. Backend modules are imported normally, while construction stays
behind explicit calls. Local models ignore `embedding.vertex`.
Complete caches can avoid backend construction entirely. Batch persistence and
restart behavior are defined in [reference resumption](experiments.md#reference-builds-and-resumption)
and [validation caching](experiments.md#validation-caching).

## Local models

`sentence_transformers.EmbeddingModel` is inference-only. `_MODEL_SPECS` holds
model-specific loading and encoding options; Sentence Transformers owns inference
batching and NumPy conversion. `encode_images()` prepares the supplied batch
without subdividing it; experiment-level dataset batches bound decoded image
memory. Model settings are recorded through experiment configuration, with no
encoder metadata dictionary or separate artifact model field.

| Model ID | Native dimension | Text protocol | Image protocol |
| --- | ---: | --- | --- |
| `jinaai/jina-clip-v2` | 1024 | Checkpoint defaults | RGB PIL inputs |
| `jinaai/jina-embeddings-v4` | 2048 | `task="retrieval"`, `prompt_name="passage"` | Retrieval task, RGB PIL inputs, no text prompt |
| `google/embeddinggemma-2` | 768 | `prompt_name="Document"` | Separate `{"image": rgb_PIL}` records, no text prompt |
| `openai/clip-vit-base-patch32` | 512 | Plain text, explicit 77-token limit | Checkpoint preprocessing; projected CLIP features |
| `Qwen/Qwen3-VL-Embedding-8B` | 4096 | Checkpoint default instruction | RGB PIL inputs |

Loading and preprocessing contracts:

- CLIP v2 loads float32 with text flash attention and vision xformers disabled.
- Jina v4 uses checkpoint dtype (`torch_dtype="auto"`) and SDPA, without CLIP
  configuration flags. Retrieval is its fixed task for both modalities; preserve
  upstream normalization without adding application normalization.
- EmbeddingGemma2 disables audio with `config_kwargs={"audio_config": None}`. It
  uses BF16 only when the selected CUDA device has native support, excluding
  emulation; otherwise it loads FP32. It never selects FP16.
- OpenAI CLIP loads FP32 and returns projected vectors, rather than vision hidden
  states. Its processor receives `model_max_length=77`.
- Qwen uses checkpoint dtype and SDPA. Its checkpoint configuration owns
  preprocessing, default instruction, last-token pooling, and normalization; the
  application adds no overrides for these or for remote code.
- Only Jina enables remote code. Its custom module receives text/image modality
  metadata only when absent. String text inputs retain the library's default
  text/path/URL interpretation; `assume_text_inputs` is not overridden.
- `device="auto"` resolves through Sentence Transformers' device helper. Loading
  uses normal library caches and downloads, without temporary configuration copies
  or enforced offline mode. Development checks mock pretrained construction.
- Path-opened originals close after RGB conversion, including conversion failure.
  RGB copies are released through garbage collection; caller-owned images stay
  open. Image loading and encoding failures propagate.

### Dimensions and comparison isolation

`EmbeddingModel(..., *, truncate_dim=None)` preserves the native dimension.
Non-null values must be from 1 through the native dimension; out-of-range values
fail before backend loading. `ExperimentConfig` owns type validation.
The selected value goes to the shared Sentence Transformers `encode()` call for
every modality and reference source. The library performs truncation; output
validation, empty matrices, and the database schema use the selected dimension.
Any valid dimension is accepted for all five models; quality must be measured.

Output validation rejects rows whose components are all zero after float32
conversion. It checks components directly so tiny finite nonzero vectors are not
rejected because a float32 norm underflows. It does not rescale the vectors.

For example, Jina v4 at 1024 dimensions uses
`--set embedding.model_id=jinaai/jina-embeddings-v4 --set embedding.truncate_dim=1024`.
Schema-defined overrides accept `embedding.truncate_dim` even when omitted from
YAML; it defaults to null.

Changing model or dimension requires fresh reference/cache tables and output
directories, with all enabled reference inputs re-embedded using the evaluation
settings. Equal vector dimensions do not make different models compatible. Keep
reference inputs, saved descriptions, image caps, and retrieval settings identical
for model comparisons. The runner deliberately performs no compatibility checking,
migration, or automatic artifact naming; see the
[artifact contract](experiments.md#artifact-and-failure-contract).

### Environment and hardware selection

Select the Pixi environment matching the model family in
[development](../development.md#environment-selection). Dependency-family and
hardware guards are absent from the loader, so environment and GPU selection remain
manual:

- Target native BF16 hardware (e.g., A100/H100) for Jina v4 and Qwen.
- Qwen has no precision adaptation or batch-size fallback.
- Use [script operations](scripts.md) for submission defaults, environment
  forwarding, and sweep batch sizes, and [experiment notes](../experiments.md) for
  dependency-isolation details and GPU smoke checks.

## Vertex

`vertex.VertexEmbeddingModel` maps the application ID `google/ge-v2` to Vertex AI
`gemini-embedding-2`. It uses `google-genai`, API `v1`, explicit project/location,
and Application Default Credentials outside YAML. It does not construct a local
Sentence Transformer or use local-model preprocessing. `embedding.device` is
ignored; experiment `batch_size` still bounds dataset and insertion batches.

[`cfg/ge_v2.yml`](../../cfg/ge_v2.yml) is the standalone configuration. Supply
`--set embedding.vertex.project=PROJECT_ID` when constructing the backend;
complete cached evaluations allow an omitted/null project. Reference-source choices,
saved vocabulary path, cache setting, and artifact names are defined in that YAML.

Each nonempty encode call owns one context-managed client and a thread pool,
defaulting to four workers. Requests contain exactly one text record or one RGB
inline PNG image, with no resizing, filenames, labels, or dataset metadata. Image
serialization occurs in workers to bound serialized-image memory. The wrapper
closes its RGB copies and buffers while preserving caller-owned images. Empty
calls construct no client and require no authentication.

Native dimension is 3072. `truncate_dim` accepts dimensions 128–3072 and maps to API `output_dimensionality`; there is no local
slicing or normalization. Each response must contain exactly one finite nonzero
vector of the selected dimension.

`embedding.vertex.classification_prompt` defaults to false. When true, text starts
with `task: classification | query: `, and image requests place that exact text
part before the image. No `task_type` is sent. Apply the mode consistently to
reference and validation inputs. Changing prompt mode requires the same fresh
tables, outputs, and re-embedding as changing model or dimension.

`ExperimentConfig` validates setting types and ranges. The constructor itself
only rejects an empty project and dimensions outside 128–3072, before client
construction; direct construction with other bad values fails later. Location defaults to `global`; timeout defaults to 60 seconds per attempt.
SDK retries use five total attempts by default, one-second initial delay, base
two, one-second jitter, and a 60-second cap. Connection/timeouts and HTTP
408/429/500/502/503/504 retry; other errors propagate. Failure or interruption
cancels queued work, waits for running requests, and closes the client. Failed
batches return no vectors; a restart may repeat paid requests from that batch.

## Verification

Use [`test_embedder.py`](../../tests/test_embedder.py) for interface/import
boundaries, [`test_embeddings.py`](../../tests/test_embeddings.py) for local model
contracts, and [`test_vertex_embeddings.py`](../../tests/test_vertex_embeddings.py)
for real SDK behavior through an in-memory HTTP transport. Local tests patch
`encoders.sentence_transformers._load_backend()` or pretrained construction;
`EmbeddingModel` has no backend-injection argument. Retain explicit dependency
skips. Synthetic Sentence Transformers modules exercise library batching, order,
and conversion without model weights.
