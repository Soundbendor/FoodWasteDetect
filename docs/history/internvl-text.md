# Retired InternVL caption and text-retrieval experiment

Read this to interpret old text artifacts, commands, or branch decisions. This
workflow is removed; source and model code are available in Git history. Current
whole-image code is mapped in the [package guide](../../src/classification/AGENTS.md).

## Historical pipeline

1. Ollama received FoodX251 class names and attempted five generic visual
   descriptions per class, configured with `gemma3:27b`. It received labels,
   not image pixels, and produced a JSON dictionary of description lists.
2. A text embedder, configured with `BAAI/bge-large-en-v1.5`, encoded reference
   sentences for Qdrant with their descriptions and class labels. Ollama generated
   text, not vectors.
3. `InternVLM` received a whole image and a visual-description prompt without the
   ground-truth label. `InternVL2_5-8B-MPO` loaded through Transformers on CUDA.
4. The compatible text embedder encoded the returned caption. Qdrant retrieved up
   to ten descriptions, optionally reranked by a cross-encoder, and classified by
   their labels, usually majority vote. Ollama was not called per query after
   reference generation. Collection compatibility was unchecked.

This was whole-image classification, without YOLO or instance masks. Per-object
InternVL captioning after detection was a proposed extension. Training-image
captions could also be saved, merged from four partitions, and grouped by known
class into reference dictionaries. Those dictionaries share the class-describer
format but have different provenance; preserve that distinction.

## Removed interfaces

The former `model.py`, `workflows.py`, and flat `embeddings.py` are absent from the
current package. Concrete encoders now live in `internvl.encoders`. BGE training,
triplet evaluation, caption/merge/conversion workflows, and their CLI commands are
retired without compatibility aliases. Old `vlm`, `llm`, and `database.embedding`
imports are unsupported. The separate `src/clip_finetune.py` image experiment and
root `build_clip_db`, `test_clip`, and `test_yolo` commands remain separate.
See [CLI documentation](../classification/cli.md) for supported commands and migration.

The `vlm-refactor` extraction originally preserved runtime behavior and deferred
bug fixes, caching/restart guarantees, training redesign, and model changes.
Later authorized whole-image work superseded that scope for the current runner;
the old deferrals do not erase its implemented batch resumption.

## Historical evaluation and limitations

The example evaluation sampled 1,000 FoodX251 validation images with seed 42 and
reported top-1, top-5, and voting accuracy. Top-5 counted description records,
possibly repeating classes. These scores measured captioning plus retrieval;
they did not isolate caption quality or visual factuality. JSON structure checks,
replacing the generated class field with the supplied label, and the demonstration
macaron query did not establish accuracy. No direct descriptor-factuality or
human-reference caption evaluation was implemented.

Recorded defects in the deleted workflows included:

- INI-style lookups against YAML configuration and passing an anchor-file path as
  the embedding model name.
- Text insertion with incompatible class/metadata arguments and missing IDs.
- Accessing the Qdrant reranker response incorrectly.
- Caption resumption checking Series indices instead of saved image identities,
  and losing the final buffered captions.
- Retrieval callers requiring the removed training-dependent embedding interface.

These findings do not describe the current runner. Extraction tests established
command wiring, import boundaries, and selected artifact behavior only.
