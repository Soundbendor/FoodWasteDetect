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
current package. Concrete encoders now live in `classification.encoders`. BGE training,
triplet evaluation, caption/merge/conversion workflows, and their CLI commands are
retired without compatibility aliases. Old `vlm`, `llm`, and `database.embedding`
imports are unsupported. The separate `src/clip_finetune.py` image experiment and
root `build_clip_db`, `test_clip`, and `test_yolo` commands remain separate.
See [CLI documentation](../classification/cli.md) for supported commands.

## Historical evaluation

The example evaluation sampled 1,000 FoodX251 validation images with seed 42 and
reported top-1, top-5, and voting accuracy. Top-5 counted description records,
possibly repeating classes. These scores measured captioning plus retrieval;
they did not isolate caption quality or visual factuality.
