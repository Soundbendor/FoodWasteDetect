# Project scope and experiment families

Read this when choosing an experiment, interpreting branch history, or planning
work across packages. The repository contains offline food-recognition experiments;
each experiment has its own data, model, retrieval, and evaluation contract.

## Research direction

The selected architecture is YOLO localization, Jina CLIP image embeddings, and
Qdrant similarity classification. Its target is described in
[CLIP architecture](clip/architecture.md); the implementation still has
[reviewed correctness gaps](clip/review.md). Existing box/class outputs do not
establish instance segmentation.

The `classification` package provides a separate whole-image classification baseline:
embed pixels, retrieve labeled references from LanceDB, and vote on the class.
Reference sources can combine class names, generated descriptions, and training
images. Local multimodal encoders and the Vertex comparison share this pipeline.
Previously named `internvl`, the package embeds query images directly. Start with its [package guide](../src/classification/AGENTS.md).

Direct YOLO-E, hosted vision models, and the proposed
[Ollama image classifier](plans/ollama_image_classification.md) are comparison
experiments. They do not replace the selected architecture by default.

## Branch and migration context

| Context | Interpretation |
| --- | --- |
| `clip` | Selected research design; the recorded review examined commit `204e467696551600f0d57c1485b840505dbf2e58` (December 2, 2025) on October 19, 2026. |
| `main` | Older InternVL caption/text-retrieval baseline at the time of that review. |
| `fw-eval` | Inherited much of `clip` and added experiments; its date does not select the architecture. |
| `vlm-refactor` | Structural extraction into `src/internvl` (now `src/classification`); runtime repairs, caching guarantees, training redesign, and model changes were initially deferred. Later authorized changes are recorded in the current module docs. |

Recheck code before acting on branch-era findings. Keep refactors within their
requested behavior; historical plans do not authorize deferred features.
The [retired text workflow](history/internvl-text.md) explains the earlier
captioning baseline; current package contracts supersede historical API and
storage descriptions.

## Ownership

The root package owns the legacy CLIP/YOLO scripts, shared crop/dataset wrappers,
and Qdrant backend. Its package discovery excludes `classification`.
`src/classification/pyproject.toml` and its lockfile own the standalone `classification`
package, CLI, and Pixi environments. Keep storage and model assumptions scoped to
their experiment: CLIP provenance requirements do not override the whole-image
runner's deliberately deferred compatibility and provenance machinery.
