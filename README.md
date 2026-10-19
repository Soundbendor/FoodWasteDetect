# FoodWasteDetect

FoodWasteDetect contains two separate research pipelines. The primary experimental
direction is YOLO localization followed by Jina CLIP image embeddings and Qdrant
classification. The `internvl` package contains the older caption and text-retrieval
experiments. Its name identifies the current captioner; its Ollama describer is a
separate component with a different task.

This refactor separates the code and commands. It preserves existing experimental
behavior, including the runtime defects listed below. These are offline research
scripts, not a deployed inference service.

## How text-based inference works

1. The Ollama describer receives known FoodX251 class names and attempts five
   generic visual descriptions per class. It receives no image pixels. The
   checked-in configuration selects `gemma3:27b`.
2. A separate sentence embedding model, configured as `BAAI/bge-large-en-v1.5`,
   embeds the reference sentences. The database builder intends to store their
   vectors, text, and class labels in Qdrant.
3. InternVL receives a whole image and a visual-description prompt, without its
   ground-truth label. The configured checkpoint is `InternVL2_5-8B-MPO`, loaded
   directly through Transformers on CUDA.
4. The text embedder embeds the caption. Qdrant retrieves up to ten reference
   descriptions. An optional cross-encoder reranks them, and their labels are
   used for nearest-neighbor or majority-vote classification.

Ollama is needed to generate the reference descriptions, not for each image query
after the database is populated. Stored and query embeddings must use compatible
models and preprocessing; the current code does not enforce collection provenance.
Caption files can also be grouped by class to produce the same reference-dictionary
format. This permits image-derived reference descriptions as another experiment.

YOLO is not used by this workflow. The captioner processes whole images and does
not output boxes or instance masks. Per-object detection and captioning would be
a separate extension. Shared datasets import YOLO only when their crop/detection
methods are called.

## Package and command migration

The package contains `model.py` for InternVL, `descriptors.py` for Ollama,
`embeddings.py` for text embedding and training, and `workflows.py` for experiment
orchestration. `cli.py` dispatches commands without loading backends until needed.
Importing `internvl` or requesting CLI help does not load models or contact services.
Datasets, Qdrant, and classification metrics remain shared with the image path.

The old `vlm.intern`, `llm.describer`, and `database.embedding` imports have been
removed. Import their classes from `internvl.model`, `internvl.descriptors`, and
`internvl.embeddings`, respectively. Class and method names are retained. Workflow
functions accept a configuration dictionary; `caption_imgs` also takes `partition`.

| Previous command | Replacement |
| --- | --- |
| `caption_imgs` | `internvl caption-images` |
| `generate_descriptions` | `internvl generate-descriptions` |
| `build_vdb` | `internvl build-db` |
| `test_classification` | `internvl evaluate` |
| `finetune_embeddings` function, previously without a console entry | `internvl finetune-embeddings` |
| `merge_captions` | `internvl merge-captions` |
| `convert_dict` | `internvl convert-captions` |

There are no compatibility aliases. The `build_clip_db`, `test_clip`, and
`test_yolo` commands retain their names and implementations.

In an existing research environment, install the checkout with `python -m pip
install -e .`. Runtime dependencies are still managed by the research environment;
the project metadata does not install them. The existing environment file is
machine-specific and incomplete. Recreate an environment only after selecting the
appropriate Python, CUDA, and model-library versions.

To inspect the CLI directly from the checkout without installing it:

```sh
PYTHONPATH=src python -m internvl --help
PYTHONPATH=src python -m internvl caption-images --help
```

After installation, command syntax is:

```sh
internvl generate-descriptions --config_file cfg/experiment_config_internvl.yml
internvl build-db --config_file cfg/experiment_config_internvl.yml
internvl evaluate --config_file cfg/experiment_config_internvl.yml
internvl caption-images --config_file cfg/experiment_config_internvl.yml --partition 1
internvl finetune-embeddings --config_file cfg/experiment_config_internvl.yml
internvl merge-captions
internvl convert-captions
```

These examples document invocation, not a verified working experiment. Review the
known defects before running them. The text configuration was renamed from
`cfg/experiment_config.yml`; its values and interpretation are otherwise unchanged.
Use the repository root as the working directory for legacy relative paths.

Captioning still partitions the training set into four slices. Merge expects
`assets/foodx251_captions_1.txt` through `_4.txt`, containing CSV data, and writes
`assets/foodx251_captions.txt`. Conversion writes `foodx251_captions.json` in the
working directory, mapping each class to a list of captions. No artifact migration
or new resume guarantees are introduced.

Ollama scripts now reside in `src/internvl/scripts` and are included in the package.
Descriptor generation retains its existing server-start attempt if the chat probe
raises `ConnectionError`. Script names, model selections, and working-directory
assumptions are historical; the download script still names DeepSeek while the
experiment config selects Gemma. Do not run these scripts, GPU experiments, or
Slurm jobs without the required data and compute authorization.

## What the evaluation measures

The classification workflow compares retrieved labels with FoodX251 validation
labels. The example configuration selects 1,000 images with sampling seed 42.
It reports top-1, top-5, and voting accuracy. Top-5 checks the first five description
records, which can repeat classes; voting selects the most common label among up
to ten records. These measure the combined caption, embedding, and retrieval path.

Text embedding training also configures a cosine triplet evaluator. An image
caption is the anchor, a same-class description is the positive, and a different
class supplies the negative. Those descriptions come from the training portion
of the description dataset, and sampling is unseeded. This evaluates embedding
separation, not independent caption correctness.

There is no direct visual-factuality or human-reference caption evaluation.
Pydantic validates the Ollama response structure; the supplied class overwrites
the response's class field. Neither proves recognition accuracy. The hardcoded
macaron query after database construction is only a demonstration.

## Known defects preserved by this extraction

- Several text workflows use INI-style `cfg.get("Models", ...)` against YAML
  dictionaries, so they resolve configuration incorrectly.
- Classification passes the caption-anchor path as the embedding model name.
- Database construction omits required IDs and supplies incompatible labels and
  metadata to the shared insertion method.
- Reranking iterates the Qdrant response before accessing its points.
- Caption resume checks Series indices instead of stored image IDs, and the last
  buffered captions are not flushed. Do not rely on this command to preserve a
  complete caption run until repaired.
- Embedding construction eagerly loads training data even for inference. Dataset
  splitting and triplet sampling are unseeded; model and collection provenance
  remain unenforced.

## CPU verification

Run `PYTHONPATH=src python -m unittest discover -s tests -v`. CLI and dispatch
checks use the standard library. Artifact and shared-dataset checks require the
small CPU dependencies NumPy, pandas, Pillow, tqdm, and PyYAML; missing ones are
reported as skips. External model and service interfaces are mocked where needed.
These checks do not establish model accuracy, real Qdrant integration, or HPC
execution. No models, datasets, Ollama servers, or GPU jobs are started by the tests.
