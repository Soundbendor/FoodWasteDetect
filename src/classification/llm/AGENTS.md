# LLM description generation

Read [description contracts](../../../docs/classification/data.md#class-descriptions-and-ollama)
before changing prompts, structured responses, generation, or Ollama lifecycle.
For download command changes, also read [CLI downloads](../../../docs/classification/cli.md#ollama-downloads).

`ollama.py` owns the Ollama client, server process, and downloads.
`generate_descriptors.py` receives a client and model name for description requests.
`experiments.classification` owns vocabulary selection and persistence; `cli`
owns command dispatch. Keep those boundaries when extending generation.
Import concrete modules directly and keep the package initializer lightweight.

Verify lifecycle changes with `tests/test_descriptor_lifecycle.py` using fake
HTTP and process backends. For caller or progress changes, also run the
classification, CLI, and tracking tests named in the description and CLI docs.
