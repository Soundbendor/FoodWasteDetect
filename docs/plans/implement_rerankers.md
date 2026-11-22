# Rerankers for FoodX-251 whole-image classification

## Context

The current pipeline embeds the query image, searches LanceDB (class-name text,
descriptions, and training images), and takes a majority vote. The question is
whether a second-stage reranker can fix fine-grained confusions among the
top-k candidates and, if so, which models to use and what fine-tuning they need.

## Findings

- **The pattern is established.** RAR (TIP 2025) retrieves the top-k candidates
  with CLIP and has an MLLM rank the class names. On Food-101 it beats CLIP+kNN by
  +8 to +12 points with ViT-B/16 and by +3 to +5 points with ViT-L/14. Top-5
  accuracy barely moves (+0.5), so the gain comes from reordering candidates the
  retriever already found. The retrieval ceiling is therefore top-k recall, which
  we already log (`top5_hit_rate`).
- **The gain shrinks as the retriever gets stronger.** Measure it against our
  best embedder, not against B/32.
- **Fine-tuning helps.** RAR's ranking SFT (~30k groups from FGVC-Aircraft,
  1 epoch, lr 1e-5) beat in-context ranking by +4 points avg for Qwen-VL. The
  results are robust to k (3-7). Food was RAR's weak spot without a strong MLLM
  (Qwen-VL 88.1 vs GPT-4V 92.6 on Food-101).
- **Off-the-shelf rerankers are trained for retrieval/RAG relevance** (documents,
  screenshots), not for "is this dish X or near-duplicate Y". Expect a zero-shot
  baseline that is useful but miscalibrated across classes.

## Candidate models

| Model | Type | Why |
| --- | --- | --- |
| `Qwen/Qwen3-VL-Reranker-2B` | Pointwise cross-encoder (yes/no logit) | Pairs with our Qwen3-VL-Embedding entry; loads via ST `CrossEncoder` (already in our `modern` env); cheap enough to sweep |
| `Qwen/Qwen3-VL-Reranker-8B` | Same, larger | Upper bound for the pointwise approach |
| `jinaai/jina-reranker-m0` | Pointwise, Qwen2-VL 2B | Second family; check its license (CC-BY-NC) before adopting |
| Listwise generative MLLM (Qwen3-VL-8B-Instruct, RAR-style) | Prompted ranking of k class names | Sees all candidates at once, so it can contrast them; this is the setup RAR fine-tunes |

Skip the text-only rerankers. LightOn-rerank and Nemotron-3.5-Rerank are tuned
for document pages, so add them only if the Qwen pointwise models look promising.

## Proposed integration (when approved)

Insert a stage between `LanceDBIndex.search` and `voting.classify`:

```python
# retrieval/rerank.py
def rerank(query: Image, candidates: list[ClassCandidate], model: CrossEncoder) -> list[ClassCandidate]:
    docs = [f"{c.label}: {c.description}" for c in candidates]   # or a reference image
    scores = model.predict([(query, d) for d in docs])
    return sorted(zip(candidates, scores), key=lambda x: -x[1])
```

- Candidates are the top-k **distinct classes** from neighbors (k=5-10), not raw
  neighbors.
- Compare three document forms: class name, name + generated description, and
  1-3 reference images per class (image-image).
- Config: `rerank.model_id`, `rerank.k`, `rerank.document=name|description|image`;
  null disables. Report `rerank_accuracy` alongside vote accuracy and top-k recall.
- CPU tests use a fake `CrossEncoder`; real models need authorization.

## Fine-tuning (only if zero-shot gain < ~2 points over vote)

1. **Pointwise LoRA on Qwen3-VL-Reranker-2B.** Train on (query image, class doc)
   pairs from the FoodX-251 train split. Use hard negatives from our own
   retriever's top-k wrong classes and BCE/listwise CE over groups of k. LoRA
   r=32 on the LM (and optionally the vision tower), 1 epoch, ~30-50k groups
   (RAR scale). The ST `CrossEncoderTrainer` covers this; there is no official
   Qwen recipe.
2. **RAR-style listwise SFT** of Qwen3-VL-8B-Instruct (prompt: image + shuffled
   k names → sorted names). This costs more and is the better choice if
   contrastive decisions matter.
3. **Splits.** Mine negatives from the train split only, and keep the validation
   split untouched. Train-split references used by the retriever must not leak
   query images into the training groups. Report both in-domain (FoodX-251) and
   transfer (Food-101/UEC from the dataset plan) results, since bin images
   differ from web photos.

## Verification

- Unit: fake reranker reorders candidates, empty/one-class candidate lists, k
  larger than distinct classes.
- Experiment (authorized GPU): fixed retriever, compare vote vs rerank accuracy,
  top-1 and top-5, latency per query, at k ∈ {5, 10}.
