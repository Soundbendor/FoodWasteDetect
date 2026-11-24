# DescriberLLM captions and class prototypes

Measure two changes independently. First, build one description prototype per
class from the existing captions. Then generate a new vocabulary using several
visual questions and build prototypes the same way. Comparing the two prototype
runs isolates the effect of changing the questions sent to Ollama.

## 1. Generate several descriptions per food

Start with five ordered generator templates. Each asks for one short sentence
that names `{label}` and describes something visible in a photograph. Send the
questions to Ollama and embed its answers.

| Template | Question for Ollama |
| --- | --- |
| Overall appearance | What does `{label}` typically look like in a food photograph? |
| Shape and arrangement | What shape or arrangement of pieces helps someone recognize `{label}`? |
| Surface | What visible surface pattern, texture, or color variation helps identify `{label}`? |
| Distinguishing feature | Which visible feature of `{label}` is especially useful for telling it apart from similar foods? |
| Common variation | Describe another common way `{label}` appears when photographed whole, cut, or served. Choose one form appropriate to the food. |

Keep `description_count` as the total number of captions per class. At the
current count of five, use each template once. For smaller counts, take the
first templates in order. For larger counts, cycle through them. Keep one
response per call so each caption has a known template and a failed response
can be traced to one request. [Current generator](../../src/classification/llm/generate_descriptors.py)

Preserve the saved `descriptions` mapping so existing vocabularies can be
indexed immediately. For newly generated files, also record the Ollama model,
ordered template IDs, and exact template text. The default config points at a
read-only vocabulary. Document how to generate a new one using
`descriptions_file=null` and a fresh `output_dir`.
[Vocabulary lifecycle](../classification/data.md#class-descriptions-and-ollama)

## 2. Write one description prototype per class

For each canonical class with descriptions enabled, have `IndexBuilder`:

1. Read its ordered caption list. Validate that every requested class has at
   least one caption and every caption is nonblank before opening the encoder
   or writing rows.
2. Encode each caption separately using the backend's existing text protocol
   and configured batch size. Each SigLIP 2 caption gets its own token window.
3. L2-normalize each caption vector, take their equal-weight arithmetic mean,
   then L2-normalize the mean. Accumulate in float64 and store the final vector
   as float32. Fail with the class name if the mean is zero or nonfinite.
4. Append one `PointRecord` with source kind `description_prototype`, the
   canonical label as its source identity, and the contributing captions
   serialized as a JSON list in `value`. This fits the existing LanceDB schema
   and lets the reader inspect a prototype's inputs.

Keep a separately enabled `class_name` reference as its own row. It supplies
the `bare` and `food_photo` baselines. With descriptions alone, five captions
for each of 251 classes produce 251 reference rows from 1,255 caption
embeddings. Reports count stored prototypes. Embedding progress counts
captions processed. [Reference builder](../../src/classification/experiments/index_builder.py)

On resume, compare each completed prototype's saved caption list with the
requested list. Skip matching prototypes before embedding, and direct the
user to a fresh reference table if a list has changed. Reject a prototype
build into a table containing legacy `generated_description` rows, since
mixed rows would give a class both its old caption votes and its new prototype
vote. Existing tables remain available for evaluation.

## 3. Verify correctness and measure each change

- **Synthetic CPU checks.** Use fake Ollama responses to test template order,
  count, label substitution, and legacy vocabulary loading. Use vectors of
  unequal norms with a hand-calculated expected prototype to catch accidental
  averaging of raw vectors. Cover one-caption classes, batches crossing class
  boundaries, cancellation to zero, mixed reference sources, and interrupted
  builds that resume without duplicating rows.
- **FoodX251 comparison.** Evaluate the existing per-caption table. Build a
  fresh prototype table from the same saved captions, then another from the
  new multi-template captions. Hold the encoder, dimension, validation images,
  and image preprocessing fixed. Include the bare-label and food-photo runs as
  one-vector-per-class baselines.
- **Metrics.** Compare top-1 accuracy and per-class recall. Preserve the
  legacy run's majority-vote accuracy when comparing against the previous
  classifier. Use `k=1` for description-only prototype runs so the predicted
  class is the one whose prototype is closest to the image.

## Open questions for the caption workflow

| Question | Recommendation |
| --- | --- |
| Should templates be fixed, configurable per experiment, or overridable per class? | Start with a versioned shared set. Give every class the same generation budget and record exactly which templates produced the vocabulary. |
| What should happen if Ollama repeats a caption or omits the food name? | Request one replacement, then fail with the class and template ID if it is still invalid. A duplicate gives one description extra weight in the prototype. Apply this rule to newly generated captions so the original vocabulary remains available for a faithful comparison. |
| Should interrupted generation resume? | Checkpoint each completed class separately from the final vocabulary. Resume only when the class list, model, templates, and count match. At five captions per FoodX251 class, this can save repeating hundreds of requests. |
| Should generation enforce SigLIP 2's 64-token window? | Ask for concise captions and measure actual token lengths per encoder. Record how many captions are truncated before adding tokenizer-specific retries to a generator whose vocabulary is shared across models. |
