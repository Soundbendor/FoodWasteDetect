# Ollama image classification

Explore direct food-image classification through a vision-capable Ollama
model, such as the currently configured `gemma3:27b`. Send an image and
the allowed categories, request a JSON-schema-constrained label, and
validate the response.

Add a separate image-classification adapter that shares Ollama client
and server-lifecycle code with description generation. Build the category
enum from the dataset labels and explicitly decide whether to allow
unknown predictions. Do not reuse the description method's behavior of
overwriting the returned class with a supplied label.

This provides a direct-classification baseline alongside the selected
YOLO/Jina/Qdrant architecture, bypassing InternVL captioning and retrieval
for this experiment. Structured output does not guarantee accuracy or
provide instance localization. Evaluate predictions on labeled validation
images before drawing comparisons.

Status: proposed; no implementation or model inference performed.
