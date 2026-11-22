# To investigate

Possible future experiments. Nothing here is implemented or scheduled.

## Food image caption prediction

The classification package predicts one label per image by retrieval. A
follow-up task is predicting free text for an image (dish name, ingredients,
cooking method) and scoring it against reference text, either by retrieving
the nearest reference caption or by generating one with a vision-language
model.

Candidate datasets:

| Dataset | Contents | Notes |
| --- | --- | --- |
| UPMC Food-101 | ~90k web images in the Food-101 classes, each with text from its source page; train/test split | HF mirror `kkim0451/UPMC-Food101` (single zip). Text is noisy web content, not written captions. |
| [MM-Food-100K](https://huggingface.co/datasets/Humanbased-AI/MM-Food-100K) | 100,000 rows with `dish_name`, `food_type`, `ingredients`, `portion_size`, `nutritional_profile`, `cooking_method` | Single `train` split, so we would define the evaluation split. Rows hold `image_url`, not image bytes. OpenRAIL license. |

Questions to answer first:

- Is the target a retrieved caption, a generated caption, or structured
  fields such as ingredients?
- Which metric: text-embedding similarity, recall@k over reference captions,
  or field-level accuracy for `dish_name` and `ingredients`?
- How were the MM-Food-100K annotations produced, and are the image URLs
  stable enough to build a fixed evaluation set?
- Can `FoodImageDataset` carry a caption column, or does this need its own
  dataset contract? See the
  [dataset plan](plans/add_classification_datasets.md).
