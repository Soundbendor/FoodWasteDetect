#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root with cu126, model caches, and shared LanceDB storage prepared.
config="${1:-cfg/classification_embeddinggemma2_images.yml}"
if (( $# > 1 )); then
  echo "Usage: bash scripts/submit_image_count_sweep.sh [config.yml]" >&2
  exit 2
fi
if [[ ! -f "$config" || ! -f sbatch/run_embedding.sbatch ]]; then
  echo "Run from the repository root and supply an existing config file." >&2
  exit 2
fi

sweep_id="embeddinggemma2_images_$(date -u +%Y%m%dT%H%M%SZ)_$$"

for count in 1 5 10 20 50 100; do
  run_id="${sweep_id}_n${count}"
  overrides=(
    --set "embedding.model_id=google/embeddinggemma-2"
    --set "embedding.device=cuda"
    --set "embed_images=true"
    --set "embed_class_labels=false"
    --set "embed_descriptions=false"
    --set "n_imgs_per_class=$count"
    --set "output_dir=assets/sweeps/$run_id"
    --set "lancedb.table=$run_id"
  )

  printf 'Submitting %s with config %q (INTERNVL_PIXI_ENV=cu126)\n' "$run_id" "$config"
  printf '  %s %q\n' "${overrides[@]}"
  INTERNVL_PIXI_ENV=cu126 sbatch \
    --job-name="$run_id" \
    --export=ALL,INTERNVL_PIXI_ENV=cu126 \
    sbatch/run_embedding.sbatch "$config" \
    "${overrides[@]}"
done
