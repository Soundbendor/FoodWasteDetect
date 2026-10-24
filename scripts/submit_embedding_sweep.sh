#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root with cu126, model caches, and shared LanceDB storage prepared.
# Jobs sharing a node need distinct Ollama endpoints when generating descriptions.
config="${1:-cfg/classification.yml}"
if (( $# > 1 )); then
  echo "Usage: bash scripts/submit_embedding_sweep.sh [config.yml]" >&2
  exit 2
fi
if [[ ! -f "$config" || ! -f sbatch/run_embedding.sbatch ]]; then
  echo "Run from the repository root and supply an existing config file." >&2
  exit 2
fi

sweep_id="embedding_$(date -u +%Y%m%dT%H%M%SZ)_$$"

models=(
  "jinaai/jina-clip-v2"
  # Add other model IDs once the embedding backend supports them.
)

for i in "${!models[@]}"; do
  model="${models[$i]}"
  run_id="${sweep_id}_${i}"

  echo "Submitting $model as $run_id"
  sbatch \
    --job-name="$run_id" \
    sbatch/run_embedding.sbatch "$config" \
    --set "embedding.model_id=$model" \
    --set "embedding.device=cuda" \
    --set "output_dir=assets/sweeps/$run_id" \
    --set "lancedb.table=$run_id"
done
