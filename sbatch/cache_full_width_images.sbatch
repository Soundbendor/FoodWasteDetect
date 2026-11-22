#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=full-width-images
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
model="${1:?Provide qwen, jina_v4, or embeddinggemma2}"
case "$model" in
  jina_v4) pixi_env=jina-cu126 ;;
  qwen|embeddinggemma2) pixi_env=cu126 ;;
  *) echo "Unsupported model: $model" >&2; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Build all training-image references, fill the validation cache, then evaluate.
# Model allowlist above excludes the paid Vertex backend.
exec pixi run --manifest-path src/classification/pyproject.toml --as-is -e "$pixi_env" \
  classification experiment run \
  --config_file "cfg/classification_${model}_images_all_full.yml"
