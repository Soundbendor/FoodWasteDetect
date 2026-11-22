#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=matroyshka
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
config="${1:?Provide the prepared config JSON}"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Cached evaluation constructs no model, so all models use the prepared CPU env.
exec pixi run --manifest-path src/classification/pyproject.toml --as-is -e default \
  classification experiment matroyshka-sweep --config_file "$config"
