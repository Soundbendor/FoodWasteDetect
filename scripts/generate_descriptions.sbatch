#!/usr/bin/env bash
#SBATCH --job-name=food-descriptions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root with Pixi available and cu126 prepared.
# Install the configured Ollama model first (set OLLAMA_MODELS if needed).
# Adjust resources above and pass your cluster's partition/account to sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"

# DescriberLLM manages Ollama using the YAML configuration.
pixi run --manifest-path src/internvl/pyproject.toml --as-is -e cu126 internvl experiment describe \
  --config_file "${1:-cfg/classification.yml}"
