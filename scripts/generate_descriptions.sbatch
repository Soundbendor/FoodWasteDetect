#!/usr/bin/env bash
#SBATCH --job-name=food-descriptions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

# Submit from the repository root with your Python environment activated and
# the configured Ollama model already installed (set OLLAMA_MODELS if needed).
# Adjust resources above and pass your cluster's partition/account to sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
# Make parent module src/ paths visible at runtime (if package not installed via pip install -e .)
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# DescriberLLM manages Ollama using the YAML configuration.
python -m internvl experiment describe \
  --config_file "${1:-cfg/classification.yml}"
