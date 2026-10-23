#!/bin/bash
#SBATCH -p dgx2
#SBATCH -A eecs
#SBATCH --job-name=intern-fw
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Prepare cu126 and model caches before submitting from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
config_file="${1:-cfg/classification.yml}"
if (( $# )); then shift; fi
exec pixi run --manifest-path src/internvl/pyproject.toml --as-is -e cu126 internvl experiment evaluate \
  --config_file "$config_file" "$@"
