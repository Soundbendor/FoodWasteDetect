#!/bin/bash
#SBATCH -p dgx2
#SBATCH -A eecs
#SBATCH --job-name=qdrant_server
#SBATCH -t 1-00:00:00
#SBATCH -c 2
#SBATCH --mem=20G
#SBATCH --export=ALL

# Prepare the default Pixi environment before submitting from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
exec pixi run --manifest-path src/internvl/pyproject.toml --as-is -e default qdrant
