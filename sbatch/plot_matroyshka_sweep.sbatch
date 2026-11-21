#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=matroyshka-plot
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
run_dir="${1:?Provide the sweep submission directory}"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matroyshka-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=1
# Failed sweeps still yield a table naming the missing runs and a partial figure.
pixi run --manifest-path scripts/plotting/pixi.toml --as-is collect "$run_dir" --allow-partial
exec pixi run --manifest-path scripts/plotting/pixi.toml --as-is plot \
  "$run_dir/results.csv" --output "$run_dir/plots/matroyshka_accuracy.png"
