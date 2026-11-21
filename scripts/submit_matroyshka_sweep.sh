#!/usr/bin/env bash
# Submit one cached, CPU-only dimension sweep per ready model.
set -euo pipefail

dry_run=false
if [[ "${1:-}" == --dry-run ]]; then
  dry_run=true
  shift
fi
if [[ ${1:-} == --help ]]; then
  echo "Usage: bash scripts/submit_matroyshka_sweep.sh [--dry-run] SOURCES.json"
  exit 0
fi
if (( $# != 1 )); then
  echo "Supply one sources JSON file. See cfg/matroyshka_sources.example.json." >&2
  exit 2
fi
sources=$1
if [[ ! -f "$sources" || ! -f sbatch/run_matroyshka_sweep.sbatch ]]; then
  echo "Run from the repository root and supply an existing sources file." >&2
  exit 2
fi
output="${MATROYSHKA_OUTPUT_ROOT:-assets/matroyshka}/$(date -u +%Y%m%dT%H%M%SZ)_$$"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default \
  python -m classification.scripts.prepare_matroyshka_sweep \
  --sources "$sources" --output-dir "$output"

while IFS=$'\t' read -r name config; do
  command=(sbatch --parsable --job-name="matroyshka-$name"
    sbatch/run_matroyshka_sweep.sbatch "$config")
  if "$dry_run"; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    job_id=$("${command[@]}")
    printf '%s\t%s\t%s\n' "$name" "$job_id" "$config" | tee -a "$output/submitted.tsv"
  fi
done < "$output/jobs.tsv"
echo "Resolved configs and job manifest: $output"
