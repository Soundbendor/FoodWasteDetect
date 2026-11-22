#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=download-datasets
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
log_dir="${1:?Provide a log directory}"
mkdir -p "$log_dir"
export CUDA_VISIBLE_DEVICES=""
run=(pixi run --manifest-path src/classification/pyproject.toml --as-is -e default)
# Conservative per-install headroom in GiB, checked again after previous installations.
declare -A needed_gib=([food101]=20 [uecfood256]=15 [foodx251]=60 [vegfru]=90 [isia500]=110)
summary="$log_dir/summary.tsv"
status=0

record() { printf '%s\t%s\t%s\n' "$name" "$@" >> "$summary"; }
fail() { record "$@"; status=1; }

printf 'dataset\tdownload\tvalidation\n' > "$summary"
for name in food101 uecfood256 foodx251 vegfru isia500; do
    config="cfg/classification_${name}.yml"
    if [[ "$name" == foodx251 ]]; then config=cfg/classification.yml; fi
    echo "$(date -Is) Starting $name"
    if (( $(df --output=avail -B1 . | tail -n 1) < needed_gib[$name] * 1024**3 )); then
        echo "Need ${needed_gib[$name]} GiB free for $name" >&2
        fail preflight_failed skipped
        continue
    fi
    if ! "${run[@]}" classification download_dataset --config_file "$config" > "$log_dir/$name.download.log" 2>&1; then
        fail failed skipped
        continue
    fi
    if "${run[@]}" python -m classification.scripts.check_dataset --config-file "$config" > "$log_dir/$name.check.log" 2>&1; then
        record passed passed
    else
        fail passed failed
    fi
done
exit "$status"
