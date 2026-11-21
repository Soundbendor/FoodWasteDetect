#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=dataset-acceptance
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
ulimit -c 0
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
report_dir="${1:?Provide an acceptance report directory}"
mkdir -p "$report_dir"
export CUDA_VISIBLE_DEVICES=""
run=(pixi run --manifest-path src/classification/pyproject.toml --as-is -e default)
"${run[@]}" python -c 'import shutil; assert shutil.which("7z"), "7z missing"' || exit 1
status=0
printf 'dataset\tdownload\tvalidation\trepeat\n' > "$report_dir/summary.tsv"
for name in food101 uecfood256 foodx251 vegfru isia500; do
    config="cfg/classification_${name}.yml"
    if [[ "$name" == foodx251 ]]; then config=cfg/classification.yml; fi
    echo "$(date -Is) Starting $name"
    if ! "${run[@]}" python - "$name" <<'PY'
import shutil
import sys
import httpx
name = sys.argv[1]
# Conservative per-install headroom, checked again after previous installations.
needed = {"food101": 20, "uecfood256": 15, "foodx251": 60, "vegfru": 90, "isia500": 110}[name]
assert shutil.disk_usage(".").free >= needed * 1024**3, f"Need {needed} GiB free for {name}"
if name == "isia500":
    from classification.datasets.isia500 import METADATA_URL
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with client.stream("GET", METADATA_URL) as response:
            response.raise_for_status()
            assert response.headers.get("Content-Length"), "Metadata response lacks Content-Length"
PY
    then
        printf '%s\tpreflight_failed\tskipped\tskipped\n' "$name" >> "$report_dir/summary.tsv"
        status=1
        continue
    fi
    if ! "${run[@]}" classification download_dataset --config_file "$config" > "$report_dir/$name.download.log" 2>&1; then
        printf '%s\tfailed\tskipped\tskipped\n' "$name" >> "$report_dir/summary.tsv"
        status=1
        continue
    fi
    if ! "${run[@]}" python -m classification.scripts.check_dataset --config-file "$config" --report-file "$report_dir/$name.json"; then
        printf '%s\tpassed\tfailed\tskipped\n' "$name" >> "$report_dir/summary.tsv"
        status=1
        continue
    fi
    if "${run[@]}" classification download_dataset --config_file "$config" > "$report_dir/$name.repeat.log" 2>&1 &&
       "${run[@]}" python -m classification.scripts.check_dataset --config-file "$config" --report-file "$report_dir/$name.repeat.json" &&
       "${run[@]}" python -c 'import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); a.pop("elapsed_seconds"); b.pop("elapsed_seconds"); assert a == b' "$report_dir/$name.json" "$report_dir/$name.repeat.json"; then
        printf '%s\tpassed\tpassed\tpassed\n' "$name" >> "$report_dir/summary.tsv"
    else
        printf '%s\tpassed\tpassed\tfailed\n' "$name" >> "$report_dir/summary.tsv"
        status=1
    fi
done
exit "$status"
