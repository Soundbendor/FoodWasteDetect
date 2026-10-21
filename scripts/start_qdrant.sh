#!/usr/bin/env bash
set -euo pipefail
export QDRANT__STORAGE__STORAGE_PATH="${QDRANT__STORAGE__STORAGE_PATH:-assets/qdrant/1.19.0/storage}"
export QDRANT__STORAGE__SNAPSHOTS_PATH="${QDRANT__STORAGE__SNAPSHOTS_PATH:-assets/qdrant/1.19.0/snapshots}"
export QDRANT__SERVICE__HOST="${QDRANT__SERVICE__HOST:-127.0.0.1}"
exec qdrant "$@"
