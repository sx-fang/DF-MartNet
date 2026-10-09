#!/usr/bin/env bash
# B-prime submission boundary: a job-private frozen package is mandatory.
# This script never trains/evaluates on the submission host and never deletes.
set -euo pipefail
frozen=${1:?Expected the absolute frozen package directory}; shift
test -f "$frozen/PACKAGE_FILES.json"
test -f "$frozen/dfm_repro/gpu_worker.py"
export ENGINE_DIR="$frozen"
exec sbatch "$@"
