#!/bin/bash

set -euo pipefail

# Resolve the repository root relative to this script so the command works
# from any current working directory on the local machine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_PROFILE="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_PROFILE}" ]]; then
    echo "Could not find conda profile at ${CONDA_PROFILE}."
    echo "Set CONDA_PROFILE manually or activate your environment before running."
    exit 1
fi

# Source conda in the current shell so activation works in a non-interactive script.
source "${CONDA_PROFILE}"
conda activate "${CONDA_ENV_NAME:-machine-learning}"

cd "${REPO_ROOT}"

# Mode B reuses the best checkpoint by default, but env overrides still work.
export PYTHONPATH="${REPO_ROOT}/siamese/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIGNATURE_RUN_NAME_PREFIX="${SIGNATURE_RUN_NAME_PREFIX:-siamese_full_hpo_lr5e4_m075}"

python -m signature_siamese.evaluate_cedar_mode_b
