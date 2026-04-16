#!/bin/bash

set -euo pipefail

# Resolve the repository root relative to this script so the command works
# from any current working directory on the local machine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_PROFILE="${HOME}/miniconda3/etc/profile.d/conda.sh"
TARGET_CONDA_ENV="${CONDA_ENV_NAME:-machine-learning}"

if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    # Reuse an already-active environment to avoid forcing a specific local setup.
    echo "Using already active conda environment: ${CONDA_DEFAULT_ENV}"
elif [[ -f "${CONDA_PROFILE}" ]]; then
    # Source conda in the current shell so activation works in a non-interactive script.
    source "${CONDA_PROFILE}"
    conda activate "${TARGET_CONDA_ENV}"
else
    echo "No active conda environment detected and could not find conda profile at ${CONDA_PROFILE}."
    echo "Activate an environment manually or set CONDA_PROFILE and rerun."
    exit 1
fi

cd "${REPO_ROOT}"

# Default to the user-provided validation pair while still allowing env overrides.
export PYTHONPATH="${REPO_ROOT}/siamese/src${PYTHONPATH:+:${PYTHONPATH}}"
export SIGNATURE_RUN_NAME_PREFIX="${SIGNATURE_RUN_NAME_PREFIX:-siamese_full_hpo_lr5e4_m075}"
export SIGNATURE_PAIR_IMAGE_A="${SIGNATURE_PAIR_IMAGE_A:-validation/B-S-83-G-04.tif}"
export SIGNATURE_PAIR_IMAGE_B="${SIGNATURE_PAIR_IMAGE_B:-validation/B-S-83-F-05.tif}"
export SIGNATURE_PAIR_EXPECTED_LABEL="${SIGNATURE_PAIR_EXPECTED_LABEL:-0}"

python -m signature_siamese.infer_single_pair
