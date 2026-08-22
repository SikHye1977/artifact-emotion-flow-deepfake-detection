#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

AASIST_URL="https://github.com/clovaai/aasist.git"
AASIST_COMMIT="a04c9863f63d44471dde8a6abcb3b082b07cd1d1"

MAE_DFER_URL="https://github.com/sunlicai/MAE-DFER.git"
MAE_DFER_COMMIT="81fcf589bb584a7a255e47af0e802c12b25f9eba"

SYNCNET_URL="https://github.com/joonson/syncnet_python.git"
SYNCNET_COMMIT="907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"

clone_or_update() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local target="${PROJECT_ROOT}/${name}"

    if [[ -e "${target}" && ! -d "${target}/.git" ]]; then
        echo "ERROR: ${target} already exists but is not a Git repository." >&2
        echo "Move or remove it manually, then run this script again." >&2
        return 1
    fi

    if [[ ! -d "${target}/.git" ]]; then
        echo "Cloning ${name}..."
        git clone "${url}" "${target}"
    else
        echo "Using existing ${name} repository."
    fi

    git -C "${target}" fetch --all --tags --prune
    git -C "${target}" checkout --detach "${commit}"
    echo "Configured ${name} at ${commit}."
}

command -v git >/dev/null 2>&1 || {
    echo "ERROR: git is not installed or not available in PATH." >&2
    exit 1
}

clone_or_update "aasist" "${AASIST_URL}" "${AASIST_COMMIT}"
clone_or_update "mae_dfer" "${MAE_DFER_URL}" "${MAE_DFER_COMMIT}"
clone_or_update "syncnet_python" "${SYNCNET_URL}" "${SYNCNET_COMMIT}"

echo "Installing project-specific SyncNet scripts..."
cp "${SCRIPT_DIR}/syncnet_custom/"*.py "${PROJECT_ROOT}/syncnet_python/"

echo
echo "External repositories are ready."
echo "Pretrained weights and trained checkpoints are not downloaded by this script."
echo "Copy them separately and verify them with checkpoint_sha256.txt."

