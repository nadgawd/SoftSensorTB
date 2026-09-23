#!/usr/bin/env bash
# Download the configured model to scratch. Run on a LOGIN node.
#
#   bash hpc/fetch_model.sh                          # default profile, qwen3.5-27b
#   MODEL_PROFILE=qwen2.5-7b bash hpc/fetch_model.sh # the 7B fallback
#   MODEL_REPO=Qwen/Qwen2.5-14B-Instruct bash hpc/fetch_model.sh
#
# Exists so long-running downloads can be launched under tmux with one simple
# command, instead of a nested-quoted one-liner that silently mangles $ENV_PY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

[[ -x "${ENV_PY}" ]] || { echo "ERROR: no env; run hpc/setup_scratch.sh first" >&2; exit 1; }

echo "Repo:   ${MODEL_REPO}"
echo "Target: ${MODEL_PATH}"
echo "Proxy:  ${https_proxy:-none}"

# -u so progress reaches tee/tmux instead of sitting in a stdio buffer for
# twenty minutes and looking like a hang.
"${ENV_PY}" -u "${HERE}/download_model.py" --repo "${MODEL_REPO}"

echo ""
echo "Size on disk:"
du -sh "${MODEL_PATH}"
