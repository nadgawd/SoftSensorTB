#!/usr/bin/env bash
# One-time setup: scratch dirs, Miniconda on scratch, the local_llm env.
#
#   source hpc/env.sh
#   bash hpc/proxy_login.sh      # only if this node has no internet yet
#   bash hpc/setup_scratch.sh
#
# Run this on a LOGIN node. It needs no GPU, and it downloads ~4 GB of wheels,
# which is not what your GPU allocation is for.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

# Fail loudly rather than creating a path you don't own. /scratch/<id> does not
# exist on this cluster; the department/programme prefix is mandatory.
parent="$(dirname "${SCRATCH_DIR}")"
if [[ ! -d "${parent}" ]]; then
  echo "ERROR: ${parent} does not exist." >&2
  echo "Find your real allocation with:  ls -l /scratch/${HPC_DEPT}" >&2
  echo "Then re-run with SCRATCH_DIR=/your/real/path bash hpc/setup_scratch.sh" >&2
  exit 1
fi

echo "Scratch:  ${SCRATCH_DIR}"
echo "Conda:    ${CONDA_ROOT}"
echo "Project:  ${PROJECT_CODE}"

mkdir -p "${SCRATCH_DIR}"/{hf_cache,models,logs,envs,conda_pkgs,pip_cache}

mark_env() {
  local line="$1"
  grep -Fqx "$line" ~/.bashrc 2>/dev/null || echo "$line" >> ~/.bashrc
}

# Persist only the things a fresh shell genuinely needs. The rest comes from
# sourcing hpc/env.sh, so there is one place to change them.
mark_env "export SCRATCH_DIR=${SCRATCH_DIR}"
mark_env "export HF_HOME=${SCRATCH_DIR}/hf_cache"
mark_env "export HF_XET_HIGH_PERFORMANCE=1"
mark_env "export PYTHONNOUSERSITE=1"
mark_env "export CONDA_PKGS_DIRS=${SCRATCH_DIR}/conda_pkgs"
mark_env "export PIP_CACHE_DIR=${SCRATCH_DIR}/pip_cache"

# Anything previously installed with `pip install --user` sits in ~/.local and
# wins over the conda env on sys.path. A transformers 4.18 left there will
# shadow the modern one and fail to recognise any Qwen architecture.
if [[ -d "$HOME/.local/lib" ]]; then
  echo ""
  echo "NOTE: ~/.local/lib exists (from an earlier 'pip install --user')."
  echo "      PYTHONNOUSERSITE=1 neutralises it. To reclaim the home quota:"
  echo "      rm -rf ~/.local/lib/python3.*/site-packages"
fi

# The login node reaches the internet through the campus proxy without any
# portal login, so this is all that's needed here. Compute nodes are stricter,
# which is why proxy_login.sh still exists.
proxy_on

load_conda

# --prefix, not --name: a named env lands in ~/.conda/envs and eats the home
# quota that torch alone would exhaust.
if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
  conda create --prefix "${CONDA_ENV_PREFIX}" python=3.10 -y
fi

# Everything below goes through $ENV_PY, never a bare `pip`. See env.sh.
[[ -x "${ENV_PY}" ]] || { echo "ERROR: no interpreter at ${ENV_PY}" >&2; exit 1; }
echo "Installing into: $("${ENV_PY}" -c 'import sys; print(sys.prefix)')"

# No `pip install --upgrade pip`: conda ships a current pip in the env, and the
# upgrade resolves the "existing installation" to the read-only system conda.
#
# torch is NOT installed separately. vLLM pins an exact torch, so installing one
# first only downloads ~3 GB that the next command immediately replaces. The
# A100 nodes run driver 590 (CUDA 13.1), so whatever CUDA build vLLM picks works.
"${ENV_PY}" -m pip install vllm huggingface_hub hf_transfer transformers accelerate datasets trl peft bitsandbytes

echo ""
echo "Installed:"
"${ENV_PY}" - <<'PY'
import torch, vllm, transformers
print(f"  torch        {torch.__version__}")
print(f"  vllm         {vllm.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  CUDA build   {torch.version.cuda}")
PY

cat <<EOF

Env: ${CONDA_ENV_PREFIX}

Done. Next:
  source hpc/env.sh && activate_env
  python hpc/download_model.py --repo ${MODEL_REPO}
  qsub hpc/serve_llm.pbs
EOF
