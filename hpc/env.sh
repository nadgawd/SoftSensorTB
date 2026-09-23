#!/usr/bin/env bash
# Shared settings for every script in hpc/. Source it, don't execute it:
#   source hpc/env.sh
#
# Everything here can be overridden from the environment, so a lab-mate on a
# different allocation only has to export a few variables rather than edit files.

# --- Identity -----------------------------------------------------------------
# Scratch on this cluster is carved up by department and programme, NOT by bare
# username: /scratch/<dept>/<programme>/<kerberos-id>. Deriving it as
# /scratch/$(whoami) silently produces a path you have no permission to create.
HPC_USER="${HPC_USER:-$(whoami)}"
HPC_DEPT="${HPC_DEPT:-chemical}"
HPC_PROGRAMME="${HPC_PROGRAMME:-dual}"

SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${HPC_DEPT}/${HPC_PROGRAMME}/${HPC_USER}/mtp}"

# Passed to qsub -P. Ask your supervisor if you don't have one.
PROJECT_CODE="${PROJECT_CODE:-ioe.che.omprakash.1}"

# --- Model --------------------------------------------------------------------
# MODEL_PROFILE picks the weights; serve_llm.pbs picks the matching vLLM flags.
# The profile name is also the alias vLLM serves under, so LOCAL_LLM_*_MODEL in
# .env is just the profile name, and run.sh reads the profile from there.
#
#   qwen3.5-27b  Qwen3.5-27B GPTQ-Int4, ~29 GB. Needs an A100 (sm_80). Default.
#   qwen2.5-7b   Qwen2.5-7B-Instruct, bf16, ~15 GB. Faster, weaker at multi-step
#                tool use. Kept as the fallback.
MODEL_PROFILE="${MODEL_PROFILE:-qwen3.5-27b}"
case "${MODEL_PROFILE}" in
  qwen3.5-27b) _profile_repo="Qwen/Qwen3.5-27B-GPTQ-Int4" ;;
  qwen2.5-7b)  _profile_repo="Qwen/Qwen2.5-7B-Instruct" ;;
  *)
    echo "Unknown MODEL_PROFILE='${MODEL_PROFILE}' (expected qwen3.5-27b or qwen2.5-7b)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
MODEL_REPO="${MODEL_REPO:-${_profile_repo}}"
unset _profile_repo
MODEL_NAME="${MODEL_NAME:-${MODEL_REPO##*/}}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_DIR}/models/${MODEL_NAME}}"

# Stable alias the backend talks to, so .env never tracks a filesystem path.
# A mismatch between this and LOCAL_LLM_*_MODEL is what produces 404s.
SERVED_NAME="${SERVED_NAME:-${MODEL_PROFILE}}"
SERVE_PORT="${SERVE_PORT:-8000}"

# --- Caches -------------------------------------------------------------------
# Home is small and quota'd; every byte of model cache belongs on scratch.
export HF_HOME="${HF_HOME:-${SCRATCH_DIR}/hf_cache}"
# hf_transfer is retired in current huggingface_hub; Xet is the replacement.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# conda and pip default their package caches to $HOME, where several GB of
# torch and vllm wheels would blow the quota. Both must live on scratch.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH_DIR}/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_DIR}/pip_cache}"

# Packages installed earlier with `pip install --user` live in ~/.local and take
# precedence over the conda env, which is how a 2022-era transformers ends up
# shadowing the one you just installed. This shuts that path off.
export PYTHONNOUSERSITE=1

# The cluster ships a conda module, so there is no reason to download our own.
# CONDA_ROOT is only the fallback if that module ever disappears.
CONDA_MODULE="${CONDA_MODULE:-apps/miniconda/24.7.1}"
CONDA_ROOT="${CONDA_ROOT:-${SCRATCH_DIR}/miniconda3}"
# Addressed by path, not by name: `conda create -n foo` would place the env in
# ~/.conda/envs and fill the home quota.
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${SCRATCH_DIR}/envs/local_llm}"

# Call this, not a bare `python`. In a non-interactive shell `conda activate`
# can return 0 while leaving PATH untouched, so `pip` silently resolves to the
# read-only system conda and the install dies on EACCES. An absolute
# interpreter path cannot be wrong.
ENV_PY="${CONDA_ENV_PREFIX}/bin/python"

# --- Campus proxy -------------------------------------------------------------
# Compute nodes have no route to the internet until the node authenticates
# against the firewall. Only downloads and pip need this; serving does not.
PROXY_HOST="${PROXY_HOST:-10.10.78.62}"
PROXY_PORT="${PROXY_PORT:-3128}"
PROXY_PORTAL="${PROXY_PORTAL:-https://proxy62.iitd.ernet.in/cgi-bin/proxy.cgi}"

proxy_on() {
  export http_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
  export https_proxy="http://${PROXY_HOST}:${PROXY_PORT}"
  export HTTP_PROXY="$http_proxy"
  export HTTPS_PROXY="$https_proxy"
  # Never proxy traffic that stays inside the cluster.
  export no_proxy="localhost,127.0.0.1,.iitd.ac.in,.iitd.ernet.in,10.0.0.0/8"
  export NO_PROXY="$no_proxy"
}

proxy_off() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
}

load_conda() {
  if module load "${CONDA_MODULE}" 2>/dev/null && command -v conda >/dev/null 2>&1; then
    # The module exports PYTHONPATH=<base>/lib/python3.12/site-packages, which
    # leaks into every interpreter afterwards — including this env's 3.10. pip
    # then finds base packages, tries to uninstall root-owned files, and dies
    # with EACCES. (It is also an ABI mismatch: 3.12 packages on a 3.10 path.)
    unset PYTHONPATH
    eval "$(conda shell.bash hook)"
  elif [[ -x "${CONDA_ROOT}/bin/conda" ]]; then
    eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
  else
    echo "No conda: neither module ${CONDA_MODULE} nor ${CONDA_ROOT}" >&2
    return 1
  fi
}

activate_env() {
  load_conda || return 1
  conda activate "${CONDA_ENV_PREFIX}"
}

# Enabled by default: downloads and pip both need it, and forgetting it makes
# huggingface_hub hang with no error rather than fail fast. Harmless when
# serving, because that path sets HF_HUB_OFFLINE and no_proxy covers localhost
# and the 10.x cluster network. Call proxy_off if you need it gone.
proxy_on

export HPC_USER HPC_DEPT HPC_PROGRAMME SCRATCH_DIR PROJECT_CODE
export MODEL_PROFILE MODEL_REPO MODEL_NAME MODEL_PATH SERVED_NAME SERVE_PORT
export CONDA_MODULE CONDA_ROOT CONDA_ENV_PREFIX ENV_PY
export PROXY_HOST PROXY_PORT PROXY_PORTAL
