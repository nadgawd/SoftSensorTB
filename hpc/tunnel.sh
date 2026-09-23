#!/usr/bin/env bash
# Run on your LAPTOP. Forwards localhost:8888 -> compute-node vLLM :8000.
#
#   COMPUTE_NODE=khas016 bash hpc/tunnel.sh
#
# The compute node has no public address, but the login node can reach it, so
# ssh does the second hop for you. Leave this terminal open for the session.
set -euo pipefail

# Uses the `iitd` alias from ~/.ssh/config when present, so an existing
# ControlMaster socket is reused and no password is requested.
SSH_TARGET="${SSH_TARGET:-iitd}"
ssh -G "${SSH_TARGET}" >/dev/null 2>&1 || SSH_TARGET="${HPC_USER:-ch7221493}@${HPC_HOST:-hpc.iitd.ac.in}"
COMPUTE_NODE="${COMPUTE_NODE:?Set COMPUTE_NODE — see 'Running on node' in hpc/logs/vllm_<profile>.out}"
LOCAL_PORT="${LOCAL_PORT:-8888}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

echo "Tunnel: localhost:${LOCAL_PORT} -> ${COMPUTE_NODE}:${REMOTE_PORT} via ${SSH_TARGET}"
echo "Backend should use LOCAL_LLM_BASE_URL=http://localhost:${LOCAL_PORT}/v1"
echo "Test from another terminal: curl http://localhost:${LOCAL_PORT}/v1/models"
echo ""

# ServerAlive keeps the hop from being reaped during a long idle spell between
# chat turns; ExitOnForwardFailure turns a silently-dead tunnel into an error.
exec ssh -N \
  -o ServerAliveInterval=60 \
  -o ExitOnForwardFailure=yes \
  -L "${LOCAL_PORT}:${COMPUTE_NODE}:${REMOTE_PORT}" \
  "${SSH_TARGET}"
