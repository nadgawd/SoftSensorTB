#!/bin/bash
# Create the vLLM API key once, on the cluster. Never prints it.
#
#   bash ~/mtp/hpc/make_llm_key.sh      # run.sh calls this for you
#
# serve_llm.pbs exports the key as VLLM_API_KEY, so every request to the server
# must carry it. That matters once ./run.sh --serve-llm exposes the server to the
# internet through Tailscale Funnel. To rotate: delete the file, then resubmit.
set -euo pipefail
cd "$(dirname "$0")/.."

KEY_FILE="hpc/.llm_api_key"
if [[ -s "${KEY_FILE}" ]]; then
  chmod 600 "${KEY_FILE}"
  echo "exists"
  exit 0
fi

umask 077
python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "${KEY_FILE}.tmp"
mv "${KEY_FILE}.tmp" "${KEY_FILE}"
chmod 600 "${KEY_FILE}"
echo "created"
