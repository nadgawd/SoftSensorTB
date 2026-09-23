#!/usr/bin/env bash
# Is the server up, and does it do tool calls?
#
#   On the login node:  bash hpc/check_server.sh khas016
#   Through the tunnel: bash hpc/check_server.sh localhost 8888
set -euo pipefail

NODE="${1:?usage: check_server.sh <node-or-localhost> [port]}"
PORT="${2:-8000}"
BASE="http://${NODE}:${PORT}/v1"

echo "== GET ${BASE}/models =="
curl -sf --noproxy "*" --max-time 10 "${BASE}/models" || { echo "unreachable"; exit 1; }
echo ""

# The alias here must match SERVED_NAME and LOCAL_LLM_*_MODEL in .env, and the
# tool round-trip is the part actually worth testing: a server that chats fine
# can still fail to emit tool calls if --tool-call-parser is wrong for the model.
MODEL="$(curl -sf --noproxy "*" "${BASE}/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "== tool-call round-trip against '${MODEL}' =="

curl -sf --noproxy "*" --max-time 60 "${BASE}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d @- <<EOF | python3 -m json.tool
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "What is the mean of column T_C?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "column_stats",
      "description": "Summary statistics for one column",
      "parameters": {
        "type": "object",
        "properties": {"column": {"type": "string"}},
        "required": ["column"]
      }
    }
  }],
  "tool_choice": "auto"
}
EOF

echo ""
echo "Expect tool_calls naming column_stats with column=T_C."
echo "Plain prose instead means the tool-call parser does not match the model."
