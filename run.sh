#!/usr/bin/env bash
# Bring the whole stack up: GPU job on the cluster, SSH tunnel, backend, frontend.
#
#   ./run.sh                 everything
#   ./run.sh --no-cluster    laptop only; chat falls back to the cloud API keys
#   ./run.sh --no-frontend   backend only
#   ./run.sh --serve-llm     only serve the HPC model to the online app, through
#                            Tailscale Funnel; keeps the Mac awake and resubmits
#                            the job when its walltime ends
#   ./run.sh --stop          tear down local processes (leaves the PBS job up)
#
# Ctrl-C stops the local processes. The PBS job deliberately survives, because
# it costs four minutes to start and you usually want it across restarts; use
# --stop then `ssh iitd qdel <id>` if you really want the GPU back.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SSH_HOST="${SSH_HOST:-iitd}"
REMOTE_DIR="${REMOTE_DIR:-mtp}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
LOCAL_PORT="${LOCAL_PORT:-8888}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
VENV="${VENV:-.venv_run}"
LOGDIR="${ROOT}/.run-logs"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
WATCH_INTERVAL="${WATCH_INTERVAL:-60}"
FUNNEL_PORT="${FUNNEL_PORT:-443}"

# The model profile (see hpc/env.sh) is whatever .env tells the backend to call,
# so the job we start or reuse always serves exactly the model the backend wants.
env_value() { grep -m1 "^$1=" .env 2>/dev/null | cut -d= -f2- | tr -d '"'"'"' ' || true; }
PROFILE="${MODEL_PROFILE:-$(env_value LOCAL_LLM_EXECUTION_MODEL)}"
PROFILE="${PROFILE:-qwen3.5-27b}"
# One job name per profile: PBS job names must be simple, so strip punctuation.
JOB_NAME="${JOB_NAME:-vllm_${PROFILE//[^A-Za-z0-9]/}}"

WANT_CLUSTER=1
WANT_BACKEND=1
WANT_FRONTEND=1
SERVE_LLM=0
for arg in "$@"; do
  case "$arg" in
    --no-cluster)  WANT_CLUSTER=0 ;;
    --no-frontend) WANT_FRONTEND=0 ;;
    --serve-llm)   SERVE_LLM=1; WANT_BACKEND=0; WANT_FRONTEND=0 ;;
    --stop)        STOP_ONLY=1 ;;
    -h|--help)     sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done
(( SERVE_LLM && ! WANT_CLUSTER )) && { echo "--serve-llm needs the cluster" >&2; exit 2; }

mkdir -p "${LOGDIR}"
PIDS=()

c()    { printf "\033[1;36m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
die()  { printf "  \033[31m✗\033[0m %s\n" "$*" >&2; exit 1; }
now()  { date +%H:%M; }

# --- teardown ---------------------------------------------------------------

# With ControlMaster, a -L forward is owned by the *master* process, not by the
# client that requested it. Killing the client leaves the listener bound, so the
# next run finds the port busy and its own forward silently dies. `ssh -O cancel`
# is the only thing that actually releases it.
# A master-owned forward can only be cancelled by its exact spec, node
# included, and it outlives this script. Remember which node it points at: a
# new job often lands on a different node, and the old forward would otherwise
# hold the port and make the new one fail.
TUNNEL_NODE_FILE="${LOGDIR}/tunnel.node"
JOB_FILE="${LOGDIR}/llm.job"

drop_forward() {
  local nodes=("${1:-}")
  [[ -f "${TUNNEL_NODE_FILE}" ]] && nodes+=("$(cat "${TUNNEL_NODE_FILE}")")
  if [[ -z "${nodes[*]// /}" ]]; then
    # Node unknown (e.g. plain --stop): ask PBS.
    nodes+=("$(ssh -o BatchMode=yes "${SSH_HOST}" \
         "bash -lc 'qstat -f \$(qselect -N ${JOB_NAME} -s R 2>/dev/null | head -1) 2>/dev/null | grep -m1 exec_host'" 2>/dev/null \
         | awk '{print $3}' | cut -d/ -f1)")
  fi
  local n
  for n in "${nodes[@]}"; do
    [[ -n "${n}" ]] && ssh -O cancel -L "${LOCAL_PORT}:${n}:${REMOTE_PORT}" "${SSH_HOST}" 2>/dev/null || true
  done
  rm -f "${TUNNEL_NODE_FILE}"
}

# Tailscale's CLI is on PATH for the open-source build; the Mac app keeps it in
# the bundle.
tailscale_cli() {
  if command -v tailscale >/dev/null 2>&1; then
    tailscale "$@"
  elif [[ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]]; then
    /Applications/Tailscale.app/Contents/MacOS/Tailscale "$@"
  else
    return 127
  fi
}

# Only /v1 goes public. vLLM's API key guards /v1/* alone; its root routes
# (/invocations, /tokenize, /metrics, ...) answer anyone.
FUNNEL_PATH="/v1"
FUNNEL_ON=0
funnel_off() {
  (( FUNNEL_ON )) || return 0
  tailscale_cli funnel --https="${FUNNEL_PORT}" --set-path="${FUNNEL_PATH}" off >/dev/null 2>&1 \
    && ok "Tailscale Funnel off" \
    || warn "could not turn the Funnel off: tailscale funnel --https=${FUNNEL_PORT} --set-path=${FUNNEL_PATH} off"
  FUNNEL_ON=0
}

stop_local() {
  if (( WANT_CLUSTER )); then
    drop_forward "${NODE:-}"
    pkill -f "ssh -N .*${LOCAL_PORT}:.*:${REMOTE_PORT}" 2>/dev/null || true
  fi
  if (( WANT_BACKEND )); then
    pkill -f "uvicorn backend.main:app" 2>/dev/null || true
  fi
  if (( WANT_FRONTEND )); then
    pkill -f "vite.*${ROOT}/frontend" 2>/dev/null || true
    # npm spawns vite as a child; matching on the directory alone can miss it.
    pkill -f "node .*${ROOT}/frontend" 2>/dev/null || true
  fi
}

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  echo ""
  c "Shutting down"
  funnel_off
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  stop_local
  ok "local processes stopped (PBS job left running)"
  exit "${rc}"
}

if [[ -n "${STOP_ONLY:-}" ]]; then
  c "Stopping"
  WANT_CLUSTER=1 WANT_BACKEND=1 WANT_FRONTEND=1
  stop_local
  ok "done"
  exit 0
fi

# --- cluster helpers --------------------------------------------------------

# The vLLM API key lives on the cluster (hpc/.llm_api_key, created once by
# hpc/make_llm_key.sh) and is copied into .env for the backend. It is only
# ever held in variables and passed on stdin, never on a command line.
LLM_KEY=""
ensure_llm_key() {
  ssh "${SSH_HOST}" "bash ${REMOTE_DIR}/hpc/make_llm_key.sh" >/dev/null \
    || die "could not create the vLLM API key on the cluster"
  LLM_KEY="$(ssh "${SSH_HOST}" "cat ${REMOTE_DIR}/hpc/.llm_api_key")"
  [[ -n "${LLM_KEY}" ]] || die "the vLLM API key on the cluster is empty"
  if [[ "$(env_value LOCAL_LLM_API_KEY)" != "${LLM_KEY}" ]]; then
    LLM_KEY="${LLM_KEY}" "${VENV}/bin/python" - <<'PY'
import os
import re
from pathlib import Path

env = Path(".env")
line = "LOCAL_LLM_API_KEY=" + os.environ["LLM_KEY"]
text = env.read_text()
if re.search(r"^LOCAL_LLM_API_KEY=.*$", text, flags=re.M):
    text = re.sub(r"^LOCAL_LLM_API_KEY=.*$", lambda _m: line, text, count=1, flags=re.M)
else:
    text = text.rstrip("\n") + "\n" + line + "\n"
env.write_text(text)
PY
    ok "vLLM API key copied into .env (LOCAL_LLM_API_KEY)"
  fi
}

# GET on the forwarded vLLM server with the key; prints the body.
llm_get() {
  printf 'Authorization: Bearer %s\n' "${LLM_KEY:-EMPTY}" \
    | curl -sf --noproxy '*' --max-time "${2:-5}" -H @- "http://localhost:${LOCAL_PORT}$1"
}

job_state() {
  ssh -o BatchMode=yes "${SSH_HOST}" "bash -lc 'qstat -f $1 2>/dev/null | grep -m1 job_state'" 2>/dev/null \
    | awk '{print $3}'
}

connect_cluster() {
  # A ControlMaster socket lets every later ssh/rsync reuse one authenticated
  # connection. If it is gone this prompts once, interactively, and the
  # password never passes through the script.
  if ssh -O check "${SSH_HOST}" >/dev/null 2>&1; then
    ok "reusing SSH connection"
  else
    warn "no shared SSH connection — authenticating (you will be prompted)"
    ssh -fN "${SSH_HOST}" || die "ssh to ${SSH_HOST} failed"
    ok "connected"
  fi

  rsync -a --exclude '.DS_Store' --exclude '__pycache__' --exclude '.llm_api_key' --exclude 'logs/' \
        hpc/ "${SSH_HOST}:${REMOTE_DIR}/hpc/" 2>/dev/null \
    && ok "scripts synced" || warn "rsync failed; using whatever is on the cluster"
  ensure_llm_key
}

# Submit or reuse the job, wait for a node, forward the port, wait for vLLM.
# Sets JOB and NODE, and records them for the watchdog.
start_job() {
  JOB="$(ssh "${SSH_HOST}" "bash -lc 'qselect -N ${JOB_NAME} -s QR 2>/dev/null | head -1'" | tr -d '[:space:]')"
  if [[ -n "${JOB}" ]]; then
    ok "reusing job ${JOB} (${PROFILE})"
  else
    JOB="$(ssh "${SSH_HOST}" "bash -lc 'cd ${REMOTE_DIR} && rm -f hpc/logs/${JOB_NAME}.* && qsub -N ${JOB_NAME} -o hpc/logs/${JOB_NAME}.out -e hpc/logs/${JOB_NAME}.err -v MODEL_PROFILE=${PROFILE} hpc/serve_llm.pbs'" | tr -d '[:space:]')"
    [[ -n "${JOB}" ]] || die "qsub produced no job id"
    ok "submitted ${JOB} (${PROFILE})"
  fi
  echo "${JOB}" > "${JOB_FILE}"

  # A server for a different profile (or from before profiles existed) would
  # hold an A100 for nothing. Point it out rather than killing someone's job.
  while read -r other_id other_name; do
    [[ -n "${other_id}" && "${other_name}" != "${JOB_NAME}" ]] || continue
    warn "another LLM job holds a GPU: ${other_name} (${other_id}) — free it with: ssh ${SSH_HOST} qdel ${other_id}"
  done < <(ssh "${SSH_HOST}" "bash -lc 'bash ${REMOTE_DIR}/hpc/list_llm_jobs.sh'" 2>/dev/null || true)

  # Queued -> Running can take a while when the A100 nodes are busy.
  printf "  waiting for the scheduler"
  local state=""
  for _ in $(seq 60); do
    state="$(job_state "${JOB}")"
    [[ "${state}" == "R" ]] && break
    printf "."; sleep 10
  done
  echo ""
  [[ "${state}" == "R" ]] || die "job ${JOB} still not running; check: ssh ${SSH_HOST} qstat -u \$USER"

  # exec_host is authoritative and available immediately. The job log also
  # prints the node, but PBS buffers job output, so it can lag by minutes.
  NODE="$(ssh "${SSH_HOST}" "bash -lc 'qstat -f ${JOB} 2>/dev/null | grep -m1 exec_host'" \
          | awk '{print $3}' | cut -d/ -f1)"
  [[ -n "${NODE}" ]] || die "could not determine compute node for ${JOB}"
  ok "running on ${NODE}"

  forward_port
  wait_for_vllm
}

forward_port() {
  # Hand the forward to the existing master: no extra process to supervise, and
  # it survives this script exiting. Any stale forward on the port must go
  # first, or the new one is refused.
  drop_forward "${NODE}"
  if ssh -O forward -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "${SSH_HOST}" 2>/dev/null; then
    echo "${NODE}" > "${TUNNEL_NODE_FILE}"
    ok "localhost:${LOCAL_PORT} -> ${NODE}:${REMOTE_PORT} (via shared connection)"
  else
    # No master (e.g. SSH_HOST has no ControlMaster config) — run our own.
    ssh -N -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes \
        -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "${SSH_HOST}" \
        >"${LOGDIR}/tunnel.log" 2>&1 &
    PIDS+=($!)
    sleep 2
    kill -0 "${PIDS[-1]}" 2>/dev/null \
      && ok "localhost:${LOCAL_PORT} -> ${NODE}:${REMOTE_PORT}" \
      || die "tunnel failed — see ${LOGDIR}/tunnel.log"
  fi
}

wait_for_vllm() {
  # vLLM is silent until it has loaded the weights (15–29 GB depending on the
  # profile) and sized the KV cache; several minutes. Connection-refused here is
  # normal, not an error.
  printf "  loading model"
  local deadline=$(( SECONDS + READY_TIMEOUT ))
  until llm_get /v1/models >/dev/null 2>&1; do
    (( SECONDS < deadline )) || { echo ""; die "not ready after ${READY_TIMEOUT}s — see: ssh ${SSH_HOST} tail ${REMOTE_DIR}/hpc/logs/${JOB_NAME}.out"; }
    printf "."; sleep 10
  done
  echo ""
  local served
  served="$(llm_get /v1/models | "${VENV}/bin/python" -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo '?')"
  ok "serving '${served}'"

  # A served name that disagrees with .env yields a 404 per turn and a silent
  # fallback to the cloud, which looks like "the cluster is being ignored".
  local want
  want="$(env_value LOCAL_LLM_EXECUTION_MODEL)"
  if [[ -n "${want}" && "${want}" != "${served}" ]]; then
    warn ".env expects '${want}' but the server offers '${served}' — every call will 404"
  fi
}

# 401 without the key means the job was started with VLLM_API_KEY set.
llm_requires_key() {
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "http://localhost:${LOCAL_PORT}/v1/models" || true)"
  [[ "${code}" == "401" ]]
}

# --- preflight --------------------------------------------------------------

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

c "Preflight"
[[ -x "${VENV}/bin/python" ]] || die "no interpreter at ${VENV}/bin/python"
[[ -f .env ]] || die ".env missing — copy .env.example and fill it in"
ok "venv and .env present"

if (( SERVE_LLM )); then
  tailscale_cli version >/dev/null 2>&1 \
    || die "Tailscale is not installed — see DEPLOY.md, section 'Tailscale Funnel'"
  tailscale_cli status >/dev/null 2>&1 || die "Tailscale is not logged in — open the Tailscale app and sign in"
  ok "Tailscale ready"
fi

if (( WANT_BACKEND )) && port_busy "${BACKEND_PORT}"; then
  warn "port ${BACKEND_PORT} busy — clearing a previous run"
  pkill -f "uvicorn backend.main:app" 2>/dev/null || true
  sleep 2
  port_busy "${BACKEND_PORT}" && die "port ${BACKEND_PORT} still held by something else"
fi
ok "ports clear"

# Installed after preflight, so a failed check leaves a running session alone.
trap cleanup EXIT
# Without an explicit exit, Ctrl-C would run cleanup and then resume the watchdog.
trap 'exit 130' INT TERM

# --- cluster ----------------------------------------------------------------

NODE=""
JOB=""
if (( WANT_CLUSTER )); then
  c "Cluster"
  connect_cluster
  start_job
else
  c "Cluster skipped — chat will use the cloud keys in .env"
fi

# --- serve the LLM to the online app ----------------------------------------

if (( SERVE_LLM )); then
  c "Funnel"
  # Public means anyone can reach it: refuse a server that answers without the key.
  llm_requires_key || die "job ${JOB} predates the API key and answers anyone. Free it with: ssh ${SSH_HOST} qdel ${JOB}  then rerun"
  ok "vLLM requires the API key"

  tailscale_cli funnel --bg --yes --https="${FUNNEL_PORT}" --set-path="${FUNNEL_PATH}" \
      "http://127.0.0.1:${LOCAL_PORT}${FUNNEL_PATH}" >"${LOGDIR}/funnel.log" 2>&1 \
    || die "tailscale funnel failed — see ${LOGDIR}/funnel.log (Funnel must be enabled for this device, see DEPLOY.md)"
  FUNNEL_ON=1
  DNS_NAME="$(tailscale_cli status --json | "${VENV}/bin/python" -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))")"
  suffix=""
  [[ "${FUNNEL_PORT}" != "443" ]] && suffix=":${FUNNEL_PORT}"
  PUBLIC_URL="https://${DNS_NAME}${suffix}${FUNNEL_PATH}"
  ok "public: ${PUBLIC_URL}"

  # -d display, -i idle, -s system sleep (on AC power); -w ends it with this script.
  caffeinate -dis -w $$ &
  PIDS+=($!)
  ok "keeping the Mac awake"

  echo ""
  c "Serving"
  echo "  HF Space secrets: LOCAL_LLM_BASE_URL=${PUBLIC_URL}"
  echo "                    LOCAL_LLM_API_KEY = the LOCAL_LLM_API_KEY line in .env"
  echo "  GPU               ${NODE} (job ${JOB})"
  echo "  Ctrl-C            turns the Funnel off; the PBS job keeps running"
  echo ""

  # Watchdog. When the walltime ends the job disappears: resubmit, wait for it,
  # re-forward. The online app answers from the cloud fallback meanwhile.
  while true; do
    sleep "${WATCH_INTERVAL}"
    if ! ssh -O check "${SSH_HOST}" >/dev/null 2>&1; then
      warn "$(now) SSH connection to ${SSH_HOST} lost — sign in again in another terminal: ssh ${SSH_HOST}"
      continue
    fi
    state="$(job_state "${JOB}")"
    if [[ "${state}" != "R" && "${state}" != "Q" ]]; then
      warn "$(now) job ${JOB} ended (walltime) — starting a new one; the cloud fallback answers meanwhile"
      # Subshell: a failed attempt must not end the watchdog; the next tick retries.
      if ( start_job ); then
        JOB="$(cat "${JOB_FILE}")"
        NODE="$(cat "${TUNNEL_NODE_FILE}" 2>/dev/null || echo "${NODE}")"
        ok "$(now) serving again from ${NODE} (job ${JOB})"
      else
        warn "$(now) restart failed; retrying in ${WATCH_INTERVAL}s"
      fi
    elif ! llm_get /v1/models >/dev/null 2>&1; then
      warn "$(now) job ${JOB} is up but the tunnel is not answering — re-forwarding"
      ( forward_port ) || warn "$(now) re-forward failed; retrying in ${WATCH_INTERVAL}s"
    fi
  done
fi

# --- backend ----------------------------------------------------------------

c "Backend"
"${VENV}/bin/uvicorn" backend.main:app --reload --port "${BACKEND_PORT}" \
  >"${LOGDIR}/backend.log" 2>&1 &
PIDS+=($!)

printf "  starting"
for _ in $(seq 40); do
  curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1 && break
  printf "."; sleep 1
done
echo ""
curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:${BACKEND_PORT}/health" >/dev/null 2>&1 \
  && ok "http://localhost:${BACKEND_PORT}" \
  || die "backend did not come up — see ${LOGDIR}/backend.log"

# --- frontend ---------------------------------------------------------------

if (( WANT_FRONTEND )); then
  c "Frontend"
  [[ -d frontend/node_modules ]] || { warn "installing dependencies"; (cd frontend && npm install); }
  (cd frontend && npm run dev >"${LOGDIR}/frontend.log" 2>&1) &
  PIDS+=($!)

  printf "  starting"
  url=""
  for _ in $(seq 40); do
    url="$(grep -oE 'http://localhost:[0-9]+' "${LOGDIR}/frontend.log" 2>/dev/null | head -1 || true)"
    [[ -n "${url}" ]] && break
    printf "."; sleep 1
  done
  echo ""
  ok "${url:-http://localhost:5173}"
fi

echo ""
c "Ready"
[[ -n "${NODE}" ]] && echo "  GPU       ${NODE} (job ${JOB})"
echo "  Logs      ${LOGDIR}/"
echo "  Ctrl-C    stops local processes; the PBS job keeps running"
echo ""

wait
