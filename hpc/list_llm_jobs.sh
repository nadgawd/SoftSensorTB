#!/usr/bin/env bash
# Print "<job id> <job name>" for each of your queued or running vllm_* jobs.
# run.sh uses this to spot servers left over from another model profile.
set -euo pipefail

for job in $(qselect -u "${USER}" -s QR 2>/dev/null); do
  name="$(qstat -f "${job}" 2>/dev/null | awk '/Job_Name =/ {print $3; exit}')"
  case "${name}" in
    vllm_*) echo "${job} ${name}" ;;
  esac
done
