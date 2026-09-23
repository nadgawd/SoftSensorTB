#!/usr/bin/env bash
# Authenticate THIS node against the campus firewall so pip/HuggingFace can
# reach the internet. Run it on whichever machine is about to download.
#
#   source hpc/env.sh && bash hpc/proxy_login.sh
#
# Two things about this portal that cost a lot of time if you don't know them:
#
#  1. It is a cookie-based web login. Earlier attempts with a bare
#     `curl -d ... URL` bounced in a 302 loop because curl discarded the session
#     cookie between the redirect hops. The fix is a cookie jar (-c/-b) plus -L,
#     which is what this script does.
#  2. Authorisation is granted per source IP, and only one session per node is
#     allowed. If another user already holds the session on your compute node
#     you cannot log in, and the only remedy is a different node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/env.sh"

COOKIE_JAR="$(mktemp -t proxysess.XXXXXX)"
trap 'rm -f "${COOKIE_JAR}"' EXIT

# The portal must be reached directly; going through the proxy to authenticate
# against the proxy is the "you can't login from a proxy server ip" error.
proxy_off

read -r -p "Kerberos ID [${HPC_USER}]: " kerb
kerb="${kerb:-${HPC_USER}}"
# -s keeps the password out of the terminal and out of ~/.bash_history. Passing
# it on a command line, as `curl -d "password=..."` would, leaks it to anyone
# who can read your history or run ps.
read -r -s -p "Kerberos password: " pass
echo ""

echo "Authenticating ${kerb} against ${PROXY_PORTAL} ..."
response="$(
  curl -k -s -L \
       -c "${COOKIE_JAR}" -b "${COOKIE_JAR}" \
       --data-urlencode "userid=${kerb}" \
       --data-urlencode "password=${pass}" \
       --data-urlencode "action=Submit" \
       "${PROXY_PORTAL}" || true
)"
unset pass

if grep -qiE "already logged in|execution aborted" <<<"${response}"; then
  echo "REFUSED: another session already holds this node's IP ($(hostname))." >&2
  echo "Exit this node and request a fresh one:" >&2
  echo "  qsub -I -P ${PROJECT_CODE} -q standard -l select=1:ncpus=8:ngpus=1 -l walltime=02:00:00" >&2
  exit 2
fi

proxy_on

# Don't trust the HTML; ask whether traffic actually leaves the cluster.
echo "Verifying egress ..."
if curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://huggingface.co | grep -q '^2'; then
  echo "OK — this node can reach the internet."
  echo "Proxy exported in THIS shell only. For another shell, source hpc/env.sh && proxy_on."
else
  echo "Login did not take. Fall back to the interactive browser:" >&2
  echo "  lynx ${PROXY_PORTAL}" >&2
  echo "  (press y at the SSL warning, fill the form, choose Login, then q y to quit)" >&2
  exit 1
fi
