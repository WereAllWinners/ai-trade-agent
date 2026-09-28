#!/usr/bin/env bash
# freeze_env.sh — snapshot the exact installed package versions (sprint02 D6b).
#
# requirements.txt only pins ranges; this captures what's ACTUALLY installed
# and tested right now, so a future "why did this break" investigation has a
# known-good baseline to diff against instead of re-deriving it from scratch
# (see D1 — pandas silently crossing a major version was exactly this kind
# of untracked drift).
#
# Usage:
#   scripts/tools/freeze_env.sh
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
_VENV_PIP="${_PROJECT_ROOT}/venv/bin/pip"
_LOCK_FILE="${_PROJECT_ROOT}/requirements.lock.txt"

if [[ ! -x "${_VENV_PIP}" ]]; then
    echo "venv pip not found at ${_VENV_PIP} — is the venv set up?" >&2
    exit 1
fi

"${_VENV_PIP}" freeze > "${_LOCK_FILE}"
echo "Wrote $(wc -l < "${_LOCK_FILE}") package versions to ${_LOCK_FILE}"
