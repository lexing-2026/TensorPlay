#!/usr/bin/env bash
# Linux wheel build orchestrator. Owns the stage contract: the Python
# modules (build_env_setup.py / build_install_deps.py / build_wheel.py)
# are non-orchestrating stages and hand env back through export files.
#
# Expects the desired interpreter already on PATH (the workflow's
# setup-python step provides it).

set -eux -o pipefail

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_ROOT="$(cd "${SCRIPTPATH}/../.." && pwd)"

ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT

python3 "${SCRIPTPATH}/build_env_setup.py" --env-out "$ENV_FILE"
# shellcheck source=/dev/null
source "$ENV_FILE"

python3 "${SCRIPTPATH}/build_install_deps.py" "${REPO_ROOT}"

cd "${REPO_ROOT}"
python3 "${SCRIPTPATH}/build_wheel.py" dist
