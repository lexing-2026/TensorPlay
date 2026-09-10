#!/usr/bin/env bash
# Linux wheel build orchestrator. Owns the stage contract: the Python
# modules (build_env_setup.py / build_install_deps.py / build_wheel.py)
# are non-orchestrating stages and hand env back through export files.
#
# Expects the desired interpreter already on PATH (the workflow's
# setup-python step provides it).

set -eux -o pipefail

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# .github/ci/<platform> -> repository root
REPO_ROOT="$(cd "${SCRIPTPATH}/../../.." && pwd)"

ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT

python3 "${SCRIPTPATH}/build_env_setup.py" --env-out "$ENV_FILE"
# shellcheck source=/dev/null
source "$ENV_FILE"

python3 "${SCRIPTPATH}/build_install_deps.py" "${REPO_ROOT}"

cd "${REPO_ROOT}"
# Build into a raw directory first. CPU wheels are then repacked through
# auditwheel: it bundles the libraries the manylinux policy requires to
# capture (libgomp among them), rewrites their RPATHs, and retags the
# wheel from the bare linux_* that PyPI rejects to the manylinux_<glibc>
# the builder qualifies for. CUDA wheels keep the bare tag: they carry a
# dependency on the toolkit libraries by design and ship through the
# release assets and the variant wheel indexes instead of PyPI.
RAW_WHEEL_DIR="$(mktemp -d)"
python3 "${SCRIPTPATH}/build_wheel.py" "${RAW_WHEEL_DIR}"
mkdir -p dist
for raw_wheel in "${RAW_WHEEL_DIR}"/*.whl; do
  case "${raw_wheel}" in
    *+cu*.whl) cp "${raw_wheel}" dist/ ;;
    *) auditwheel repair -w dist "${raw_wheel}" ;;
  esac
done
rm -rf "${RAW_WHEEL_DIR}"
