#!/usr/bin/env bash
# Windows wheel build orchestrator: chains the three phases and hands the
# environment between them via export files, because bash and the Python
# helpers do not share environment state.
#
#   1. build_env_setup.py   - lane env + MKL prefix, then capture the MSVC env
#   2. build_install_deps.py - pip build deps + pinned MKL trio + libuv
#   3. build_wheel.py       - python -m build --wheel --no-isolation
#
# Sourcing order matters: the vcvarsall capture runs last inside its own
# script so its PATH/LIB extensions stack on top of the pip-provisioned
# pieces.
#
# Lane selection: GPU_ARCH_TYPE=cpu (default) or cuda. The CUDA lane
# requires CUDA_PATH (from the toolkit install step) to contain nvcc.exe.
# The interpreter itself comes from the caller (actions/setup-python).
#
# The output directory defaults to dist/ in the repo root and can be
# overridden with the first argument.

set -eux -o pipefail

CI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CI_DIR/../.." && pwd)"

export GPU_ARCH_TYPE="${GPU_ARCH_TYPE:-cpu}"

ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT

python "$CI_DIR/build_env_setup.py" --env-out "$ENV_FILE"
# shellcheck source=/dev/null
source "$ENV_FILE"

python "$CI_DIR/build_install_deps.py" --env-out "$ENV_FILE"
# shellcheck source=/dev/null
source "$ENV_FILE"

cd "$REPO_ROOT"
python "$CI_DIR/build_wheel.py" "${1:-dist}"
