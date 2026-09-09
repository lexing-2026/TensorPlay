#!/usr/bin/env python3
"""macOS arm64 build environment setup.

macOS needs far less toolchain wiring than the other platforms: there is
no CUDA lane and the heavy deps come from the runner image or Homebrew.
This script emits the macOS build flags to the --env-out file so the
caller (build.sh) can source them into the wheel build subprocess.
Without that handoff the exports made here die with this process.

Environment variables written (to --env-out):
    OMP_PREFIX - if /opt/llvm-openmp exists it is exported so the build
                 links the conda-forge libomp (supports older macOS than
                 the Homebrew build). See install_libomp.sh.
    (plus the static macOS build flags in MACOS_BUILD_ENV)
"""

import argparse
from pathlib import Path


# macOS arm64 build flags. oneDNN is off on Apple silicon; the deployment
# target matches the oldest runner the matrix builds on.
MACOS_BUILD_ENV: dict[str, str] = {
    "TENSORPLAY_BINARY_BUILD": "1",
    "USE_CUDA": "0",
    "MACOSX_DEPLOYMENT_TARGET": "14.0",
    "USE_TP_DISTRIBUTED": "0",
    "USE_MKLDNN": "OFF",
}

OMP_PREFIX = Path("/opt/llvm-openmp")


def shell_export_lines(env: dict[str, str]) -> list[str]:
    lines = []
    for k, v in env.items():
        if v and not all(c.isalnum() or c in "_-./:=," for c in v):
            v = "'" + v.replace("'", "'\\''") + "'"
        lines.append(f"export {k}={v}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-out",
        type=Path,
        help="Write `export KEY=VALUE` lines here for build.sh to source.",
    )
    args = parser.parse_args()

    env_out = dict(MACOS_BUILD_ENV)
    # Prefer the conda-forge libomp staged at /opt/llvm-openmp; otherwise
    # the build falls back to the Homebrew libomp installed by
    # build_install_deps.py.
    if OMP_PREFIX.is_dir():
        env_out["OMP_PREFIX"] = str(OMP_PREFIX)

    if args.env_out is not None:
        args.env_out.write_text("\n".join(shell_export_lines(env_out)) + "\n")
    print("macOS build environment configured")


if __name__ == "__main__":
    main()
