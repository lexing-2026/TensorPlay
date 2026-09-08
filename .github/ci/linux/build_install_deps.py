#!/usr/bin/env python3
"""Install build-time dependencies for a Linux wheel build.

Usage: build_install_deps.py <package_dir>

Installs the build backend requirements (needed by the --no-isolation
wheel build) plus the runtime package set for the aarch64 lane, which
links the system OpenBLAS instead of the x86_64 MKL staging.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


BUILD_PACKAGES: list[str] = [
    "cmake<4.0",
    "ninja",
    "numpy",
    "pybind11",
    "pyyaml",
    "wheel",
    "build",
    "scikit-build-core>=1.0",
]


def retry(cmd: list[str], delays: tuple[int, ...] = (1, 2, 4, 8)) -> None:
    """Run cmd, retrying with backoff on failure."""
    last_rc = 0
    for delay in (0, *delays):
        if delay:
            time.sleep(delay)
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        last_rc = result.returncode
    sys.exit(last_rc)


def pip_install(*args: str) -> None:
    retry([sys.executable, "-m", "pip", "install", *args])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    pip_install("-q", *BUILD_PACKAGES)

    # The CMake build picks up ccache as a compiler launcher when it is on
    # PATH; a persistent cache makes the second build of a lane far cheaper.
    if shutil.which("ccache") is None:
        apt_prefix = [] if os.geteuid() == 0 else ["sudo"]
        retry(apt_prefix + ["apt-get", "update", "-qq"])
        retry(apt_prefix + ["apt-get", "install", "-y", "-qq", "ccache"])

    if platform.machine() == "aarch64":
        # Redirection is shell syntax and must not leak into argv: an arg
        # like "1>/dev/null" reaches apt-get as a package name.
        apt_prefix = [] if os.geteuid() == 0 else ["sudo"]
        retry(apt_prefix + ["apt-get", "update", "-qq"])
        retry(apt_prefix + ["apt-get", "install", "-y", "-qq", "libopenblas-dev"])
        print("aarch64: system OpenBLAS installed")

    print("build_install_deps complete")


if __name__ == "__main__":
    main()
