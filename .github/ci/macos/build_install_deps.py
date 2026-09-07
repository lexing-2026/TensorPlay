#!/usr/bin/env python3
"""Install build-time dependencies for the macOS arm64 wheel build.

Usage: build_install_deps.py <package_dir>

Installs the build backend requirements and, when the conda-forge libomp
is not staged at /opt/llvm-openmp, falls back to Homebrew's libomp.
"""

import argparse
import os
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

OMP_PREFIX = Path("/opt/llvm-openmp")


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

    # OpenMP: prefer the conda-forge libomp staged at /opt/llvm-openmp (set
    # up by install_libomp.sh as a separate step). Otherwise fall back to
    # Homebrew, which only supports the build machine's macOS version or
    # higher.
    if not OMP_PREFIX.is_dir():
        if shutil.which("brew") is None:
            sys.exit("libomp not staged at /opt/llvm-openmp and brew not available")
        print("libomp not found at /opt/llvm-openmp, installing via brew")
        retry(["brew", "install", "libomp"])

    print("build_install_deps complete")


if __name__ == "__main__":
    main()
