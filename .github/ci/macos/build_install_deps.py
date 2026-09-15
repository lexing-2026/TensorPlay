#!/usr/bin/env python3
"""Install build-time dependencies for the macOS arm64 wheel build.

Usage: build_install_deps.py <package_dir>

Installs the build backend requirements and, when the conda-forge libomp
is not staged at /opt/llvm-openmp, falls back to Homebrew's libomp.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
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

    # The CMake build picks up sccache as the compiler launcher when it is
    # on PATH; a shared object-store cache makes the second build of a
    # lane far cheaper. Homebrew's sccache floats with the runner image
    # and ships a much newer release whose object-store client wedges
    # against the pinned endpoint, so stage the exact release the other
    # platforms use and let build.sh put it first on PATH.
    machine = platform.machine()
    sccache_arch = {
        "arm64": "aarch64-apple-darwin",
        "x86_64": "x86_64-apple-darwin",
    }.get(machine)
    if sccache_arch is None:
        sys.exit(f"no sccache tarball mapping for {machine}")
    version = "0.18.0"
    install_dir = (
        Path(args.package_dir) / ".github" / "ci" / "macos" / "sccache-bin"
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    sccache_bin = install_dir / "sccache"
    if not sccache_bin.exists():
        workdir = Path("sccache-extract")
        workdir.mkdir(exist_ok=True)
        tarball = workdir / "sccache.tar.gz"
        url = (
            "https://github.com/mozilla/sccache/releases/download/"
            f"v{version}/sccache-v{version}-{sccache_arch}.tar.gz"
        )
        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball) as archive:
            archive.extractall(workdir)
        payload = workdir / f"sccache-v{version}-{sccache_arch}" / "sccache"
        if not payload.is_file():
            sys.exit(f"sccache extraction did not produce {payload}")
        sccache_bin.write_bytes(payload.read_bytes())
        sccache_bin.chmod(0o755)
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"sccache pinned to v{version} at {sccache_bin}")

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
