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

    # The CMake build wires a compiler launcher through its cache-tool
    # fallback (ccache preferred, sccache otherwise). ccache only supports
    # the nvcc driver experimentally and every .cu request misses, so CUDA
    # lanes would recompile the whole module set on each run; sccache caches
    # the nvcc pipeline natively. CUDA lanes therefore run sccache alone --
    # removing ccache keeps the launcher selection deterministic -- while
    # CPU lanes stay on ccache, which serves their C++ requests well.
    apt_prefix = [] if os.geteuid() == 0 else ["sudo"]
    if os.environ.get("GPU_ARCH_TYPE") == "cuda":
        if shutil.which("ccache") is not None:
            subprocess.run(
                apt_prefix + ["apt-get", "remove", "-y", "-qq", "ccache"],
                check=False,
            )
        if shutil.which("ccache") is not None:
            sys.exit("ccache is still on PATH; the launcher choice would be ambiguous")

        machine = platform.machine()
        sccache_arch = {
            "x86_64": "x86_64-unknown-linux-musl",
            "aarch64": "aarch64-unknown-linux-musl",
        }.get(machine)
        if sccache_arch is None:
            sys.exit(f"no sccache tarball mapping for {machine}")
        sccache_bin = Path("/usr/local/bin/sccache")
        if not sccache_bin.exists():
            version = "0.8.1"
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
            payload = (
                workdir / f"sccache-v{version}-{sccache_arch}" / "sccache"
            )
            if not payload.is_file():
                sys.exit(f"sccache extraction did not produce {payload}")
            retry(apt_prefix + ["install", "-m", "0755", str(payload), str(sccache_bin)])
            shutil.rmtree(workdir, ignore_errors=True)
    elif shutil.which("ccache") is None:
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
