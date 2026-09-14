#!/usr/bin/env python3
"""Install build-time dependencies for a Linux wheel build.

Usage: build_install_deps.py <package_dir>

Installs the build backend requirements (needed by the --no-isolation
wheel build) plus the runtime package set for the aarch64 lane, which
links the system OpenBLAS instead of the x86_64 MKL staging.
"""

import argparse
import importlib
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
    # The PyPI index rejects bare linux_* platform tags, so every Linux
    # wheel is repacked through auditwheel (with patchelf rewriting the
    # captured libraries' RPATHs) into a manylinux-tagged one before it
    # leaves the builder.
    "auditwheel",
    "patchelf",
]

CUDA_RUNTIME_PACKAGES = {
    "12": ("nvidia-cudnn-cu12", "nvidia-nccl-cu12"),
    "13": ("nvidia-cudnn-cu13==9.20.0.48", "nvidia-nccl-cu13==2.29.7"),
}


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


def toolkit_major() -> str:
    toolkit = os.environ.get("CUDA_PATH", "")
    leaf = Path(toolkit).name
    if leaf.startswith("cuda-"):
        leaf = leaf[5:]
    major = leaf.split(".", 1)[0]
    if not major.isdigit():
        sys.exit(f"cannot determine CUDA major version from CUDA_PATH={toolkit!r}")
    return major


def package_root(module_name: str) -> Path:
    module = importlib.import_module(module_name)
    roots = list(module.__path__)
    if len(roots) != 1:
        sys.exit(f"expected one install root for {module_name}, got {roots}")
    return Path(roots[0])


def install_cuda_runtime() -> None:
    if os.environ.get("GPU_ARCH_TYPE", "cpu") != "cuda":
        return
    packages = CUDA_RUNTIME_PACKAGES.get(toolkit_major())
    if packages is None:
        sys.exit(f"no CUDA runtime package set for CUDA major {toolkit_major()}")
    pip_install("-q", *packages)

    cudnn_root = package_root("nvidia.cudnn")
    nccl_root = package_root("nvidia.nccl")
    required = [cudnn_root / "include" / "cudnn.h",
                nccl_root / "include" / "nccl.h"]
    cudnn_libs = list((cudnn_root / "lib").glob("libcudnn.so*"))
    nccl_libs = list((nccl_root / "lib").glob("libnccl.so*"))
    if not cudnn_libs:
        required.append(cudnn_root / "lib/libcudnn.so*")
    if not nccl_libs:
        required.append(nccl_root / "lib/libnccl.so*")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        sys.exit(f"CUDA runtime installation is incomplete: {missing}")
    print(f"cuDNN root: {cudnn_root}")
    print(f"NCCL root: {nccl_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package_dir", type=Path)
    args = parser.parse_args()

    os.chdir(args.package_dir)
    pip_install("-q", *BUILD_PACKAGES)
    install_cuda_runtime()

    # The CMake build wires sccache in as the compiler launcher on every
    # lane. ccache only supports the nvcc driver experimentally and every
    # .cu request misses, so a mixed fleet would make lane behavior depend
    # on the launcher; removing ccache keeps the selection deterministic
    # and lets every lane share the same object-store cache.
    apt_prefix = [] if os.geteuid() == 0 else ["sudo"]
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
    sccache_bin: Path | None
    found = shutil.which("sccache")
    if found is not None:
        sccache_bin = Path(found)
    else:
        # Root-less builders cannot write system prefixes, so stage into a
        # user-owned bin directory that is already on the runner's PATH.
        install_dir = (
            Path("/usr/local/bin")
            if os.geteuid() == 0
            else Path.home() / ".local" / "bin"
        )
        install_dir.mkdir(parents=True, exist_ok=True)
        sccache_bin = install_dir / "sccache"
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
            retry(["install", "-m", "0755", str(payload), str(sccache_bin)])
            shutil.rmtree(workdir, ignore_errors=True)

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
