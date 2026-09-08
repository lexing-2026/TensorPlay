#!/usr/bin/env python3
"""Install build-time dependencies for a Windows wheel build.

Installs the pinned pip packages (build backend requirements plus the
pinned MKL trio) and the prebuilt libuv tarball, then hands the resulting
environment back to the parent bash wrapper via --env-out.

Environment variables expected:
    SKIP_SETUP_CLEAN - reserved; no tree cleaning is performed today
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import download, write_env_exports


# Fixed build-time pip deps: the build backend requirements from
# pyproject.toml plus the pinned MKL trio. MKL wheels unpack into a
# conda-style Library/ tree under the interpreter prefix; the parent env
# setup points CMAKE_PREFIX_PATH at that tree so find_package(MKL) in
# config mode resolves it.
PIP_PACKAGES: list[str] = [
    "cmake<4.0",
    "ninja",
    "numpy",
    "pybind11",
    "pyyaml",
    "wheel",
    "build",
    "scikit-build-core>=1.0",
    "mkl==2024.2.0",
    "mkl-static==2024.2.0",
    "mkl-include==2024.2.0",
]


LIBUV_URL = "https://s3.amazonaws.com/ossci-windows/libuv-1.40.0-h8ffe710_0.tar.bz2"
# Mozilla's prebuilt release; the same channel the build ecosystems use.
SCCACHE_URL = "https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-pc-windows-msvc.zip"


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


def install_libuv(workdir: Path, python_prefix: Path) -> Path:
    """Curl + 7z + tar extract libuv into the running Python's prefix.

    gloo's CMake glue locates libuv through a find_library call rooted at
    libuv_ROOT, so the extraction must produce <prefix>/Library. Returns
    the extracted libuv root.
    """
    tarball_bz2 = workdir / "libuv-1.40.0-h8ffe710_0.tar.bz2"
    tarball = workdir / "libuv-1.40.0-h8ffe710_0.tar"
    download(LIBUV_URL, tarball_bz2)
    # 7z and tar are both present on Windows CI runners (7-Zip preinstalled,
    # tar ships with Windows 10+).
    subprocess.run(["7z", "x", "-aoa", str(tarball_bz2), f"-o{workdir}"], check=True)
    python_prefix.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(tarball), "-C", str(python_prefix)], check=True)
    libuv_root = python_prefix / "Library"
    if not libuv_root.is_dir():
        sys.exit(
            f"libuv extraction did not produce {libuv_root}; "
            "the tarball layout may have changed"
        )
    return libuv_root


def install_sccache(workdir: Path) -> dict[str, str]:
    """Unpack sccache into a private directory and put it on PATH.

    Returns the env diff for --env-out. CMake's USE_CCACHE block finds the
    binary through its sccache fallback and wires the compiler launchers;
    sccache itself keeps its cache in SCCACHE_DIR (pinned by the workflow).
    """
    sccache_dir = workdir / "sccache-bin"
    sccache_dir.mkdir(parents=True, exist_ok=True)
    archive = workdir / "sccache.zip"
    download(SCCACHE_URL, archive)
    # 7-Zip is preinstalled on the Windows CI runners.
    subprocess.run(
        ["7z", "x", "-aoa", str(archive), f"-o{sccache_dir}"], check=True
    )
    sccache_exe = sccache_dir / "sccache.exe"
    if not sccache_exe.is_file():
        sys.exit(f"sccache extraction did not produce {sccache_exe}")
    return {"PATH": f"{sccache_dir};{os.environ.get('PATH', '')}"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-out", type=Path)
    args = parser.parse_args()

    pip_install("-q", *PIP_PACKAGES)

    env_out: dict[str, str] = {}
    env_out.update(install_sccache(Path(__file__).resolve().parent))

    libuv_root = install_libuv(Path(__file__).resolve().parent, Path(sys.prefix))

    env_out["libuv_ROOT"] = str(libuv_root)
    write_env_exports(env_out, args.env_out)
    print(f"libuv_ROOT={libuv_root}")
    print("build_install_deps complete")


if __name__ == "__main__":
    main()
