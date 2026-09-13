#!/usr/bin/env python3
"""Build a wheel on Linux.

Usage: build_wheel.py <output_dir>

Expects the build env (USE_CUDA, MKLROOT, OPENBLAS_ROOT_DIR, ...) to be
set by the caller (build.sh sources build_env_setup.py's export file).
This script only adds the BLAS plumbing that depends on the host
architecture: on x86_64 the MKL headers/libs staged under /opt/intel are
handed to CMake through CMAKE_{INCLUDE,LIBRARY}_PATH.
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def configure_blas_env() -> None:
    """Tell CMake which BLAS to use, based on architecture."""
    arch = platform.machine()
    if arch == "x86_64":
        mkl_root = Path(os.environ.get("TP_MKL_ROOT", "/opt/intel"))
        if (mkl_root / "include").is_dir():
            os.environ["CMAKE_INCLUDE_PATH"] = f"{mkl_root}/include"
            os.environ["CMAKE_LIBRARY_PATH"] = f"{mkl_root}/lib:/lib"
        return
    if arch == "aarch64":
        # The system OpenBLAS resolves through OPENBLAS_ROOT_DIR (exported
        # by build_env_setup.py); nothing further to wire here.
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configure_blas_env()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(args.output_dir),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
