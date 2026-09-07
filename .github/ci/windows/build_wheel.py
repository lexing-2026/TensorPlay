#!/usr/bin/env python3
"""Build a wheel on Windows.

Expects the MSVC env (PATH/LIB/INCLUDE from vcvarsall), the MKL prefix
(CMAKE_PREFIX_PATH), and libuv_ROOT to already be configured by the
sibling vc_env_setup.py + build_install_deps.py, all sourced by the
parent bash wrapper. This script is the wheel-build step proper.

Usage: build_wheel.py <output_dir>
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
