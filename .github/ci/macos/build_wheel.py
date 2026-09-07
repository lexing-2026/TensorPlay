#!/usr/bin/env python3
"""Build a wheel on macOS arm64.

Usage: build_wheel.py <output_dir>

Expects the build env (MACOSX_DEPLOYMENT_TARGET, OMP_PREFIX, ...) to be
set by the caller (build.sh sources build_env_setup.py's export file).
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
