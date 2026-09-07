#!/usr/bin/env python3
"""Build environment setup for Linux wheel builds.

Emits the build-flag exports to the file given by --env-out for the
parent bash wrapper to source. Without that handoff the exports made here
die with this process.

BLAS wiring follows what the CMake detector consumes:
  * x86_64: MKL static libs + headers staged under /opt/intel (MKLROOT);
    the module-mode find_package(MKL) assembles the link line from that
    layout.
  * aarch64: system OpenBLAS installed by build_install_deps.py; the
    OpenBLAS root is exported so the detector prefers it over a vendored
    fallback.

The CUDA lane relies on the toolkit the workflow installs (PATH); no
GPU-dependent extras are configured here -- free hosted runners have no
GPU, so kernels are never executed.

Environment variables expected:
    GPU_ARCH_TYPE - cpu (default) or cuda
    CUDA_PATH     - CUDA toolkit root (CUDA lanes only)
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


MKL_VERSION = "2024.2.0"
MKL_ROOT = Path("/opt/intel")


CPU_BUILD_ENV: dict[str, str] = {
    "USE_CUDA": "0",
}


def setup_cuda() -> dict[str, str]:
    """Wire the CUDA toolkit installed by the workflow's toolkit step."""
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path or not (Path(cuda_path) / "bin" / "nvcc").is_file():
        sys.exit(
            f"CUDA toolkit not found under CUDA_PATH={cuda_path!r}; "
            "install the toolkit before the build phase"
        )
    return {
        "USE_CUDA": "1",
        "CUDA_PATH": cuda_path,
        "PATH": f"{cuda_path}/bin:{os.environ.get('PATH', '')}",
    }


def setup_mkl() -> None:
    """Stage the pinned MKL static libs + headers under MKL_ROOT.

    The mkl wheels unpack into a wheel-layout data tree; the lib/ and
    include/ subtrees are re-hosted under the MKLROOT prefix so the
    module-mode find_package(MKL) can assemble the link line from them.
    Idempotent: an existing staged tree is left untouched. Writing under
    /opt needs root, so the copy runs through sudo (passwordless on CI).
    """
    if (MKL_ROOT / "lib").is_dir():
        return
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "wheel"], check=True
        )
        for pkg in (f"mkl-static=={MKL_VERSION}", f"mkl-include=={MKL_VERSION}"):
            subprocess.run(
                [sys.executable, "-m", "pip", "download", "--quiet",
                 "-d", str(workdir), pkg],
                check=True,
            )
        wheels = list(workdir.glob("*.whl"))
        if not wheels:
            sys.exit(f"No MKL wheels downloaded to {workdir}")
        for wheel in wheels:
            subprocess.run(
                [sys.executable, "-m", "wheel", "unpack", str(wheel)],
                cwd=workdir,
                check=True,
            )
        pairs = []
        static_dir = workdir / f"mkl_static-{MKL_VERSION}"
        include_dir = workdir / f"mkl_include-{MKL_VERSION}"
        for src_root, sub in ((static_dir, "lib"), (include_dir, "include")):
            src = src_root / f"{src_root.name}.data" / "data" / sub
            if not src.is_dir():
                sys.exit(f"Unexpected MKL wheel layout: {src} missing")
            pairs.append((src, MKL_ROOT / sub))
        subprocess.run(["sudo", "mkdir", "-p", str(MKL_ROOT)], check=True)
        for src, dest in pairs:
            subprocess.run(["sudo", "cp", "-a", str(src) + "/.", str(dest)], check=True)


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

    arch = platform.machine()
    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "cpu")
    print(f"build_env_setup.py: arch={arch} GPU_ARCH_TYPE={gpu_arch_type}")

    env_out: dict[str, str] = {"TENSORPLAY_BINARY_BUILD": "1"}

    if gpu_arch_type == "cuda":
        env_out.update(setup_cuda())
    else:
        env_out.update(CPU_BUILD_ENV)

    if arch == "x86_64":
        setup_mkl()
        env_out["MKLROOT"] = str(MKL_ROOT)
        print(f"MKLROOT={MKL_ROOT}")
    elif arch == "aarch64":
        # build_install_deps.py installs the system OpenBLAS; export the
        # prefix so the BLAS detector pins it ahead of any vendored copy.
        env_out["OPENBLAS_ROOT_DIR"] = "/usr"
        print("aarch64: OpenBLAS from the system package")
    else:
        sys.exit(f"Unsupported Linux architecture: {arch}")

    if args.env_out is not None:
        args.env_out.write_text("\n".join(shell_export_lines(env_out)) + "\n")
    print("build env setup complete")


if __name__ == "__main__":
    main()
