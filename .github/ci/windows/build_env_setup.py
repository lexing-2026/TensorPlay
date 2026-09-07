#!/usr/bin/env python3
"""Build environment setup for Windows wheel builds.

Installs nothing: this phase captures the MSVC toolchain environment
(locating vcvarsall.bat via vswhere.exe), wires the MKL prefix so CMake
finds the pip-provisioned MKL, and emits `export KEY=VALUE` lines to the
file given by --env-out for the parent bash wrapper to source. Without
that handoff the env configured here would die with this process -- bash
and cmd do not share environment state.

Environment variables expected:
    GPU_ARCH_TYPE  - cpu (default) or cuda
    CUDA_PATH      - CUDA toolkit root (CUDA lanes only)
    VSDEVCMD_ARGS  - extra args for vcvarsall.bat (optional)
    VC_YEAR        - 2022 (or 2019)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import write_env_exports


# Common env applied to every Windows wheel build.
COMMON_BUILD_ENV: dict[str, str] = {
    "TENSORPLAY_BINARY_BUILD": "1",
    "BUILD_TEST": "0",
    "INSTALL_TEST": "0",
    "MSSdk": "1",
    "DISTUTILS_USE_SDK": "1",
}


CPU_BUILD_ENV: dict[str, str] = {
    "USE_CUDA": "0",
}


def prepend_path(*entries: Path | str) -> str:
    """Build a Windows PATH (`;`-separated) prepending entries to the current PATH."""
    current = os.environ.get("PATH", "")
    return ";".join((*[str(e) for e in entries], current))


def setup_cuda() -> dict[str, str]:
    """Wire the CUDA toolkit installed by the workflow's toolkit step.

    The toolkit step exports CUDA_PATH; the native build only needs that
    root on PATH (nvcc) and as CUDA_PATH, matching the CMake detector.
    """
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path or not (Path(cuda_path) / "bin" / "nvcc.exe").is_file():
        sys.exit(
            f"CUDA toolkit not found under CUDA_PATH={cuda_path!r}; "
            "install the toolkit before the build phase"
        )
    return {
        "USE_CUDA": "1",
        "CUDA_PATH": cuda_path,
        "PATH": prepend_path(Path(cuda_path) / "bin"),
    }


def find_vcvarsall(vc_year: str) -> Path:
    """Locate vcvarsall.bat via vswhere.exe.

    Honors a pre-set %VS15VCVARSALL% so a runner image can short-circuit
    the discovery. vswhere covers both the classic BuildTools placement
    and per-preview channel installs that fixed paths would miss.
    """
    pre_set = os.environ.get("VS15VCVARSALL")
    if pre_set and Path(pre_set).is_file():
        return Path(pre_set)

    program_files = os.environ.get(
        "ProgramFiles(x86)",
        os.environ.get("ProgramFiles", r"C:\Program Files (x86)"),
    )
    vswhere = (
        Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    if not vswhere.is_file():
        sys.exit(
            f"vswhere.exe not found at {vswhere}; Visual Studio {vc_year} "
            "C++ BuildTools is required to compile the wheel"
        )

    vc_lower, vc_upper = ("16", "17") if vc_year == "2019" else ("17", "18")
    output = subprocess.check_output(
        [
            str(vswhere),
            "-legacy",
            "-products",
            "*",
            "-version",
            f"[{vc_lower},{vc_upper})",
            "-property",
            "installationPath",
        ],
        text=True,
    ).strip()
    for line in output.splitlines():
        candidate = Path(line.strip()) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if candidate.is_file():
            return candidate
    sys.exit(
        f"Visual Studio {vc_year} C++ BuildTools is required to compile the wheel"
    )


def _capture_cmd_env(command: str, fail_prefix: str) -> dict[str, str]:
    """Run `command` under `cmd /u /c`, capture env, return diff vs current env.

    `cmd /u` forces UTF-16LE output so non-ASCII paths in localized Windows
    installs round-trip intact. The diff is against the live `os.environ`,
    so callers can pre-populate os.environ with any vars they want layered
    on top of (rather than shadowed by) the command's exports.
    """
    try:
        raw = subprocess.check_output(
            f"cmd /u /c {command}",
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(
            f"{fail_prefix} failed:\n{exc.output.decode('utf-16le', errors='replace')}"
        )
    text = raw.decode("utf-16le", errors="replace")

    new_env: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        # Skip cmd's `=C:` / `=ExitCode` / `=::` pseudo-vars and banner lines.
        if not sep or not key or key.startswith("="):
            continue
        new_env[key] = value

    old_env = os.environ
    return {k: v for k, v in new_env.items() if old_env.get(k) != v}


def capture_vcvars_env(vcvarsall: Path, args: str) -> dict[str, str]:
    """Capture the env diff produced by `vcvarsall.bat <args>`."""
    return _capture_cmd_env(f'"{vcvarsall}" {args} && set', f"vcvarsall.bat {args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-out",
        type=Path,
        help="Write `export KEY=VALUE` lines here for bash to source.",
    )
    args = parser.parse_args()

    gpu_arch_type = os.environ.get("GPU_ARCH_TYPE", "cpu")
    vc_year = os.environ.get("VC_YEAR", "2022")
    print(f"build_env_setup.py: GPU_ARCH_TYPE={gpu_arch_type} VC_YEAR={vc_year}")

    env_out: dict[str, str] = {**COMMON_BUILD_ENV}

    if gpu_arch_type == "cpu":
        env_out.update(CPU_BUILD_ENV)
        print("CPU environment configured")
    elif gpu_arch_type == "cuda":
        env_out.update(setup_cuda())
    else:
        sys.exit(
            f"build_env_setup.py: GPU_ARCH_TYPE={gpu_arch_type!r} not supported. "
            "Expected one of: cpu, cuda."
        )

    # Point CMake at the running Python's Library/ tree so it finds the pip
    # MKL (installed by build_install_deps.py into that conda-style layout:
    # import libs + headers under Library/lib and Library/include, runtime
    # DLLs under Library/bin). Without it the BLAS detector falls through to
    # the unaccelerated fallback. Prepend rather than assign so a lane that
    # already set its own prefix keeps it.
    mkl_prefix = str(Path(sys.prefix) / "Library")
    device_prefix = env_out.get("CMAKE_PREFIX_PATH", "")
    env_out["CMAKE_PREFIX_PATH"] = (
        f"{mkl_prefix};{device_prefix}" if device_prefix else mkl_prefix
    )

    # Push our env into the current process so vcvarsall's PATH extension
    # layers on top of (rather than replaces) our additions when captured.
    os.environ.update(env_out)

    # vcvarsall env -- captured last so its PATH/LIB/etc. extensions stack
    # on top of whatever we just configured.
    vcvarsall = find_vcvarsall(vc_year)
    print(f"Sourcing {vcvarsall}")
    vsdevcmd_args = os.environ.get("VSDEVCMD_ARGS", "")
    env_out.update(capture_vcvars_env(vcvarsall, f"x64 {vsdevcmd_args}".strip()))

    write_env_exports(env_out, args.env_out)
    print("build env setup complete")


if __name__ == "__main__":
    main()
