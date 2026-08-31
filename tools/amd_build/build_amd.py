#!/usr/bin/env python3
# Build-time source translation for the AMD GPU backend.
#
# Stages the requested source subtrees into the build tree, rewrites CUDA
# runtime/library calls to their HIP equivalents (textual API-name and
# include-path substitution; semantics are unchanged), and records the
# result under renamed paths (cuda/CUDA -> hip/HIP, .cu -> .hip) so both
# toolchains can compile the same kernel sources side by side.

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "pytorch" / "torch" / "utils"))

STAGE_GLOBS = ["*.h", "*.hpp", "*.cuh", "*.cpp", "*.cc", "*.cu", "*.in"]

# Post-translation include fixes for headers the mapping table leaves in
# CUDA spelling even though the compatibility packages live elsewhere.
# The tensorpipe entry reverts a mapping-table invention: the vendored
# tensorpipe ships no HIP-spelled header, and the include is macro-guarded
# anyway, so the original spelling must survive the translation.
INCLUDE_FIXES = {
    "#include <cudnn.h>": '#include "tp_amd_compat/cudnn_stub.h"',
    "#include <cusolverDn.h>": "#include <hipsolver/hipsolver.h>",
    "#include <tensorpipe/tensorpipe_hip.h>": "#include <tensorpipe/tensorpipe_cuda.h>",
}

# Symbols the mapping table lacks for the solver compatibility layer.
SYMBOL_FIXES = {
    "cusolverStatus_t": "hipsolverStatus_t",
    "CUSOLVER_STATUS_SUCCESS": "HIPSOLVER_STATUS_SUCCESS",
    "cuFloatComplex": "hipFloatComplex",
    "cuDoubleComplex": "hipDoubleComplex",
    "cuConj": "hipConj",
}

TEXT_SUFFIXES = {".h", ".hpp", ".cuh", ".cpp", ".cc", ".hip", ".cu"}


def write_compat_shim(staging: Path) -> None:
    # Satisfies the unconditional DNN include; every DNN-backed code path
    # stays behind USE_CUDNN and is inactive in this backend configuration.
    shim_dir = staging / "p10" / "include" / "tp_amd_compat"
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "cudnn_stub.h").write_text(
        "// Placeholder for builds without the DNN library: references that\n"
        "// reach this header are guarded by USE_CUDNN and compile to nothing.\n"
        "#pragma once\n"
    )


def fix_includes(staging: Path) -> None:
    for path in staging.rglob("*"):
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        fixed = text
        for old, new in INCLUDE_FIXES.items():
            fixed = fixed.replace(old, new)
        for old, new in SYMBOL_FIXES.items():
            fixed = fixed.replace(old, new)
        if fixed != text:
            path.write_text(fixed)


def rewrite_leftovers(staging: Path) -> None:
    # The translation writes renamed copies (hip/HIP*, .hip) next to the
    # originals.  TUs that keep compiling from the original tree still
    # resolve the original header names, so every staged file that did not
    # get a renamed copy gets the same textual rewrite in place.  This keeps
    # both lookup orders (original name and renamed name) on HIP semantics.
    # Prepend the vendored translation module search path so the in-place
    # pass can reuse the same mapping tables.
    sys.path.insert(
        0, str(REPO_ROOT / "third_party" / "pytorch" / "torch" / "utils"))
    from hipify import hipify_python as hp

    for path in staging.rglob("*"):
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if text.startswith(hp.HIPIFY_C_BREADCRUMB):
            continue  # already a renamed translation product
        fixed = hp.RE_PYTORCH_PREPROCESSOR.sub(
            lambda m: hp.PYTORCH_MAP[m.group(0)], text)
        for old, new in INCLUDE_FIXES.items():
            fixed = fixed.replace(old, new)
        for old, new in SYMBOL_FIXES.items():
            fixed = fixed.replace(old, new)
        if fixed != text:
            path.write_text(fixed)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage and translate GPU kernel sources for the AMD backend"
    )
    ap.add_argument("--output-dir", required=True, help="staging root in the build tree")
    ap.add_argument(
        "--subdir",
        action="append",
        required=True,
        help="repo-relative source directory to stage (repeatable)",
    )
    ap.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="directory name to exclude from staging (repeatable)",
    )
    args = ap.parse_args()

    staging = Path(args.output_dir).resolve()
    ignore = shutil.ignore_patterns(*args.ignore) if args.ignore else None
    for sub in args.subdir:
        src = REPO_ROOT / sub
        dst = staging / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=ignore)

    write_compat_shim(staging)

    from hipify import hipify_python

    hipify_python.hipify(
        project_directory=str(staging),
        output_directory=str(staging),
        includes=STAGE_GLOBS,
        ignores=["*/tp_amd_compat/*"],
        show_progress=False,
        hip_clang_launch=True,
        is_pytorch_extension=True,
        header_include_dirs=["p10/include"],
    )

    fix_includes(staging)
    rewrite_leftovers(staging)


if __name__ == "__main__":
    main()
