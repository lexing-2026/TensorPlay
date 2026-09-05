#!/usr/bin/env python3
# Build-time source translation for the AMD GPU backend.
#
# Stages the requested source subtrees into the build tree, rewrites CUDA
# runtime/library calls to their HIP equivalents (textual API-name and
# include-path substitution; semantics are unchanged), and records the
# result under renamed paths (cuda/CUDA -> hip/HIP, .cu -> .hip) so both
# toolchains can compile the same kernel sources side by side.

import argparse
import re
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
    "#include <cudnn_frontend.h>": '#include "tp_amd_compat/cudnn_frontend_disabled.h"',
    "#include <cusolverDn.h>": "#include <hipsolver/hipsolver.h>",
    "#include <tensorpipe/tensorpipe_hip.h>": "#include <tensorpipe/tensorpipe_cuda.h>",
    # The HIP staging copies the loops header to hip/HIPLoops.cuh; the
    # hipifier drops this include when file ordering puts the target
    # after the includer, so the mapping is pinned here.
    '#include "CUDALoops.cuh"': '#include "HIPLoops.cuh"',
}

# Symbols the mapping table lacks for the solver compatibility layer.
SYMBOL_FIXES = {
    "cusolverStatus_t": "hipsolverStatus_t",
    "CUSOLVER_STATUS_SUCCESS": "HIPSOLVER_STATUS_SUCCESS",
    "cuFloatComplex": "hipFloatComplex",
    "cuDoubleComplex": "hipDoubleComplex",
    "cuConj": "hipConj",
}

# The primitives wrapper (backend/cuda/GPUPrimitives.cuh) owns the backend
# switch and re-exports the collectives under one namespace, so every
# mapping-table entry carrying the `cub` spelling must survive translation
# unchanged; the wrapper resolves it on the HIP side via a namespace alias.
# The list is derived from the table itself so new entries are picked up.
def _passthrough_keys(hipify_python) -> tuple:
    return tuple(
        key for key in hipify_python.PYTORCH_MAP
        if key == "cub::" or key.startswith("cub/") or key.startswith("cub::")
    )

TEXT_SUFFIXES = {".h", ".hpp", ".cuh", ".cpp", ".cc", ".hip", ".cu"}


def write_compat_shim(staging: Path) -> None:
    # The DNN compatibility header maps the descriptor-style calls in the
    # staged sources onto the AMD DNN library; it carries the whole mapping so
    # the kernel files themselves need no edits.  The frontend-disable stub
    # keeps the __has_include guard in ConvKernels on the legacy path (the
    # backend graph API is C-only here; the C++ builder layer is not shipped).
    shim_dir = staging / "p10" / "include" / "tp_amd_compat"
    shim_dir.mkdir(parents=True, exist_ok=True)
    compat = (Path(__file__).resolve().parent / "cudnn_compat.h").read_text()
    (shim_dir / "cudnn_stub.h").write_text(compat)
    (shim_dir / "cudnn_frontend_disabled.h").write_text(
        "// Marker header staged in place of the C++ DNN frontend builder\n"
        "// layer; its presence tells the guarded code to stay on the\n"
        "// descriptor-based path.\n"
        "#pragma once\n"
    )


def _is_compat(path: Path) -> bool:
    # The compatibility headers ship CUDA-spelled names on purpose; none of
    # the textual passes may touch them (the main translation pass already
    # ignores this directory).
    return "tp_amd_compat" in path.parts


def fix_includes(staging: Path) -> None:
    for path in staging.rglob("*"):
        if path.suffix not in TEXT_SUFFIXES or not path.is_file() or _is_compat(path):
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
        if path.suffix not in TEXT_SUFFIXES or not path.is_file() or _is_compat(path):
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


_INCLUDE_RE = re.compile(r'(#\s*include\s*)(["<])([^">]+)([">])')


def _staged_index(staging: Path) -> set:
    # All file paths inside the staging tree (staging-root relative).  The
    # include rewrite check below matches candidates by path suffix so a
    # bare "Half.h" resolves against staged "p10/include/Half.h".
    return {
        p.relative_to(staging).as_posix()
        for p in staging.rglob("*")
        if p.is_file()
    }


def _unrenamed_candidates(target: str):
    # Reverse of the translation rename rules.  Candidates are tagged with
    # the check each needs:
    #   "strip"  – "_hip"/"_HIP" removed from an otherwise-unchanged stem;
    #              resolves next to the includer or via any -I root.
    #   "stem"   – cuda/CUDA spelling restored inside the stem; the original
    #              must be staged under the public include root so the
    #              reverted name is guaranteed to resolve.
    # System includes (<hip/hip_fp16.h> etc.) have no staged original and
    # keep their translated spelling either way.
    fname = target.rsplit("/", 1)[-1]
    stem, dot, ext = fname.rpartition(".")
    if not stem:
        return
    head = target[: len(target) - len(fname)]
    for suffix in ("_hip", "_HIP"):
        if stem.endswith(suffix):
            yield (head + stem[: -len(suffix)] + dot + ext, "strip")
    for old, new in (("HIP", "CUDA"), ("hip", "cuda")):
        if old in stem:
            yield (head + stem.replace(old, new) + dot + ext, "stem")


def canonicalize_includes(staging: Path) -> None:
    # The translation writes renamed copies next to the originals and
    # rewrites every include that referenced them.  Both spellings resolve
    # inside the staging tree, so one TU can pull the same header twice
    # (once under each name) and hit a duplicate-definition error.  Point
    # every include back at the original-name copy whenever that copy is
    # staged; includes with no staged original keep their translated
    # spelling (ROCm system headers).
    index = _staged_index(staging)
    for path in staging.rglob("*"):
        if path.suffix not in TEXT_SUFFIXES or not path.is_file() or _is_compat(path):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        changed = False

        def _repl(m: "re.Match") -> str:
            nonlocal changed
            prefix, lead, target, trail = m.groups()
            for cand, kind in _unrenamed_candidates(target):
                if kind == "stem":
                    # The reverted name must resolve to the staged copy of
                    # the public header, not to an untranslatable original
                    # somewhere else on the include path.
                    base = cand.rsplit("/", 1)[-1]
                    hit = ("p10/include/" + base) in index
                else:
                    probe = cand
                    while probe.startswith("../"):
                        probe = probe[3:]
                    hit = any(
                        entry == probe or entry.endswith("/" + probe)
                        for entry in index
                    )
                if hit:
                    changed = True
                    return f"{prefix}{lead}{cand}{trail}"
            return m.group(0)

        fixed = _INCLUDE_RE.sub(_repl, text)
        if changed:
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
    # Rebuild the staging root from scratch: a narrowed subdir set must not
    # leave previous-run copies behind, or the build can silently pick up a
    # stale translation that no longer matches the current sources.
    if staging.exists():
        shutil.rmtree(staging)
    for sub in args.subdir:
        src = REPO_ROOT / sub
        dst = staging / sub
        shutil.copytree(src, dst, ignore=ignore)

    write_compat_shim(staging)

    from hipify import hipify_python

    for key in _passthrough_keys(hipify_python):
        hipify_python.PYTORCH_MAP[key] = key

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
    canonicalize_includes(staging)


if __name__ == "__main__":
    main()
