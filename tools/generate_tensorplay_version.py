from __future__ import annotations

import datetime
import email
import os
import re
import subprocess
from pathlib import Path

from packaging.version import Version


UNKNOWN = "Unknown"
RELEASE_PATTERN = re.compile(r"/v[0-9]+(\.[0-9]+)*(-rc[0-9]+)?/")


def get_sha(tensorplay_root: str | Path) -> str:
    try:
        rev = None
        if os.path.exists(os.path.join(tensorplay_root, ".git")):
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=tensorplay_root
            )
        elif os.path.exists(os.path.join(tensorplay_root, ".hg")):
            rev = subprocess.check_output(
                ["hg", "identify", "-r", "."], cwd=tensorplay_root
            )
        if rev:
            return rev.decode("ascii").strip()
    except Exception:
        pass
    return UNKNOWN


def get_tag(tensorplay_root: str | Path) -> str:
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=tensorplay_root,
            encoding="ascii",
            capture_output=True,
        ).stdout.strip()
        if RELEASE_PATTERN.match(tag):
            return tag
        else:
            return UNKNOWN
    except Exception:
        return UNKNOWN


def get_tensorplay_version(sha: str | None = None) -> str:
    """Determine the tensorplay version string.

    The version is determined from one of the following sources, in order of
    precedence:
    1. The TENSORPLAY_BUILD_VERSION and TENSORPLAY_BUILD_NUMBER environment
       variables. These are set by the CI when building official releases.
       If built from an sdist, it is checked that the version matches the
       sdist version.
    2. The PKG-INFO file, if it exists. This file is included in source
       distributions (sdist) and contains the version of the sdist.
    3. The version.txt file, which contains the base version string. If the
       git commit SHA is available, it is appended to the version string to
       indicate that this is a development build.
    """
    tensorplay_root = Path(__file__).absolute().parent.parent
    pkg_info_path = tensorplay_root / "PKG-INFO"
    if pkg_info_path.exists():
        with open(pkg_info_path) as f:
            pkg_info = email.message_from_file(f)
        sdist_version = pkg_info["Version"]
    else:
        sdist_version = None
    if os.getenv("TENSORPLAY_BUILD_VERSION"):
        if os.getenv("TENSORPLAY_BUILD_NUMBER") is None:
            raise AssertionError(
                "TENSORPLAY_BUILD_NUMBER must be set when TENSORPLAY_BUILD_VERSION is set"
            )
        build_number = int(os.getenv("TENSORPLAY_BUILD_NUMBER", ""))
        version = os.getenv("TENSORPLAY_BUILD_VERSION", "")
        if build_number > 1:
            version += ".post" + str(build_number)
        origin = "TENSORPLAY_BUILD_{VERSION,NUMBER} env variables"
    elif sdist_version:
        version = sdist_version
        origin = "PKG-INFO"
    else:
        version = Path(tensorplay_root / "version.txt").read_text().strip()
        origin = "version.txt"
        if sdist_version is None and sha != UNKNOWN:
            if sha is None:
                sha = get_sha(tensorplay_root)
            version += "+git" + sha[:7]
            origin += " and git commit"
    # Validate that the version is PEP 440 compliant
    parsed_version = Version(version)
    if sdist_version:
        if (l := parsed_version.local) and l.startswith("git"):
            # Assume local version is git<sha> and
            # hence whole version is source version
            source_version = version
        else:
            # local version is absent or platform tag
            source_version = version.partition("+")[0]
        if sdist_version != source_version:
            raise AssertionError(
                f"Source part '{source_version}' of version '{version}' from "
                f"{origin} does not match version '{sdist_version}' from PKG-INFO"
            )
    return version


def compute_nightly_version(today: str | None = None) -> str:
    """Compute the nightly base version from version.txt.

    Mirrors pytorch's nightly version rule (.ci/pytorch/binary_populate_env.sh):
    the prerelease suffix is stripped from version.txt ("1.0.0a0" -> "1.0.0")
    and a calendar dev segment is appended, e.g. "1.0.0.dev20260828". Variant
    local labels such as "+cu124" or "+cpu" are appended by the packaging
    layer, not here.
    """
    tensorplay_root = Path(__file__).absolute().parent.parent
    base = Path(tensorplay_root / "version.txt").read_text().strip().partition("a")[0]
    if today is None:
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    version = f"{base}.dev{today}"
    # Validate that the version is PEP 440 compliant
    Version(version)
    return version


if __name__ == "__main__":
    # Invoked by CMake at build time to write tensorplay/version.py, which
    # tensorplay/__init__.py consumes (mirrors pytorch's CLI of the same
    # name; the hip/rocm/xpu fields are omitted as TensorPlay does not
    # support those platforms).
    import argparse

    def strtobool(val):
        # pytorch keeps this in tools/strtobool.py
        val = str(val).lower()
        if val in ("y", "yes", "t", "true", "on", "1"):
            return 1
        if val in ("n", "no", "f", "false", "off", "0"):
            return 0
        raise ValueError(f"invalid truth value {val!r}")

    parser = argparse.ArgumentParser(
        description="Generate tensorplay/version.py from build and environment metadata."
    )
    parser.add_argument(
        "--is-debug",
        "--is_debug",
        type=strtobool,
        help="Whether this build is debug mode or not.",
    )
    # CMake may emit a bare "--cuda-version" when no CUDA toolkit is found;
    # tolerate the missing value instead of erroring out mid-build.
    parser.add_argument("--cuda-version", "--cuda_version", type=str,
                        nargs='?', const='', default=None)

    args = parser.parse_args()

    if args.is_debug is None:
        raise AssertionError("is_debug argument must be provided")
    args.cuda_version = None if args.cuda_version == "" else args.cuda_version

    tensorplay_root = Path(__file__).parent.parent
    version_path = tensorplay_root / "tensorplay" / "version.py"
    # Attempt to get tag first, fall back to sha if a tag was not found
    tagged_version = get_tag(tensorplay_root)
    sha = get_sha(tensorplay_root)
    if tagged_version == UNKNOWN:
        version = get_tensorplay_version(sha)
    else:
        version = tagged_version

    with open(version_path, "w") as f:
        f.write("from typing import Optional\n\n")
        f.write("__all__ = ['__version__', 'debug', 'cuda', 'git_version']\n")
        f.write(f"__version__ = '{version}'\n")
        # NB: This is not 100% accurate, because you could have built the
        # library code with DEBUG, but csrc without DEBUG (in which case
        # this would claim to be a release build when it's not.)
        f.write(f"debug = {repr(bool(args.is_debug))}\n")
        f.write(f"cuda: Optional[str] = {repr(args.cuda_version)}\n")
        f.write(f"git_version = {repr(sha)}\n")
