#!/usr/bin/env python3
"""Compute the nightly version, honoring the dispatch build number.

The publish-skip step matches release asset names against the version this
script prints. A build number above 1 must therefore be folded into the
version itself (as a PEP 440 post segment ahead of the dev segment), or
every wheel leg would be dropped as already published.

Environment:
    TP_BUILD_NUMBER   dispatch build number; empty means 1
    TP_INPUT_VERSION  explicit version input; empty derives the nightly rule
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))

from generate_tensorplay_version import append_build_number, compute_nightly_version


def main() -> None:
    build_number = int(os.environ.get("TP_BUILD_NUMBER", "").strip() or 1)
    raw_version = os.environ.get("TP_INPUT_VERSION", "").strip()
    if raw_version:
        version = append_build_number(raw_version, build_number)
    else:
        version = compute_nightly_version(build_number=build_number)
    print(version)


if __name__ == "__main__":
    main()
