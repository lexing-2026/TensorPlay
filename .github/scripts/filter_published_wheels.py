#!/usr/bin/env python3
"""Drop wheel-matrix legs whose wheel is already published.

A workflow re-run after a partial failure must not rebuild legs whose
wheel — same computed version, variant, Python tag, and platform — is
already on the target release. This script reads the candidate matrices
and the release's asset names, removes the already-published legs, and
emits GITHUB_OUTPUT assignments for the filtered matrices.

Matching is deliberately loose on platform tags (x86_64 instead of the
full manylinux/local tag) so the skip survives packaging-tag drift; the
version, local label, and Python tag must match exactly. Labeling rule:
CUDA/ROCm wheels get +<variant>, CPU wheels get +cpu, macOS wheels carry
no local label at all.

Usage:
  filter_published_wheels.py --version V --cpu-matrix JSON \
      --cuda-matrix JSON [--existing FILE]   # emits cpu=... cuda=...
"""

from __future__ import annotations

import argparse
import json
import sys


# Loose platform discriminators per matrix platform name. An unknown
# platform never skips: building twice beats never publishing.
PLATFORM_TOKENS = {
    "linux-x86_64": ("x86_64",),
    "linux-arm64": ("aarch64",),
    "macos-arm64": ("macosx", "arm64"),
    "windows-x86_64": ("win_amd64",),
}


def published(name: str, version: str, entry: dict) -> bool:
    py = "cp" + entry["python"].replace(".", "")
    tokens = PLATFORM_TOKENS.get(entry["platform"])
    if tokens is None:
        return False
    variant = entry.get("variant", "cpu")
    if variant == "cpu" and entry["platform"] == "macos-arm64":
        prefix = f"tensorplay-{version}-{py}-"
    else:
        prefix = f"tensorplay-{version}+{variant}-{py}-"
    return name.startswith(prefix) and all(token in name for token in tokens)


def filter_matrix(raw: str, version: str, existing: list[str]) -> tuple[list, list]:
    kept, skipped = [], []
    for entry in json.loads(raw):
        if any(published(name, version, entry) for name in existing):
            skipped.append(entry)
        else:
            kept.append(entry)
    return kept, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Computed PEP 440 version, e.g. 1.0.0.dev20260828 or 1.2.0rc1")
    parser.add_argument("--cpu-matrix", required=True, help="CPU matrix JSON from the matrix job")
    parser.add_argument("--cuda-matrix", required=True, help="CUDA matrix JSON from the matrix job")
    parser.add_argument(
        "--existing",
        default="",
        help="File with one release asset name per line; empty or missing means nothing is published yet",
    )
    args = parser.parse_args()

    existing: list[str] = []
    if args.existing:
        with open(args.existing, encoding="utf-8") as stream:
            existing = [line.strip() for line in stream if line.strip()]

    cpu, cpu_skipped = filter_matrix(args.cpu_matrix, args.version, existing)
    cuda, cuda_skipped = filter_matrix(args.cuda_matrix, args.version, existing)
    for entry in cpu_skipped + cuda_skipped:
        # Diagnostics go to stderr: stdout is appended to GITHUB_OUTPUT.
        print(f"Skipping {entry['build_name']}: wheel already published for {args.version}", file=sys.stderr)
    print(f"cpu={json.dumps(cpu)}")
    print(f"cuda={json.dumps(cuda)}")


if __name__ == "__main__":
    main()
