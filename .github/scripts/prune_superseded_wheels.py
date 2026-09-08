#!/usr/bin/env python3
"""Keep only the newest wheel per tag group on a rolling release.

A rolling nightly release accumulates one wheel per dev version per
wheel tag (local label + Python tag + platform tag) unless older
versions are pruned. Consumers that glob a wheel pattern and install
the whole match set would then be handed mutually exclusive versions.
This script groups the release assets by wheel tag, keeps the
highest-version wheel of each group, and deletes the rest.

Version ordering uses `sort -V`, which orders the X.Y.0.dev<UTC date>
versions produced by tools/generate_tensorplay_version.py correctly,
including .postN same-day republishes.

Usage:
  prune_superseded_wheels.py --repo OWNER/REPO --release TAG
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# Wheel file name: tensorplay-<version>[+<variant>]-<python/platform tag>.whl
# macOS CPU wheels carry no local label at all (see the labeling rule in
# filter_published_wheels.py), so the variant part is optional.
WHEEL_RE = re.compile(
    r"^tensorplay-(?P<version>[0-9][^-+]*?)(?:\+(?P<variant>[^-]+))?-(?P<rest>.+)\.whl$"
)


def gh(*args: str) -> str:
    result = subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    )
    return result.stdout


def sort_versions(versions: list[str]) -> list[str]:
    """Return the versions ordered by `sort -V` (dev/post aware enough)."""
    result = subprocess.run(
        ["sort", "-V"], input="\n".join(versions) + "\n", capture_output=True, text=True
    )
    return result.stdout.splitlines()


def prune(repo: str, release: str) -> None:
    listing = gh(
        "api",
        f"repos/{repo}/releases/tags/{release}",
        "--jq",
        r'.assets[] | "\(.id) \(.name)"',
    )
    asset_ids: dict[str, str] = {}
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    unparsed: list[tuple[str, str]] = []
    for line in listing.splitlines():
        asset_id, _, name = line.strip().partition(" ")
        asset_ids[name] = asset_id
        match = WHEEL_RE.match(name)
        if match is None:
            unparsed.append((asset_id, name))
            continue
        key = (match["variant"] or "", match["rest"])
        groups.setdefault(key, []).append((match["version"], asset_id, name))

    for (variant, rest), members in sorted(groups.items()):
        if len(members) == 1:
            continue
        versions = sort_versions([m[0] for m in members])
        newest = versions[-1]
        for version, asset_id, name in members:
            if version == newest:
                continue
            # Deletion failures are logged but not fatal: the next upload
            # of a newer wheel retries the prune for this tag group.
            result = subprocess.run(
                ["gh", "api", "-X", "DELETE", f"repos/{repo}/releases/assets/{asset_id}"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print(f"pruned {name} (superseded by {newest} for +{variant}-{rest})")
                sidecar_name = f"{name}.sigstore.json"
                sidecar_id = asset_ids.get(sidecar_name)
                if sidecar_id:
                    sidecar_result = subprocess.run(
                        ["gh", "api", "-X", "DELETE", f"repos/{repo}/releases/assets/{sidecar_id}"],
                        capture_output=True,
                        text=True,
                    )
                    if sidecar_result.returncode == 0:
                        print(f"pruned {sidecar_name} with {name}")
                    else:
                        print(
                            f"failed to prune {sidecar_name}: {sidecar_result.stderr.strip()}",
                            file=sys.stderr,
                        )
            else:
                print(
                    f"failed to prune {name}: {result.stderr.strip()}",
                    file=sys.stderr,
                )
    for _asset_id, name in unparsed:
        print(f"left untouched (not a wheel name): {name}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="OWNER/REPO of the release")
    parser.add_argument("--release", required=True, help="release tag, e.g. nightly")
    args = parser.parse_args()
    prune(args.repo, args.release)


if __name__ == "__main__":
    main()
