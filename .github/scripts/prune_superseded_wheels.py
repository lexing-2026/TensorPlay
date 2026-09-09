#!/usr/bin/env python3
"""Keep only the newest calendar day of wheels on a rolling release.

A rolling nightly release can contain a partial set of wheel tags while
the matrix is still running. The release therefore retains every wheel
from the newest calendar day, regardless of which platform or Python tag
has finished. Within that day, the highest same-tag build is retained.

Version ordering uses `sort -V`, which orders same-day .postN republishes.

Usage:
  prune_superseded_wheels.py --repo OWNER/REPO --release TAG
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import subprocess
import sys

# Wheel file name: tensorplay-<version>[+<variant>]-<python/platform tag>.whl
# macOS CPU wheels carry no local label at all (see the labeling rule in
# filter_published_wheels.py), so the variant part is optional.
WHEEL_RE = re.compile(
    r"^tensorplay-(?P<version>[0-9][^-+]*?)(?:\+(?P<variant>[^-]+))?-(?P<rest>.+)\.whl$"
)
DEV_DATE_RE = re.compile(r"\.dev(?P<day>[0-9]{8})(?:\.post[0-9]+)?$")


@dataclass(frozen=True)
class WheelAsset:
    asset_id: str
    name: str
    version: str
    day: str
    variant: str
    rest: str


def parse_wheel(asset_id: str, name: str) -> WheelAsset | None:
    match = WHEEL_RE.match(name)
    if match is None:
        return None
    day_match = DEV_DATE_RE.search(match["version"])
    if day_match is None:
        return None
    return WheelAsset(
        asset_id=asset_id,
        name=name,
        version=match["version"],
        day=day_match["day"],
        variant=match["variant"] or "",
        rest=match["rest"],
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
    wheels: list[WheelAsset] = []
    unparsed: list[tuple[str, str]] = []
    for line in listing.splitlines():
        asset_id, _, name = line.strip().partition(" ")
        asset_ids[name] = asset_id
        wheel = parse_wheel(asset_id, name)
        if wheel is None:
            unparsed.append((asset_id, name))
            continue
        wheels.append(wheel)

    if wheels:
        newest_day = max(wheel.day for wheel in wheels)
        print(f"retaining nightly wheel day {newest_day}")
        groups: dict[tuple[str, str], list[WheelAsset]] = {}
        for wheel in wheels:
            if wheel.day != newest_day:
                delete_asset(repo, wheel, asset_ids, f"superseded by day {newest_day}")
                continue
            groups.setdefault((wheel.variant, wheel.rest), []).append(wheel)

        for (variant, rest), members in sorted(groups.items()):
            if len(members) == 1:
                continue
            newest = sort_versions([member.version for member in members])[-1]
            for wheel in members:
                if wheel.version != newest:
                    delete_asset(repo, wheel, asset_ids, f"superseded by {newest} for +{variant}-{rest}")
    for _asset_id, name in unparsed:
        print(f"left untouched (not a wheel name): {name}", file=sys.stderr)


def delete_asset(
    repo: str,
    wheel: WheelAsset,
    asset_ids: dict[str, str],
    reason: str,
) -> None:
    result = subprocess.run(
        ["gh", "api", "-X", "DELETE", f"repos/{repo}/releases/assets/{wheel.asset_id}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"failed to prune {wheel.name}: {result.stderr.strip()}", file=sys.stderr)
        return

    print(f"pruned {wheel.name} ({reason})")
    sidecar_name = f"{wheel.name}.sigstore.json"
    sidecar_id = asset_ids.get(sidecar_name)
    if sidecar_id is None:
        return
    sidecar_result = subprocess.run(
        ["gh", "api", "-X", "DELETE", f"repos/{repo}/releases/assets/{sidecar_id}"],
        capture_output=True,
        text=True,
    )
    if sidecar_result.returncode == 0:
        print(f"pruned {sidecar_name} with {wheel.name}")
    else:
        print(
            f"failed to prune {sidecar_name}: {sidecar_result.stderr.strip()}",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="OWNER/REPO of the release")
    parser.add_argument("--release", required=True, help="release tag, e.g. nightly")
    args = parser.parse_args()
    prune(args.repo, args.release)


if __name__ == "__main__":
    main()
