#!/usr/bin/env python3
"""Report objects whose header dependencies the build has never learned.

Ninja works out which headers an object reads by reading the dependency file the
compiler writes as the edge runs, and it folds that list into its own database.
A launcher that answers a compile from a cache hands back the object without
ever running the compiler, so no file is produced and the edge records nothing:
the object is left with an empty dependency list.  From then on an edit to any
header it includes does not dirty it, the build reports "no work to do", and the
library keeps the old code while the source says otherwise.  That is a silent
no-op, not a failure.

The state arrives with a build directory populated from a warm cache -- every
edge answered from the cache, so nothing ever had a dependency list -- and
persists object by object until one is really compiled.  A build directory that
was configured more than once also carries objects no longer in the graph; those
are dead files rather than frozen edges, and they are reported apart.

Only a pass that both runs the compiler and refreshes the cache teaches an edge
its headers.  Turning the cache off is not enough: the answer still comes from
the launcher, which never hands the dependency file over.  Recaching compiles
for real, writes the object a second time, and leaves the edge knowing every
header it read.  The command to do that is printed with the report.

Usage:
    python tools/check_depfiles.py --build-dir build --target p10
    python tools/check_depfiles.py --build-dir build --target p10 --jobs 4
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_DEPS_LINE = re.compile(r"#deps (\d+)(INVALID)?")

# Three different states: an edge that records no headers, an edge that knows
# them, and a file left behind by an older configuration.
FROZEN = "frozen"
LEARNED = "learned"
STALE = "stale"


def objects_under(build_dir: Path, target: str) -> list[str]:
    """Every built object of one target, in the paths ninja names them by."""
    root = build_dir / target / "CMakeFiles" / f"{target}.dir"
    if not root.is_dir():
        sys.exit(f"no object directory for target {target}: {root}")
    return sorted(str(path.relative_to(build_dir)) for path in root.rglob("*.o"))


def classify(build_dir: Path, obj: str) -> tuple[str, bool]:
    """What the build knows about one object's headers, and whether that is stale."""
    proc = subprocess.run(
        ["ninja", "-t", "deps", obj],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    first = proc.stdout.splitlines()[0] if proc.stdout else ""
    match = _DEPS_LINE.search(first)
    if proc.returncode != 0 or match is None:
        return STALE, False
    return (FROZEN if int(match.group(1)) == 0 else LEARNED), bool(match.group(2))


def source_of(build_dir: Path, obj: str) -> str | None:
    """The translation unit an object edge compiles."""
    proc = subprocess.run(
        ["ninja", "-t", "commands", obj],
        cwd=build_dir,
        capture_output=True,
        text=True,
    )
    for line in reversed(proc.stdout.splitlines()):
        argv = shlex.split(line)
        if "-c" in argv:
            return argv[argv.index("-c") + 1]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--target", default="p10")
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="parallelism to quote in the printed remedy",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="name the translation unit behind each frozen object",
    )
    args = parser.parse_args()

    build_dir = args.build_dir
    if not (build_dir / "build.ninja").is_file():
        sys.exit(f"{build_dir} is not a ninja build directory")

    every = objects_under(build_dir, args.target)
    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(pool.map(lambda obj: classify(build_dir, obj), every))

    frozen: list[str] = []
    stale: list[str] = []
    broken: list[str] = []
    for obj, (state, dep_missing) in zip(every, states):
        if state == FROZEN:
            frozen.append(obj)
        elif state == STALE:
            stale.append(obj)
        elif dep_missing:
            broken.append(obj)

    print(
        f"{args.target}: {len(frozen)} of {len(every)} objects carry no header "
        f"list; {len(every) - len(frozen) - len(stale)} learn their headers, "
        f"{len(stale)} are no longer built"
    )
    if broken:
        print(f"  {len(broken)} objects list headers that no longer exist; "
              "they rebuild on the next pass")

    if not frozen:
        return 0

    print()
    for obj in frozen:
        print(f"  {obj}")
        if args.sources:
            print(f"      <- {source_of(build_dir, obj) or 'source unknown'}")

    targets = " ".join(shlex.quote(obj) for obj in frozen)
    print(
        "\none recached pass teaches the build these dependencies (deleting the\n"
        "object, not touching its source: an unchanged source is answered from\n"
        f"the cache again):\n\n  rm {targets} && "
        f"env SCCACHE_RECACHE=1 ninja -C {build_dir} -j{args.jobs} {targets}\n"
    )
    print("CUDA edges rebuild slowly; run them apart from the C++ ones if the\n"
          "box is busy. Confirm with this same report -- an object still listed\n"
          "was not compiled.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
