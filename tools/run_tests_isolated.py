"""Run the test suite one file per process and report a per-file verdict.

A single pytest process shares one device context, one allocator and one
set of module-level caches across every test file.  One illegal access or
one leaked allocation therefore turns every later device test into the
same failure, and the run can no longer say which file is broken.  This
runner gives each file its own interpreter, so a failure stays local and
the summary can separate "this file fails" from "everything after this
file fails".

    python tools/run_tests_isolated.py                  # all of test/
    python tools/run_tests_isolated.py test/nn -k sdpa  # a subtree
    python tools/run_tests_isolated.py --jobs 4         # four at a time

Files run sequentially by default: device memory is a shared resource and
parallel runs would make the timings and the out-of-memory behaviour
depend on the schedule.  ``--jobs`` exists for CPU-only runs.

Anything after ``--`` goes to pytest unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def discover(paths):
    files = []
    for path in paths:
        if os.path.isfile(path):
            files.append(path)
            continue
        for root, dirs, names in os.walk(path):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for name in sorted(names):
                if name.startswith("test_") and name.endswith(".py"):
                    files.append(os.path.join(root, name))
    return files


def parse_junit(path):
    """Return (passed, failed, skipped, errored) from a junit report."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errored": 0}
    for suite in suites:
        for case in suite.findall("testcase"):
            outcomes = [child.tag for child in case]
            if "failure" in outcomes:
                counts["failed"] += 1
            elif "error" in outcomes:
                counts["errored"] += 1
            elif "skipped" in outcomes:
                counts["skipped"] += 1
            else:
                counts["passed"] += 1
    return counts


def run_file(path, extra, timeout):
    started = time.monotonic()
    tmpdir = tempfile.mkdtemp(prefix="tpi-")
    report = os.path.join(tmpdir, "report.xml")
    command = [sys.executable, "-m", "pytest", path, "-q", "--tb=no",
               f"--junitxml={report}", "-p", "no:cacheprovider", *extra]
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout)
        output = proc.stdout + proc.stderr
        code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        code = -1
        output += f"\n[runner] timed out after {timeout}s\n"
    counts = parse_junit(report)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return {
        "file": path,
        "code": code,
        "counts": counts,
        "seconds": round(time.monotonic() - started, 1),
        "tail": output.strip().splitlines()[-1] if output.strip() else "",
    }


def verdict(result):
    if result["code"] == 0:
        return "ok"
    if result["code"] == -1:
        return "timeout"
    counts = result["counts"]
    if counts is None:
        return "error"
    if counts["errored"]:
        return "error"
    if counts["failed"]:
        return "fail"
    # Non-zero without a failure: collection error, interrupted run, or a
    # crash in the interpreter's exit path.
    return "crash"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["test"],
                        help="files or directories to run (default: test)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="files to run at once (default 1)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="per-file timeout in seconds (default 3600)")
    parser.add_argument("--json", default=None,
                        help="write the per-file results to this path")
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    else:
        extra = []
    args = parser.parse_args(argv)

    files = discover(args.paths or ["test"])
    if not files:
        print("no test files found", file=sys.stderr)
        return 2

    print(f"{len(files)} files, jobs={args.jobs}, timeout={args.timeout}s")
    results = []
    started = time.monotonic()
    if args.jobs == 1:
        for index, path in enumerate(files, 1):
            result = run_file(path, extra, args.timeout)
            results.append(result)
            counts = result["counts"] or {}
            print(f"[{index:>3}/{len(files)}] {verdict(result):<8} "
                  f"{result['seconds']:>7.1f}s  {path}  "
                  f"({counts.get('passed', 0)}p/{counts.get('failed', 0)}f/"
                  f"{counts.get('skipped', 0)}s/{counts.get('errored', 0)}e)",
                  flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            futures = {pool.submit(run_file, path, extra, args.timeout): path
                       for path in files}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                counts = result["counts"] or {}
                print(f"{verdict(result):<8} {result['seconds']:>7.1f}s  "
                      f"{result['file']}  ({counts.get('passed', 0)}p/"
                      f"{counts.get('failed', 0)}f/{counts.get('skipped', 0)}s/"
                      f"{counts.get('errored', 0)}e)", flush=True)

    elapsed = time.monotonic() - started
    totals = {key: 0 for key in ("passed", "failed", "skipped", "errored")}
    for result in results:
        for key, value in (result["counts"] or {}).items():
            totals[key] += value
    bad = [r for r in results if verdict(r) != "ok"]
    print()
    print(f"{len(results)} files in {elapsed / 60:.1f} min: "
          f"{totals['passed']} passed, {totals['failed']} failed, "
          f"{totals['errored']} errored, {totals['skipped']} skipped")
    if bad:
        print(f"{len(bad)} file(s) did not pass:")
        for result in sorted(bad, key=lambda r: r["file"]):
            print(f"  {verdict(result):<8} {result['file']}  {result['tail']}")

    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"elapsed_seconds": round(elapsed, 1),
                       "totals": totals,
                       "files": sorted(results, key=lambda r: r["file"])},
                      handle, indent=2)
        print(f"wrote {args.json}")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
