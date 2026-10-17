#!/usr/bin/env python3
"""MPS speedup gate.

The MPS operator benchmark times the same case inventory twice, once per
runtime, in two separate processes: this tree's kernels in one, a second
runtime on the same device in the other. Separate processes keep the two
passes from competing for device memory and let each side warm up on its own.

This gate joins the two result files by case name, turns the pair of times into
a per-case speedup (theirs / ours), a geometric-mean overall speedup and a win
rate, then fails when the overall number is below the required floor.

A run whose second pass is missing or empty has no ratios to gate on; it is
reported as missing rather than silently passing.

Usage:
    check_mps_reference_speedup.py --operator-json mps-operator.json \
        --reference-json mps-operator-reference.json \
        --min-geomean 1.0 --min-win-rate 0.0 \
        --summary-out mps-speedup-summary.md
"""

import argparse
import json
import math
import statistics
from pathlib import Path

SCHEMA_VERSION = 1
TOP_LOSSES = 12
TOP_WINS = 8


def load(path, require_measurements=True):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("benchmark") != "mps-operator":
        raise ValueError(f"{path} is not an mps-operator benchmark")
    measurements = payload.get("measurements")
    if not isinstance(measurements, list):
        raise ValueError(f"{path} has no measurements")
    if require_measurements and not measurements:
        raise ValueError(f"{path} has no measurements")
    return payload


def _positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def join(ours, reference):
    """Pair each of our measurements with the reference's time for the case."""
    for field in ("suite", "dtype", "device"):
        if ours.get(field) != reference.get(field):
            raise ValueError(f"{field} differs between the two runs")
    if ours.get("side") != "ours":
        raise ValueError("the operator run is not the 'ours' side")
    if reference.get("side") != "reference":
        raise ValueError("the reference run is not the 'reference' side")
    theirs = {item["name"]: item.get("seconds")
              for item in reference["measurements"]}
    measurements = []
    for item in ours["measurements"]:
        paired = dict(item)
        seconds = theirs.get(item["name"])
        if _positive(seconds):
            paired["reference_seconds"] = seconds
        measurements.append(paired)
    joined = dict(ours)
    joined["measurements"] = measurements
    joined["reference"] = reference.get("runtime") or {
        "runtime": "reference", "version": "",
        "device": reference.get("device"),
    }
    return joined


def summarize(payload):
    """Reduce paired measurements to speedup rows and aggregate numbers."""
    rows = []
    for item in payload["measurements"]:
        ours = item.get("seconds")
        theirs = item.get("reference_seconds")
        if not _positive(ours) or not _positive(theirs):
            continue
        rows.append({
            "name": item["name"],
            "category": item.get("category", ""),
            "ours": ours,
            "theirs": theirs,
            "speedup": theirs / ours,
        })
    if not rows:
        return {"rows": [], "geomean": 0.0, "win_rate": 0.0,
                "wins": 0, "compared": 0,
                "reference": payload.get("reference", {})}
    speedups = [row["speedup"] for row in rows]
    wins = sum(1 for value in speedups if value >= 1.0)
    return {
        "rows": rows,
        "geomean": math.exp(statistics.mean(map(math.log, speedups))),
        "win_rate": wins / len(rows),
        "wins": wins,
        "compared": len(rows),
        "reference": payload.get("reference", {}),
    }


def evaluate(summary, min_geomean, min_win_rate, max_case_slowdown):
    """Return the list of gate violations for one summary."""
    violations = []
    if summary["compared"] == 0:
        return ["no case carries both timings; rerun the benchmark with "
                "--reference"]
    if summary["geomean"] < min_geomean:
        violations.append(
            f"overall speedup {summary['geomean']:.2f}x is below the "
            f"required {min_geomean:.2f}x"
        )
    if summary["win_rate"] < min_win_rate:
        violations.append(
            f"win rate {summary['win_rate'] * 100:.0f}% "
            f"({summary['wins']}/{summary['compared']}) is below the "
            f"required {min_win_rate * 100:.0f}%"
        )
    if max_case_slowdown is not None:
        limit = 1.0 / (1.0 + max_case_slowdown)
        for row in summary["rows"]:
            if row["speedup"] < limit:
                violations.append(
                    f"{row['name']}: {row['speedup']:.2f}x is below the "
                    f"per-case floor {limit:.2f}x"
                )
    return violations


def write_summary(path, summary, violations, min_geomean, min_win_rate):
    lines = ["# MPS speedup against the reference runtime", ""]
    if summary["compared"] == 0:
        lines.append("No paired timings in this run.")
        lines.append("")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    reference = summary["reference"] or {}
    runtime = reference.get("runtime", "reference")
    version = reference.get("version", "")
    lines.append(f"- cases compared: {summary['compared']}")
    lines.append(f"- reference: {runtime} {version}".rstrip())
    lines.append(
        f"- overall speedup: **{summary['geomean']:.2f}x** "
        f"(floor {min_geomean:.2f}x)"
    )
    lines.append(
        f"- win rate: **{summary['win_rate'] * 100:.0f}%** "
        f"({summary['wins']}/{summary['compared']}, floor "
        f"{min_win_rate * 100:.0f}%)"
    )
    lines.append("")
    losses = sorted(summary["rows"], key=lambda row: row["speedup"])[:TOP_LOSSES]
    wins = sorted(summary["rows"], key=lambda row: -row["speedup"])[:TOP_WINS]
    lines.extend([
        "## Furthest behind",
        "",
        "| case | ours (ms) | reference (ms) | speedup |",
        "| --- | ---: | ---: | ---: |",
    ])
    lines.extend(
        f"| {row['name']} | {row['ours'] * 1e3:.3f} | "
        f"{row['theirs'] * 1e3:.3f} | {row['speedup']:.2f}x |"
        for row in losses
    )
    lines.extend([
        "",
        "## Furthest ahead",
        "",
        "| case | ours (ms) | reference (ms) | speedup |",
        "| --- | ---: | ---: | ---: |",
    ])
    lines.extend(
        f"| {row['name']} | {row['ours'] * 1e3:.3f} | "
        f"{row['theirs'] * 1e3:.3f} | {row['speedup']:.2f}x |"
        for row in wins
    )
    lines.append("")
    if violations:
        lines.extend(["## Gate", ""])
        lines.extend(f"- {item}" for item in violations)
    else:
        lines.append("No speedup-floor violations detected.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, default=None,
                        help="second pass, timed by the comparison runtime")
    parser.add_argument("--min-geomean", type=float, default=1.0)
    parser.add_argument("--min-win-rate", type=float, default=0.0)
    parser.add_argument("--max-case-slowdown", type=float, default=None)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    for name, value, upper in (("min-geomean", args.min_geomean, None),
                               ("min-win-rate", args.min_win_rate, 1.0),
                               ("max-case-slowdown", args.max_case_slowdown,
                                None)):
        if value is None:
            continue
        if not math.isfinite(value) or value < 0 or (upper and value > upper):
            parser.error(f"{name} is out of range")

    payload = load(args.operator_json)
    if args.reference_json is not None and args.reference_json.exists():
        # The second pass may legitimately record nothing when the device
        # cannot provision it; the join then reports the missing pairs.
        reference = load(args.reference_json, require_measurements=False)
        payload = join(payload, reference)
    summary = summarize(payload)
    violations = evaluate(summary, args.min_geomean, args.min_win_rate,
                          args.max_case_slowdown)
    write_summary(args.summary_out, summary, violations, args.min_geomean,
                  args.min_win_rate)
    if summary["compared"]:
        print(f"cases compared: {summary['compared']}")
        print(f"overall speedup: {summary['geomean']:.2f}x")
        print(f"win rate: {summary['win_rate'] * 100:.0f}%")
    for item in violations:
        print(item)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
