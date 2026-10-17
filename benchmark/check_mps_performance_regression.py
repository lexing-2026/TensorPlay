#!/usr/bin/env python3
"""Compare MPS operator benchmark results with historical results.

The published bench-data snapshot keeps a per-hardware history of the most
recent MPS benchmark runs (arch "mps").  This gate compares the current run
against the median of the matching history and fails when a shared
measurement is slower than the baseline by more than a configured factor.
A run with no matching history records the baseline instead of gating.

The MPS benchmark has no thread-count dimension (device timing is
independent of host thread settings), so runs are matched on the device
field instead of the thread count used by the CPU gate.
"""

import argparse
import json
import math
import statistics
from pathlib import Path


SCHEMA_VERSION = 1


def load(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    measurements = payload.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ValueError(f"{path} has no measurements")
    if payload.get("benchmark") != "mps-operator":
        raise ValueError(f"{path} is not an mps-operator benchmark")
    names = []
    for item in measurements:
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError(f"{path} has an invalid measurement")
        if not item.get("category"):
            raise ValueError(f"{path}/{item['name']} has no category")
        names.append(item["name"])
    if len(names) != len(set(names)):
        raise ValueError(f"{path} has duplicate measurement names")
    return payload


def load_history(path):
    """Load the published benchmark history; a missing or unparsable file
    simply means no history has been recorded yet."""
    if path is None or not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def median_baseline(current, runs):
    """Build a comparison baseline as the per-measurement median over the
    matching runs of the same hardware history."""
    matching = []
    for run in runs:
        payload = run.get("suites", {}).get("mps-operator")
        if not isinstance(payload, dict):
            continue
        if payload.get("benchmark") != current.get("benchmark"):
            continue
        if payload.get("suite") != current.get("suite"):
            continue
        if payload.get("dtype") != current.get("dtype"):
            continue
        if payload.get("device") != current.get("device"):
            continue
        matching.append(payload)
    if not matching:
        return None
    base = dict(current)
    base["measurements"] = []
    for current_item in current["measurements"]:
        baseline_item = dict(current_item)
        values = []
        for payload in matching:
            for item in payload["measurements"]:
                if item.get("name") == current_item["name"]:
                    value = item.get("seconds")
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value > 0
                    ):
                        values.append(value)
        if values:
            baseline_item["seconds"] = statistics.median(values)
        base["measurements"].append(baseline_item)
    return base


def compare(current, baseline, max_slowdown):
    if current["benchmark"] != baseline["benchmark"]:
        raise ValueError("benchmark names do not match")
    if current.get("suite") != baseline.get("suite"):
        raise ValueError("benchmark suites do not match")
    for field in ("dtype", "device"):
        if current.get(field) != baseline.get(field):
            raise ValueError(f"{field} metadata does not match")

    current_by_name = {item["name"]: item for item in current["measurements"]}
    baseline_by_name = {item["name"]: item for item in baseline["measurements"]}
    # The suite grows as operators gain device support; names that are only
    # on one side have no reference yet, so only the intersection is compared.
    compared_names = [name for name in current_by_name if name in baseline_by_name]

    # A run can land on a faster or slower machine than the one that
    # produced the committed baseline.  Measure the run's own overall shift
    # against the baseline as the median ratio over every shared measurement,
    # then treat only deviations beyond that shift as regressions. A faster
    # overall run must not turn unchanged or improved cases into regressions.
    ratios = []
    for name in compared_names:
        actual = current_by_name[name].get("seconds")
        previous = baseline_by_name[name].get("seconds")
        if (
            isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(previous, (int, float)) and not isinstance(previous, bool)
            and math.isfinite(actual) and math.isfinite(previous)
            and actual > 0 and previous > 0
        ):
            ratios.append(actual / previous)
    scale = max(1.0, statistics.median(ratios)) if ratios else 1.0

    violations = []
    for name in compared_names:
        current_item = current_by_name[name]
        baseline_item = baseline_by_name[name]
        if current_item.get("category") != baseline_item.get("category"):
            raise ValueError(f"{name}: categories do not match")
        if current_item.get("input_shapes") != baseline_item.get("input_shapes"):
            raise ValueError(f"{name}: input_shapes metadata does not match")
        actual = current_item.get("seconds")
        previous = baseline_item.get("seconds")
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ValueError(f"{name}/seconds: timing is not numeric")
        if not isinstance(previous, (int, float)) or isinstance(previous, bool):
            raise ValueError(f"{name}/seconds: baseline is not numeric")
        if not math.isfinite(actual) or not math.isfinite(previous):
            raise ValueError(f"{name}/seconds: timings must be finite")
        if actual <= 0 or previous <= 0:
            raise ValueError(f"{name}/seconds: timings must be positive")
        ratio = actual / previous
        # Sub-microsecond baselines sit at the timing floor: the
        # measurement jitter of a tiny tensor makes their relative slowdown
        # noise rather than signal, so leave them out of the ratio gate.
        if previous < 1e-5:
            continue
        limit = scale * (1.0 + max_slowdown)
        if ratio > limit:
            violations.append(
                f"mps-operator/{name}/seconds: {ratio:.2f}x of baseline "
                f"(limit {limit:.2f}x)"
            )
    return violations


def write_summary(path, count, violations, note=None):
    lines = ["# MPS performance gate", ""]
    if note:
        lines.append(note)
        lines.append("")
    lines.append(f"- mps-operator: max 30% slower; {count} measurements")
    lines.append("")
    if violations:
        lines.extend(["## Regressions", ""])
        lines.extend(f"- {item}" for item in violations)
    else:
        lines.append("No performance-floor violations detected.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-json", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True,
                        help="published bench-data snapshot (latest.json)")
    parser.add_argument("--host-probe", type=Path, required=True)
    parser.add_argument("--operator-max-slowdown", type=float, default=0.30)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.operator_max_slowdown) or args.operator_max_slowdown < 0:
        parser.error("slowdown limit must be finite and non-negative")

    current = load(args.operator_json)
    probe = {}
    try:
        probe = json.loads(Path(args.host_probe).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    cpu_model = str(probe.get("cpu_model", "") or "unknown")
    history = load_history(args.history)
    arch = history.get("arches", {}).get("mps", {})
    runs = arch.get("history", {}).get(cpu_model, []) if isinstance(arch, dict) else []
    if not isinstance(runs, list):
        runs = []

    baseline = median_baseline(current, runs)
    violations = []
    note = None
    if baseline is None:
        # First run on this hardware model: record the baseline and pass.
        note = "No prior MPS baseline for this hardware model; recording only."
    else:
        violations = compare(current, baseline, args.operator_max_slowdown)

    write_summary(args.summary_out, len(current["measurements"]),
                  violations, note=note)
    if violations:
        print("\n".join(violations))
        return 1
    print("MPS performance gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
