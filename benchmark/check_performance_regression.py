#!/usr/bin/env python3
"""Compare benchmark results with versioned historical results."""

import argparse
import json
import math
import statistics
import sys
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
    if not payload.get("benchmark"):
        raise ValueError(f"{path} has no benchmark name")
    threads = payload.get("threads")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError(f"{path} has invalid thread count")
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


def median_baseline(current, runs, suite_key):
    """Build a comparison baseline for one suite as the per-measurement
    median over the matching runs of the same hardware history."""
    matching = []
    for run in runs:
        payload = run.get("suites", {}).get(suite_key)
        if not isinstance(payload, dict):
            continue
        if payload.get("benchmark") != current.get("benchmark"):
            continue
        if payload.get("suite") != current.get("suite"):
            continue
        if payload.get("dtype") != current.get("dtype"):
            continue
        if payload.get("threads") != current.get("threads"):
            continue
        matching.append(payload)
    if not matching:
        return None
    base = dict(current)
    base["measurements"] = []
    for current_item in current["measurements"]:
        baseline_item = dict(current_item)
        for metric in ("seconds", "forward_seconds", "step_seconds",
                       "compiled_seconds", "compile_seconds"):
            if metric not in current_item:
                continue
            values = []
            for payload in matching:
                for item in payload["measurements"]:
                    if item.get("name") == current_item["name"] and metric in item:
                        value = item[metric]
                        if (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(value)
                            and value > 0
                        ):
                            values.append(value)
            if values:
                baseline_item[metric] = statistics.median(values)
        base["measurements"].append(baseline_item)
    return base


def compare(current, baseline, metrics, max_slowdown, label):
    if current["benchmark"] != baseline["benchmark"]:
        raise ValueError(f"{label}: benchmark names do not match")
    if current.get("suite") != baseline.get("suite"):
        raise ValueError(f"{label}: benchmark suites do not match")
    if current.get("threads") != baseline.get("threads"):
        raise ValueError(f"{label}: thread counts do not match")
    for field in ("dtype", "backend"):
        if current.get(field) != baseline.get(field):
            raise ValueError(f"{label}: {field} metadata does not match")

    current_by_name = {item["name"]: item for item in current["measurements"]}
    baseline_by_name = {item["name"]: item for item in baseline["measurements"]}
    # The suite grows as operators gain support for a dtype; names that are
    # only on one side simply have no reference yet or are no longer emitted,
    # so only the intersection is compared.  New names are recorded into the
    # history when this run is published and gate from then on.
    compared_names = [name for name in current_by_name if name in baseline_by_name]

    # A run can land on a faster or slower machine, or on a noisier
    # neighbour, than the one that produced the committed baseline.  Measure
    # the run's own overall shift against the baseline as the median ratio
    # over every shared measurement, then treat only deviations beyond that
    # shift as regressions.  The median stays robust when a handful of
    # operators genuinely regress.
    scale = {}
    for metric in metrics:
        ratios = []
        for name in compared_names:
            current_item = current_by_name[name]
            baseline_item = baseline_by_name[name]
            actual = current_item.get(metric)
            previous = baseline_item.get(metric)
            if (
                isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and isinstance(previous, (int, float)) and not isinstance(previous, bool)
                and math.isfinite(actual) and math.isfinite(previous)
                and actual > 0 and previous > 0
            ):
                ratios.append(actual / previous)
        scale[metric] = statistics.median(ratios) if ratios else 1.0

    violations = []
    for name in compared_names:
        current_item = current_by_name[name]
        baseline_item = baseline_by_name[name]
        if current_item.get("category") != baseline_item.get("category"):
            raise ValueError(f"{label}/{name}: categories do not match")
        identity_fields = (
            "input_shapes",
            "batch_tokens",
            "in_features",
            "out_features",
            "batch",
            "features",
            "hidden",
        )
        for field in identity_fields:
            if current_item.get(field) != baseline_item.get(field):
                raise ValueError(f"{label}/{name}: {field} metadata does not match")
        for metric in metrics:
            actual = current_item.get(metric)
            previous = baseline_item.get(metric)
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                raise ValueError(f"{label}/{name}/{metric}: timing is not numeric")
            if not isinstance(previous, (int, float)) or isinstance(previous, bool):
                raise ValueError(f"{label}/{name}/{metric}: baseline is not numeric")
            if not math.isfinite(actual) or not math.isfinite(previous):
                raise ValueError(
                    f"{label}/{name}/{metric}: timings must be finite"
                )
            if actual <= 0 or previous <= 0:
                raise ValueError(f"{label}/{name}/{metric}: timings must be positive")
            ratio = actual / previous
            # Sub-microsecond baselines sit at the timing floor: the
            # measurement jitter of a tiny tensor makes their relative
            # slowdown noise rather than signal, so leave them out of the
            # ratio gate.
            if previous < 1e-5:
                continue
            limit = scale[metric] * (1.0 + max_slowdown)
            if ratio > limit:
                violations.append(
                    f"{label}/{name}/{metric}: {ratio:.2f}x of baseline "
                    f"(limit {limit:.2f}x)"
                )
    return violations


def write_summary(path, sections, violations, note=None):
    lines = ["# CPU performance gate", ""]
    if note:
        lines.append(note)
        lines.append("")
    for title, limit, count in sections:
        lines.append(f"- {title}: max {limit:.0%} slower; {count} measurements")
    lines.append("")
    if violations:
        lines.extend(["## Regressions", ""])
        lines.extend(f"- {item}" for item in violations)
    else:
        lines.append("No performance-floor violations detected.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json_optional(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-json", type=Path, required=True)
    parser.add_argument("--operator-f64-json", type=Path)
    parser.add_argument("--operator-bf16-json", type=Path)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--stax-json", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True,
                        help="published bench-data snapshot (latest.json)")
    parser.add_argument("--host-probe", type=Path, required=True)
    parser.add_argument("--operator-baseline-json", type=Path,
                        help="legacy committed baseline; used only when no history exists")
    parser.add_argument("--operator-f64-baseline-json", type=Path)
    parser.add_argument("--operator-bf16-baseline-json", type=Path)
    parser.add_argument("--train-baseline-json", type=Path)
    parser.add_argument("--stax-baseline-json", type=Path)
    parser.add_argument("--operator-max-slowdown", type=float, default=0.30)
    parser.add_argument("--operator-f64-max-slowdown", type=float, default=0.30)
    parser.add_argument("--train-max-slowdown", type=float, default=0.20)
    parser.add_argument("--stax-max-slowdown", type=float, default=0.25)
    parser.add_argument("--stax-compile-max-slowdown", type=float, default=0.50)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    limits = (
        args.operator_max_slowdown,
        args.operator_f64_max_slowdown,
        args.train_max_slowdown,
        args.stax_max_slowdown,
        args.stax_compile_max_slowdown,
    )
    if any(not math.isfinite(value) or value < 0 for value in limits):
        parser.error("slowdown limits must be finite and non-negative")

    operator = load(args.operator_json)
    operator_f64 = (
        load(args.operator_f64_json) if args.operator_f64_json is not None else None
    )
    operator_bf16 = (
        load(args.operator_bf16_json) if args.operator_bf16_json is not None else None
    )
    train = load(args.train_json)
    stax = load(args.stax_json)

    probe = _load_json_optional(args.host_probe)
    cpu_model = str(probe.get("cpu_model", "") or "unknown")
    history = load_history(args.history)
    arch = history.get("arches", {}).get("cpu", {})
    runs = arch.get("history", {}).get(cpu_model, []) if isinstance(arch, dict) else []
    if not isinstance(runs, list):
        runs = []

    def baseline_for(current, suite_key, legacy_path):
        baseline = median_baseline(current, runs, suite_key)
        if baseline is not None:
            return baseline
        if legacy_path is not None:
            return load(legacy_path)
        return None

    operator_baseline = baseline_for(
        operator, "cpu-operator", args.operator_baseline_json)
    operator_f64_baseline = (
        baseline_for(operator_f64, "cpu-operator-f64",
                     args.operator_f64_baseline_json)
        if operator_f64 is not None
        else None
    )
    operator_bf16_baseline = (
        baseline_for(operator_bf16, "cpu-operator-bf16",
                     args.operator_bf16_baseline_json)
        if operator_bf16 is not None
        else None
    )
    train_baseline = baseline_for(train, "cpu-train", args.train_baseline_json)
    stax_baseline = baseline_for(stax, "stax-cpu", args.stax_baseline_json)

    violations = []
    sections = []
    for title, current, baseline, metrics, limit in (
        ("operator", operator, operator_baseline, ("seconds",),
         args.operator_max_slowdown),
        ("operator f64", operator_f64, operator_f64_baseline, ("seconds",),
         args.operator_f64_max_slowdown),
        ("operator bf16", operator_bf16, operator_bf16_baseline, ("seconds",),
         args.operator_max_slowdown),
        ("train", train, train_baseline,
         ("forward_seconds", "step_seconds"), args.train_max_slowdown),
        ("stax compiled", stax, stax_baseline, ("compiled_seconds",),
         args.stax_max_slowdown),
        ("stax compile", stax, stax_baseline, ("compile_seconds",),
         args.stax_compile_max_slowdown),
    ):
        if current is None or baseline is None:
            if current is not None:
                sections.append((title, limit, len(current["measurements"])))
            continue
        violations.extend(compare(current, baseline, metrics, limit, title))
        sections.append((title, limit, len(current["measurements"])))

    note = None
    if not runs:
        note = (
            f"No same-hardware history for {cpu_model!r}; this run records "
            f"the first baseline."
        )
    write_summary(args.summary_out, sections, violations, note=note)
    if violations:
        print("CPU performance baseline check failed:")
        for item in violations:
            print(f"  {item}")
        return 1
    print("CPU performance baseline check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
