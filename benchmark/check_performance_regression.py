#!/usr/bin/env python3
"""Compare benchmark results with versioned historical results."""

import argparse
import json
import math
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
    if set(current_by_name) != set(baseline_by_name):
        raise ValueError(f"{label}: measurement names do not match")

    violations = []
    limit = 1.0 + max_slowdown
    for name, current_item in current_by_name.items():
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
            if ratio > limit:
                violations.append(
                    f"{label}/{name}/{metric}: {ratio:.2f}x of baseline "
                    f"(limit {limit:.2f}x)"
                )
    return violations


def write_summary(path, sections, violations):
    lines = ["# CPU performance gate", ""]
    for title, limit, count in sections:
        lines.append(f"- {title}: max {limit:.0%} slower; {count} measurements")
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
    parser.add_argument("--operator-baseline-json", type=Path, required=True)
    parser.add_argument("--operator-f64-json", type=Path)
    parser.add_argument("--operator-f64-baseline-json", type=Path)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--train-baseline-json", type=Path, required=True)
    parser.add_argument("--stax-json", type=Path, required=True)
    parser.add_argument("--stax-baseline-json", type=Path, required=True)
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
    operator_baseline = load(args.operator_baseline_json)
    if (args.operator_f64_json is None) != (args.operator_f64_baseline_json is None):
        parser.error("operator f64 result and baseline must be supplied together")
    operator_f64 = (
        load(args.operator_f64_json) if args.operator_f64_json is not None else None
    )
    operator_f64_baseline = (
        load(args.operator_f64_baseline_json)
        if args.operator_f64_baseline_json is not None
        else None
    )
    train = load(args.train_json)
    train_baseline = load(args.train_baseline_json)
    stax = load(args.stax_json)
    stax_baseline = load(args.stax_baseline_json)

    violations = []
    violations.extend(compare(
        operator, operator_baseline, ("seconds",),
        args.operator_max_slowdown, "operator",
    ))
    if operator_f64 is not None and operator_f64_baseline is not None:
        violations.extend(compare(
            operator_f64, operator_f64_baseline, ("seconds",),
            args.operator_f64_max_slowdown, "operator f64",
        ))
    violations.extend(compare(
        train, train_baseline, ("forward_seconds", "step_seconds"),
        args.train_max_slowdown, "train",
    ))
    violations.extend(compare(
        stax, stax_baseline, ("compiled_seconds",),
        args.stax_max_slowdown, "stax",
    ))
    violations.extend(compare(
        stax, stax_baseline, ("compile_seconds",),
        args.stax_compile_max_slowdown, "stax compile",
    ))
    sections = [
        ("operator", args.operator_max_slowdown, len(operator["measurements"])),
    ]
    if operator_f64 is not None:
        sections.append((
            "operator f64", args.operator_f64_max_slowdown,
            len(operator_f64["measurements"]),
        ))
    sections.extend((
        ("train", args.train_max_slowdown, len(train["measurements"])),
        ("stax compiled", args.stax_max_slowdown, len(stax["measurements"])),
        ("stax compile", args.stax_compile_max_slowdown, len(stax["measurements"])),
    ))
    write_summary(args.summary_out, sections, violations)
    if violations:
        print("CPU performance baseline check failed:")
        for item in violations:
            print(f"  {item}")
        return 1
    print("CPU performance baseline check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
