#!/usr/bin/env python3
"""Merge benchmark result JSONs into a multi-arch site snapshot.

Usage: publish_bench_data.py <results-dir> <output-file> [--arch NAME] [--merge FILE]

The results directory holds the JSON files produced by the CI benchmark
suites. Each run updates one architecture section inside the output file:
``--merge`` carries sections from previous runs forward so that arches
published by different workflows coexist in a single snapshot. The output
keeps a stable structure (schema_version 3 with an ``arches`` map) and, per
architecture, a per-runner history of the most recent runs so the
performance gate can compare a run only against the same hardware model.
"""

import argparse
import datetime
import json
import os
import platform
import sys

HISTORY_LIMIT = 5


def load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("output_file")
    parser.add_argument("--arch", default=os.environ.get("TP_BENCH_ARCH", "cpu"))
    parser.add_argument("--merge", default=None, help="previous snapshot to update in place")
    parser.add_argument("--baselines-dir", default=None, help="versioned baseline JSONs to embed for on-site comparison")
    parser.add_argument("--host-probe", default=None, help="host_probe.py JSON with hardware facts and measured peak")
    args = parser.parse_args()

    repository = os.environ.get("GITHUB_REPOSITORY", "lexing-2026/TensorPlay")
    run_id = os.environ.get("GITHUB_RUN_ID", "")

    suites = {}
    for name in sorted(os.listdir(args.results_dir)):
        if not name.endswith(".json"):
            continue
        payload = load_json(os.path.join(args.results_dir, name))
        if payload is not None:
            suites[name[: -len(".json")]] = payload

    if not suites:
        print("no benchmark JSON files found", file=sys.stderr)
        return 1

    probe = {}
    if args.host_probe:
        probe = load_json(args.host_probe) or {}
    cpu_model = str(probe.get("cpu_model", "") or "unknown")

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source = {
        "repository": repository,
        "run_id": run_id,
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}"
        if run_id
        else "",
        "commit": os.environ.get("GITHUB_SHA", ""),
        "commit_url": f"https://github.com/{repository}/commit/{os.environ.get('GITHUB_SHA', '')}"
        if os.environ.get("GITHUB_SHA")
        else "",
        "ref": os.environ.get("GITHUB_REF", ""),
        "event": os.environ.get("GITHUB_EVENT_NAME", ""),
        "runner_image": os.environ.get("ImageOS", ""),
        "arch": args.arch,
        "host_arch": platform.machine(),
    }
    run_snapshot = {
        "generated_at": generated_at,
        "source": source,
        "suites": suites,
    }

    snapshot = {"schema_version": 3, "arches": {}}
    previous_arch = {}
    if args.merge:
        previous = load_json(args.merge)
        if isinstance(previous, dict):
            if isinstance(previous.get("arches"), dict):
                snapshot["arches"] = previous["arches"]
                previous_arch = previous["arches"].get(args.arch) or {}
            elif isinstance(previous.get("suites"), dict):
                # Schema v1 snapshot: treat its data as a single-arch entry.
                arch = previous.get("source", {}).get("arch") or "cpu"
                previous_arch = {
                    "generated_at": previous.get("generated_at", ""),
                    "source": previous.get("source", {}),
                    "suites": previous["suites"],
                }
                snapshot["arches"][arch] = previous_arch
    if not isinstance(previous_arch, dict):
        previous_arch = {}

    history = previous_arch.get("history", {})
    if not isinstance(history, dict):
        history = {}
    cpu_history = history.get(cpu_model, [])
    if not isinstance(cpu_history, list):
        cpu_history = []
    cpu_history.append(run_snapshot)
    cpu_history = cpu_history[-HISTORY_LIMIT:]
    history[cpu_model] = cpu_history

    arch_snapshot = {
        "generated_at": generated_at,
        "source": source,
        "suites": suites,
        "baselines": previous_arch.get("baselines", {}),
        "host": probe,
        "history": history,
    }

    if args.baselines_dir:
        embedded = {}
        for name in sorted(os.listdir(args.baselines_dir)):
            if not name.endswith(".json"):
                continue
            payload = load_json(os.path.join(args.baselines_dir, name))
            if payload is not None:
                embedded[name[: -len(".json")]] = payload
        arch_snapshot["baselines"] = embedded

    snapshot["arches"][args.arch] = arch_snapshot

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {args.output_file} arches={list(snapshot['arches'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
