#!/usr/bin/env python3
"""Collect host hardware facts and a measured matmul throughput anchor.

The measured GFLOP/s number is the practical ceiling that on-site
"percent of machine peak" utilization bars are computed against: it comes
from the same process, thread count, and BLAS path as the benchmark
results, so the ratio is directly meaningful.

Usage: host_probe.py --json-out host.json [--threads N]
"""

import argparse
import json
import multiprocessing
import platform
import re
import time
from pathlib import Path

import tensorplay as tp


def _cpu_model():
    if platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            match = re.search(r"model name\s*:\s*(.+)", text)
            if match:
                return match.group(1).strip()
        except OSError:
            pass
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or "unknown"


def _memory_bytes():
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return 0


def _measure_mem_bw_gbps(threads):
    """Effective DRAM streaming bandwidth via a large scalar multiply.

    Cache-tier anchors proved too noisy across machines to publish as
    utilization denominators; only the DRAM-tier figure is recorded, as a
    coarse context metric rather than a per-op reference.
    """
    if threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(threads)
    n = 1 << 27  # 128M f32 elements -> 512 MiB per tensor
    a = tp.randn(n)
    c = a * 1.000001
    best = float("inf")
    for _ in range(5):
        started = time.perf_counter()
        c = a * 1.000001
        best = min(best, time.perf_counter() - started)
    del a, c
    return round((2 * n * 4) / best / 1e9, 2)


def _measure_mm_gflops(threads):
    """Achievable matmul throughput on this host.

    Returns (peak, realistic): the best over repeated 2048^3 runs as the
    ceiling, and the best 1024^3 run as a realistic mid-size anchor that
    accounts for cache pressure at typical training shapes.
    """
    if threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(threads)
    best = {"1024": float("inf"), "2048": float("inf")}
    for size in (1024, 2048):
        a = tp.randn(size, size)
        b = tp.randn(size, size)
        c = a @ b
        for _ in range(8):
            started = time.perf_counter()
            c = a @ b
            elapsed = time.perf_counter() - started
            best[str(size)] = min(best[str(size)], elapsed)
        del a, b, c
    peak = (2 * 1024 ** 3) / best["1024"] / 1e9
    realistic = (2 * 2048 ** 3) / best["2048"] / 1e9
    return peak, realistic


# Training-task anchors so benchmark numbers translate into "how long would
# this take". GFLOP-per-unit figures use the standard 6*N (fwd+bwd) rule for
# transformers and 3x-forward for ResNet-style vision training.
TASK_ANCHORS = [
    {
        "model": "ResNet-18",
        "workload": "ImageNet-1k, 90 epochs",
        "unit": "img/s",
        "unit_work_gflop": 5.4,
        "work_units": 1.28e6 * 90,
    },
    {
        "model": "GPT-2 124M",
        "workload": "1B training tokens",
        "unit": "tok/s",
        "unit_work_gflop": 0.744,
        "work_units": 1e9,
    },
    {
        "model": "Llama-2 7B",
        "workload": "1B training tokens",
        "unit": "tok/s",
        "unit_work_gflop": 40.4,
        "work_units": 1e9,
    },
]


def _estimate_tasks(realistic_gflops, peak_gflops):
    tasks = []
    for anchor in TASK_ANCHORS:
        work = anchor["unit_work_gflop"]
        realistic_rate = realistic_gflops / work
        peak_rate = peak_gflops / work
        tasks.append({
            **anchor,
            "throughput_realistic": round(realistic_rate, 1),
            "throughput_peak": round(peak_rate, 1),
            "work_days_realistic": round(anchor["work_units"] / realistic_rate / 86400, 2),
            "work_days_peak": round(anchor["work_units"] / peak_rate / 86400, 2),
        })
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    threads = args.threads or getattr(tp, "get_num_threads", lambda: 0)() or multiprocessing.cpu_count()
    peak_gflops, realistic_gflops = _measure_mm_gflops(threads)
    mem_bw_gbps = _measure_mem_bw_gbps(threads)

    payload = {
        "schema_version": 1,
        "benchmark": "host-probe",
        "cpu_model": _cpu_model(),
        "host_arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "logical_cores": multiprocessing.cpu_count(),
        "threads": threads,
        "memory_bytes": _memory_bytes(),
        "measured_mm_gflops": round(peak_gflops, 2),
        "measured_mm_gflops_realistic": round(realistic_gflops, 2),
        "measured_mem_bw_gbps": mem_bw_gbps,
        "task_estimates": _estimate_tasks(realistic_gflops, peak_gflops),
        "python": platform.python_version(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"host: {payload['cpu_model']} | {threads}T | "
        f"peak {peak_gflops:.1f} / realistic {realistic_gflops:.1f} GFLOP/s | "
        f"DRAM {mem_bw_gbps:.1f} GB/s"
    )


if __name__ == "__main__":
    main()
