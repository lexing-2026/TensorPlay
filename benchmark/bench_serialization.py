#!/usr/bin/env python3
"""Measure checkpoint save/load throughput for TensorPlay and an optional peer runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import tensorplay as tp


def _build_state(size_mb: float, tensor_count: int):
    if size_mb <= 0 or tensor_count <= 0:
        raise ValueError("size_mb and tensor_count must be positive")
    total_bytes = int(size_mb * 1024 * 1024)
    count = max(1, total_bytes // 4)
    per_tensor = max(1, (count + tensor_count - 1) // tensor_count)
    states = {}
    for index in range(tensor_count):
        values = np.arange(per_tensor, dtype=np.float32) + index
        states[f"tensor_{index:04d}"] = tp.from_numpy(values)
    return states


def _to_peer(state):
    try:
        import torch
    except Exception:
        return None
    return {
        name: torch.from_numpy(np.asarray(tensor.numpy()))
        for name, tensor in state.items()
    }


def _timed(callable_, reps: int, warmups: int):
    for _ in range(warmups):
        callable_()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - started)
    return {
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "mean_s": statistics.fmean(samples),
    }


def _record(result, runtime, operation, path, timings):
    size = path.stat().st_size if path.exists() else 0
    throughput = (
        size / (1024.0 * 1024.0) / timings["median_s"]
        if timings["median_s"] > 0
        else 0.0
    )
    result.append(
        {
            "runtime": runtime,
            "operation": operation,
            "format": path.suffix,
            "file_bytes": size,
            "throughput_mib_s": throughput,
            **timings,
        }
    )


def run(args):
    state = _build_state(args.size_mb, args.tensors)
    peer_state = _to_peer(state)
    results = []
    with tempfile.TemporaryDirectory(prefix="tensorplay-serialization-") as directory:
        root = Path(directory)
        tp_path = root / "tensorplay.pt"
        _timed(lambda: tp.save(state, tp_path), 1, 0)
        _record(
            results,
            "tensorplay",
            "save",
            tp_path,
            _timed(lambda: tp.save(state, tp_path), args.reps, args.warmups),
        )
        _record(
            results,
            "tensorplay",
            "load",
            tp_path,
            _timed(lambda: tp.load(tp_path, mmap=args.mmap), args.reps, args.warmups),
        )

        if "mega" in args.formats:
            mega_path = root / "tensorplay.mega"
            _timed(lambda: tp.save(state, mega_path, checksum=args.checksum), 1, 0)
            _record(
                results,
                "tensorplay",
                "save",
                mega_path,
                _timed(
                    lambda: tp.save(state, mega_path, checksum=args.checksum),
                    args.reps,
                    args.warmups,
                ),
            )
            _record(
                results,
                "tensorplay",
                "load",
                mega_path,
                _timed(
                    lambda: tp.load(mega_path, mmap=args.mmap),
                    args.reps,
                    args.warmups,
                ),
            )

        if peer_state is not None and "peer" in args.formats:
            import torch

            peer_path = root / "peer.pt"
            torch.save(peer_state, peer_path)
            _record(
                results,
                "peer",
                "save",
                peer_path,
                _timed(lambda: torch.save(peer_state, peer_path), args.reps, args.warmups),
            )
            _record(
                results,
                "peer",
                "load",
                peer_path,
                _timed(
                    lambda: torch.load(
                        peer_path, map_location="cpu", weights_only=True, mmap=args.mmap
                    ),
                    args.reps,
                    args.warmups,
                ),
            )
            _record(
                results,
                "tensorplay",
                "load_peer_file",
                peer_path,
                _timed(
                    lambda: tp.load(peer_path, mmap=args.mmap),
                    args.reps,
                    args.warmups,
                ),
            )
            _record(
                results,
                "peer",
                "load_tensorplay_file",
                tp_path,
                _timed(
                    lambda: torch.load(
                        tp_path, map_location="cpu", weights_only=True, mmap=args.mmap
                    ),
                    args.reps,
                    args.warmups,
                ),
            )

        gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mb", type=float, default=256.0)
    parser.add_argument("--tensors", type=int, default=16)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument(
        "--checksum",
        choices=("none", "crc32", "sha256"),
        default="crc32",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("mega", "peer"),
        default=("mega", "peer"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = run(args)
    print(json.dumps(rows, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
