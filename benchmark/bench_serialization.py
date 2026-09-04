#!/usr/bin/env python3
"""Measure checkpoint save/load throughput for TensorPlay and an optional peer runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _measure(result, runtime, operation, path, callback, reps, warmups):
    _record(
        result,
        runtime,
        operation,
        path,
        _timed(callback, reps, warmups),
    )


def run(args):
    state = _build_state(args.size_mb, args.tensors)
    peer_state = _to_peer(state)
    results = []
    with tempfile.TemporaryDirectory(prefix="tensorplay-serialization-") as directory:
        root = Path(directory)
        tensorplay_files = {}

        pt_path = root / "tensorplay.pt"
        tensorplay_files["pt"] = pt_path
        _timed(lambda: tp.save(state, pt_path), 1, 0)
        _measure(
            results,
            "tensorplay",
            "save",
            pt_path,
            lambda: tp.save(state, pt_path),
            args.reps,
            args.warmups,
        )
        _measure(
            results,
            "tensorplay",
            "load",
            pt_path,
            lambda: tp.load(pt_path, mmap=args.mmap),
            args.reps,
            args.warmups,
        )

        if "legacy" in args.formats:
            legacy_path = root / "tensorplay.pth"
            tensorplay_files["legacy"] = legacy_path
            save_legacy = lambda: tp.save(
                state, legacy_path, _use_new_zipfile_serialization=False
            )
            load_legacy = lambda: tp.load(legacy_path, mmap=args.mmap)
            _timed(save_legacy, 1, 0)
            _measure(
                results,
                "tensorplay",
                "save",
                legacy_path,
                save_legacy,
                args.reps,
                args.warmups,
            )
            _measure(
                results,
                "tensorplay",
                "load",
                legacy_path,
                load_legacy,
                args.reps,
                args.warmups,
            )

        if "safetensors" in args.formats:
            safe_path = root / "tensorplay.safetensors"
            tensorplay_files["safetensors"] = safe_path
            save_safe = lambda: tp.save(state, safe_path)
            load_safe = lambda: tp.load(safe_path, mmap=args.mmap)
            _timed(save_safe, 1, 0)
            _measure(
                results,
                "tensorplay",
                "save",
                safe_path,
                save_safe,
                args.reps,
                args.warmups,
            )
            _measure(
                results,
                "tensorplay",
                "load",
                safe_path,
                load_safe,
                args.reps,
                args.warmups,
            )

        if "mega" in args.formats:
            mega_path = root / "tensorplay.mega"
            tensorplay_files["mega"] = mega_path
            save_mega = lambda: tp.save(
                state, mega_path, checksum=args.checksum
            )
            load_mega = lambda: tp.load(mega_path, mmap=args.mmap)
            _timed(save_mega, 1, 0)
            _measure(
                results,
                "tensorplay",
                "save",
                mega_path,
                save_mega,
                args.reps,
                args.warmups,
            )
            _measure(
                results,
                "tensorplay",
                "load",
                mega_path,
                load_mega,
                args.reps,
                args.warmups,
            )

        if peer_state is not None and "peer" in args.formats:
            import torch

            peer_files = {}

            def peer_load(path):
                options = {"map_location": "cpu", "weights_only": True}
                if path.suffix == ".pt":
                    options["mmap"] = args.mmap
                return torch.load(path, **options)

            peer_path = root / "peer.pt"
            peer_files["pt"] = peer_path
            save_peer = lambda: torch.save(peer_state, peer_path)
            _timed(save_peer, 1, 0)
            _measure(
                results,
                "peer",
                "save",
                peer_path,
                save_peer,
                args.reps,
                args.warmups,
            )
            _measure(
                results,
                "peer",
                "load",
                peer_path,
                lambda: peer_load(peer_path),
                args.reps,
                args.warmups,
            )

            if "legacy" in args.formats:
                peer_legacy_path = root / "peer.pth"
                peer_files["legacy"] = peer_legacy_path
                save_peer_legacy = lambda: torch.save(
                    peer_state,
                    peer_legacy_path,
                    _use_new_zipfile_serialization=False,
                )
                _timed(save_peer_legacy, 1, 0)
                _measure(
                    results,
                    "peer",
                    "save",
                    peer_legacy_path,
                    save_peer_legacy,
                    args.reps,
                    args.warmups,
                )
                _measure(
                    results,
                    "peer",
                    "load",
                    peer_legacy_path,
                    lambda: peer_load(peer_legacy_path),
                    args.reps,
                    args.warmups,
                )

            for label, peer_file in peer_files.items():
                _measure(
                    results,
                    "tensorplay",
                    "load_peer_file",
                    peer_file,
                    lambda peer_file=peer_file: tp.load(
                        peer_file, mmap=args.mmap
                    ),
                    args.reps,
                    args.warmups,
                )

            for label in ("pt", "legacy"):
                tensorplay_file = tensorplay_files.get(label)
                if tensorplay_file is None:
                    continue
                _measure(
                    results,
                    "peer",
                    "load_tensorplay_file",
                    tensorplay_file,
                    lambda tensorplay_file=tensorplay_file: peer_load(
                        tensorplay_file
                    ),
                    args.reps,
                    args.warmups,
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
        choices=("mega", "safetensors", "legacy", "peer"),
        default=("mega", "safetensors", "legacy", "peer"),
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    rows = run(args)
    print(json.dumps(rows, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
