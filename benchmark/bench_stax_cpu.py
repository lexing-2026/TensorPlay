#!/usr/bin/env python3
"""CPU Stax native compile benchmark with compile and steady-state metrics."""

import argparse
import json
import time
from pathlib import Path

import tensorplay as tp


CASES = (
    ("mm-bias", (64, 256, 512), "linear"),
    ("mm-bias-relu", (128, 512, 1024), "relu"),
    ("mm-bias-sigmoid", (128, 512, 1024), "sigmoid"),
    ("mm-bias-sin", (64, 1024, 1024), "sin"),
    ("mm-bias-tanh", (64, 1024, 1024), "tanh"),
    ("mm-bias-pointwise", (32, 1024, 2048), "pointwise"),
    ("mm-bias-chain", (64, 256, 512), "chain"),
    ("two-linear-relu", (64, 256, 256), "two_linear"),
    ("transpose-mm", (256, 64, 512), "transpose_mm"),
    ("mm-reduce", (128, 512, 1024), "reduce"),
)


def _program(kind):
    if kind == "linear":
        return lambda inputs, weight, bias: inputs @ weight.t() + bias
    if kind == "relu":
        return lambda inputs, weight, bias: (inputs @ weight.t() + bias).relu()
    if kind == "sigmoid":
        return lambda inputs, weight, bias: (inputs @ weight.t() + bias).sigmoid()
    if kind == "sin":
        return lambda inputs, weight, bias: (inputs @ weight.t() + bias).sin()
    if kind == "tanh":
        return lambda inputs, weight, bias: (inputs @ weight.t() + bias).tanh()
    if kind == "pointwise":
        return lambda inputs, weight, bias: (inputs @ weight.t() + bias) * 2 + 1
    if kind == "chain":
        return lambda inputs, weight, bias: (
            (inputs @ weight.t() + bias).relu() * 0.5 + 1
        ).tanh()
    if kind == "two_linear":
        return lambda inputs, weight, bias: (
            (inputs @ weight.t() + bias).relu() @ weight + bias
        )
    if kind == "transpose_mm":
        return lambda inputs, weight, bias: inputs.t() @ weight + bias
    return lambda inputs, weight, bias: (inputs @ weight.t() + bias).sum(dim=1)


def _consume(fn):
    with tp.no_grad():
        output = fn()
        return float(output.sum())


def _time(fn, warmup, reps):
    for _ in range(warmup):
        _consume(fn)
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        _consume(fn)
        samples.append(time.perf_counter() - started)
    return min(samples)


def _codegens(compiled):
    cache = getattr(compiled, "_tensorplay_cache", {})
    codegens = []
    for lowering in cache.values():
        codegen = getattr(lowering, "_tensorplay_codegen", None)
        if not codegen:
            raise RuntimeError("Stax cache entry is missing native codegen metadata")
        codegens.append(str(codegen))
    return codegens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.reps < 1 or args.warmup < 0 or args.threads < 0:
        parser.error(
            "reps must be positive, warmup and threads must be non-negative"
        )
    if args.threads and hasattr(tp, "set_num_threads"):
        tp.set_num_threads(args.threads)

    measurements = []
    for name, (batch, features, hidden), kind in CASES:
        program = _program(kind)
        inputs = tp.randn((batch, features))
        weight = tp.randn((hidden, features))
        bias = tp.randn((hidden,))
        if kind == "transpose_mm":
            inputs = tp.randn((features, batch))
            weight = tp.randn((features, hidden))
        eager = lambda: program(inputs, weight, bias)

        eager_seconds = _time(eager, args.warmup, args.reps)

        started = time.perf_counter()
        compiled = tp.compile(
            program,
            backend="stax",
            mode="default",
            fullgraph=True,
            strict_native=True,
        )
        compile_seconds = time.perf_counter() - started
        compiled_call = lambda: compiled(inputs, weight, bias)

        started = time.perf_counter()
        _consume(compiled_call)
        first_pass_seconds = time.perf_counter() - started
        compiled_seconds = _time(
            compiled_call, args.warmup, args.reps
        )
        codegens = _codegens(compiled)
        if not codegens:
            raise RuntimeError(f"{name} produced no native Stax lowering")

        speedup = eager_seconds / compiled_seconds
        print(
            f"{name:20} eager={eager_seconds * 1e3:9.3f} ms "
            f"compiled={compiled_seconds * 1e3:9.3f} ms "
            f"compile={compile_seconds:9.3f} s "
            f"speedup={speedup:6.2f}x"
        )
        measurements.append({
            "name": name,
            "category": "compile",
            "batch": batch,
            "features": features,
            "hidden": hidden,
            "eager_seconds": eager_seconds,
            "compiled_seconds": compiled_seconds,
            "compile_seconds": compile_seconds,
            "first_pass_seconds": first_pass_seconds,
            "speedup": speedup,
            "codegens": codegens,
        })

    payload = {
        "schema_version": 1,
        "benchmark": "stax-cpu",
        "backend": "stax",
        "threads": args.threads or getattr(tp, "get_num_threads", lambda: 0)(),
        "measurements": measurements,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n",
                             encoding="utf-8")
    print(f"wrote results: {args.json_out}")


if __name__ == "__main__":
    main()
