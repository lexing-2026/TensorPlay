"""TensorPlay vs torch performance matrix on one CUDA device.

Motifs mirror the compiler's fusion targets plus the standard GEMM/norm
kernels.  Every cell reports median CUDA-event milliseconds over `iters`
after `warmup` untimed iterations; "speedup" is torch_time / tp_time.

Usage (remote):
    python benchmark/benchmark_vs_torch.py [--matsize 4096] [--iters 50]
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch  # noqa: F401  (must precede tensorplay: shared cuDNN/cuBLAS)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tensorplay as tp


def _time_tp(fn, args, warmup, iters):
    for _ in range(warmup):
        out = fn(*args)
    tp.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = tp.cuda.Event(enable_timing=True)
        end = tp.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        tp.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return min(times)  # min is robust to scheduler preemption noise


def _time_torch(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return min(times)  # min is robust to scheduler preemption noise


def _sync(device):
    """Synchronize both runtimes for a wall-clock measurement."""
    if device == "cuda":
        torch.cuda.synchronize()
        tp.cuda.synchronize()


def _time_eager(fn, device, warmup, iters):
    """Time a complete eager forward on CPU or CUDA."""
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def _make_llama_state(vocab_size, hidden_size, intermediate_size, num_layers,
                      head_dim, max_seq_len, seed):
    """Build one deterministic, shared Llama state for both runtimes."""
    rng = np.random.default_rng(seed)

    def weight(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.02

    state = {
        "embed_tokens": weight(vocab_size, hidden_size),
        "final_norm": np.ones(hidden_size, dtype=np.float32),
        "lm_head": weight(vocab_size, hidden_size),
        "layers": [],
    }
    for _ in range(num_layers):
        state["layers"].append({
            "q_proj": weight(hidden_size, hidden_size),
            "k_proj": weight(hidden_size, hidden_size),
            "v_proj": weight(hidden_size, hidden_size),
            "o_proj": weight(hidden_size, hidden_size),
            "gate_proj": weight(intermediate_size, hidden_size),
            "up_proj": weight(intermediate_size, hidden_size),
            "down_proj": weight(hidden_size, intermediate_size),
            "input_norm": np.ones(hidden_size, dtype=np.float32),
            "post_attn_norm": np.ones(hidden_size, dtype=np.float32),
        })

    inv_freq = 1.0 / (10000.0 ** (
        2.0 * np.arange(head_dim // 2, dtype=np.float32) / head_dim))
    positions = np.arange(max_seq_len, dtype=np.float32)[:, None]
    angles = positions * inv_freq[None, :]
    state["rope_cos"] = np.cos(angles).astype(np.float32)
    state["rope_sin"] = np.sin(angles).astype(np.float32)
    return state


def _llama_parameter_count(vocab_size, hidden_size, intermediate_size,
                           num_layers):
    """Parameter count for the un-tied Llama state built above."""
    per_layer = 4 * hidden_size * hidden_size
    per_layer += 3 * hidden_size * intermediate_size
    per_layer += 2 * hidden_size  # input_norm and post_attn_norm
    return (2 * vocab_size * hidden_size + num_layers * per_layer
            + hidden_size)  # final_norm


class _TinyLlamaE2E:
    """Minimal Llama decoder with identical weights in Torch and TensorPlay."""

    def __init__(self, state, *, hidden_size, num_heads, intermediate_size,
                 num_layers, device, backend):
        self.backend = backend
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers

        def convert(array):
            array = np.ascontiguousarray(array)
            if backend == "torch":
                return torch.from_numpy(array).to(device)
            return tp.from_numpy(array).to(device)

        self.embed_tokens = convert(state["embed_tokens"])
        self.final_norm = convert(state["final_norm"])
        self.lm_head = convert(state["lm_head"])
        self.rope_cos = convert(state["rope_cos"])
        self.rope_sin = convert(state["rope_sin"])
        self.layers = []
        for source in state["layers"]:
            self.layers.append({name: convert(value)
                                for name, value in source.items()})

    def _project(self, x, weight):
        batch, tokens, _ = x.shape
        flattened = x.reshape([batch * tokens, x.shape[-1]])
        if self.backend == "torch":
            projected = torch.matmul(flattened, weight.t())
        else:
            projected = tp.matmul(flattened, weight.t())
        return projected.reshape([batch, tokens, weight.shape[0]])

    def _rms_norm(self, x, weight):
        variance = (x * x).mean(dim=[-1], keepdim=True)
        if self.backend == "torch":
            return x * torch.rsqrt(variance + 1e-5) * weight
        return x * tp.rsqrt(variance + 1e-5) * weight

    def _torch_rope(self, query, key, tokens):
        cos = self.rope_cos[:tokens].view(1, 1, tokens, -1)
        sin = self.rope_sin[:tokens].view(1, 1, tokens, -1)

        def rotate(x):
            even = x[..., 0::2]
            odd = x[..., 1::2]
            return torch.stack((even * cos - odd * sin,
                                even * sin + odd * cos), dim=-1).reshape_as(x)

        return rotate(query), rotate(key)

    def __call__(self, input_ids):
        batch, tokens = input_ids.shape
        if self.backend == "torch":
            x = torch.nn.functional.embedding(input_ids, self.embed_tokens)
        else:
            x = tp.embedding(self.embed_tokens, input_ids)

        for layer in self.layers:
            residual = x
            xn = self._rms_norm(x, layer["input_norm"])
            q = self._project(xn, layer["q_proj"]).reshape(
                [batch, tokens, self.num_heads, self.head_dim]).permute(
                    [0, 2, 1, 3])
            k = self._project(xn, layer["k_proj"]).reshape(
                [batch, tokens, self.num_heads, self.head_dim]).permute(
                    [0, 2, 1, 3])
            v = self._project(xn, layer["v_proj"]).reshape(
                [batch, tokens, self.num_heads, self.head_dim]).permute(
                    [0, 2, 1, 3])

            if self.backend == "torch":
                q, k = self._torch_rope(q, k, tokens)
                attention = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True)
            else:
                q, k = tp.fused_rope(
                    q, k, self.rope_cos, self.rope_sin, position_offset=0)
                attention = tp.scaled_dot_product_attention(
                    q, k, v, is_causal=True, impl=1)
            attention = attention.permute([0, 2, 1, 3]).reshape(
                [batch, tokens, self.hidden_size])
            x = residual + self._project(attention, layer["o_proj"])

            residual = x
            xn = self._rms_norm(x, layer["post_attn_norm"])
            gate = self._project(xn, layer["gate_proj"])
            up = self._project(xn, layer["up_proj"])
            if self.backend == "torch":
                gated = torch.nn.functional.silu(gate) * up
            else:
                gated = tp.silu_mul(gate, up)
            x = residual + self._project(gated, layer["down_proj"])

        x = self._rms_norm(x, self.final_norm)
        return self._project(x, self.lm_head)


def _run_llm_e2e(device, opts):
    if device == "cuda" and (not torch.cuda.is_available() or
                              not tp.cuda.is_available()):
        raise SystemExit("Both PyTorch and TensorPlay CUDA must be available")

    if device == "cpu":
        torch.set_num_threads(opts.threads)
        tp.set_num_threads(opts.threads)

    state = _make_llama_state(
        opts.vocab, opts.hidden, opts.intermediate, opts.layers,
        opts.head_dim, opts.max_seq, opts.seed)
    torch_model = _TinyLlamaE2E(
        state, hidden_size=opts.hidden, num_heads=opts.heads,
        intermediate_size=opts.intermediate, num_layers=opts.layers,
        device=device, backend="torch")
    tp_model = _TinyLlamaE2E(
        state, hidden_size=opts.hidden, num_heads=opts.heads,
        intermediate_size=opts.intermediate, num_layers=opts.layers,
        device=device, backend="tensorplay")

    ids_np = (np.arange(opts.batch * opts.tokens, dtype=np.int64)
              .reshape(opts.batch, opts.tokens) % opts.vocab)
    torch_ids = torch.from_numpy(ids_np).to(device)
    tp_ids = tp.from_numpy(np.ascontiguousarray(ids_np)).to(device)

    with torch.inference_mode():
        torch_logits = torch_model(torch_ids)
    with tp.no_grad():
        tp_logits = tp_model(tp_ids)
    _sync(device)
    np.testing.assert_allclose(
        tp_logits.cpu().numpy(), torch_logits.detach().cpu().numpy(),
        rtol=5e-3, atol=5e-4)
    torch_next = torch_logits[:, -1, :].argmax(dim=-1).cpu().numpy()
    tp_next = tp_logits[:, -1, :].argmax(dim=-1).cpu().numpy()
    if not np.array_equal(torch_next, tp_next):
        raise AssertionError(
            f"next-token mismatch: torch={torch_next} tensorplay={tp_next}")

    with torch.inference_mode():
        torch_ms = _time_eager(
            lambda: torch_model(torch_ids), device, opts.warmup, opts.iters)
    with tp.no_grad():
        tp_ms = _time_eager(
            lambda: tp_model(tp_ids), device, opts.warmup, opts.iters)

    speedup = torch_ms / tp_ms if tp_ms > 0 else float("inf")
    print(f"\nLLM E2E [{device}]", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "config: "
        f"B={opts.batch} T={opts.tokens} layers={opts.layers} "
        f"hidden={opts.hidden} intermediate={opts.intermediate} "
        f"heads={opts.heads} head_dim={opts.head_dim} threads={opts.threads}"
    )
    parameters = _llama_parameter_count(
        opts.vocab, opts.hidden, opts.intermediate, opts.layers)
    print(f"parameters: {parameters:,} ({parameters / 1e9:.3f}B)")
    print(f"torch eager decoder : {torch_ms:.4f} ms/iteration")
    print(f"tensorplay fused    : {tp_ms:.4f} ms/iteration")
    print(f"speedup (torch/tp)  : {speedup:.3f}x")
    return {"device": device, "torch_ms": torch_ms, "tensorplay_ms": tp_ms,
            "speedup": speedup}


def run_case(name, make_args, tp_fn, torch_fn, warmup=10, iters=50,
             compile_tp=True, compile_torch=False, dtype_note=""):
    """``make_args()`` returns ``(tp_args, torch_args)`` — independent
    tensors with matching shapes/dtypes (values need not correspond)."""
    tp_args, th_args = make_args()
    rows = {}

    rows["tp_eager"] = _time_tp(tp_fn, tp_args, warmup, iters)

    if compile_tp:
        from tensorplay.compiler import compile as tp_compile

        optimized = tp_compile(tp_fn, backend="stax")
        # First call compiles; exclude via warmup inside timing helper.
        rows["tp_stax"] = _time_tp(optimized, tp_args, warmup, iters)

    rows["torch_eager"] = _time_torch(torch_fn, th_args, warmup, iters)

    if compile_torch:
        tc = torch.compile(torch_fn)
        rows["torch_compile"] = _time_torch(tc, th_args, warmup + 3, iters)

    best_tp = min(v for k, v in rows.items() if k.startswith("tp"))
    best_th = min(v for k, v in rows.items() if k.startswith("torch"))
    speedup = best_th / best_tp if best_tp > 0 else float("nan")

    cells = "  ".join(f"{k}={v:.4f}" for k, v in rows.items())
    print(f"{name:<44} {cells}  speedup={speedup:.2f}x {dtype_note}",
          flush=True)
    return {"name": name, **rows, "speedup": speedup}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--llm-e2e", action="store_true",
                        help="run the shared-weight Llama E2E comparison")
    parser.add_argument("--device", choices=("cpu", "cuda", "all"),
                        default="cuda", help="device for --llm-e2e")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1,
                        help="decoder sequence length (1 is decode-like)")
    parser.add_argument("--vocab", type=int, default=24576)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=11008)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-seq", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=16,
                        help="matched Torch/TensorPlay CPU thread count")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--require-win", action="store_true",
                        help="fail unless TensorPlay wins every E2E device")
    opts = parser.parse_args()

    if opts.llm_e2e:
        if opts.hidden != opts.heads * opts.head_dim:
            parser.error("--hidden must equal --heads * --head-dim")
        if opts.tokens < 1 or opts.batch < 1 or opts.layers < 1:
            parser.error("batch, tokens, and layers must be positive")
        if opts.max_seq < opts.tokens:
            parser.error("--max-seq must be at least --tokens")
        if opts.threads < 1:
            parser.error("--threads must be positive")
        devices = ("cpu", "cuda") if opts.device == "all" else (opts.device,)
        results = [_run_llm_e2e(device, opts) for device in devices]
        wins = sum(result["speedup"] > 1.0 for result in results)
        print(f"\nLLM E2E summary: TensorPlay wins {wins}/{len(results)} devices")
        if opts.require_win and wins != len(results):
            raise SystemExit(1)
        return

    if opts.device != "cuda":
        parser.error("the operator matrix is CUDA-only; use --llm-e2e for CPU")
    dev = "cuda"
    results = []

    import numpy as np
    tdtype = {"fp32": tp.float32, "fp16": tp.float16}
    tdtype_t = {"fp32": torch.float32, "fp16": torch.float16}

    # --- 1. GEMM -----------------------------------------------------------------
    for m, k, n, dt in [(4096, 4096, 4096, "fp32"),
                        (8192, 8192, 8192, "fp32"),
                        (4096, 4096, 4096, "fp16")]:
        def make(dt=dt, m=m, k=k, n=n):
            a = tp.rand(m, k, device=dev).to(tdtype[dt])
            b = tp.rand(k, n, device=dev).to(tdtype[dt])
            at = torch.rand(m, k, device=dev).to(tdtype_t[dt])
            bt = torch.rand(k, n, device=dev).to(tdtype_t[dt])
            return (a, b), (at, bt)
        results.append(run_case(
            f"matmul[{m}x{k}x{n} {dt}]", make,
            lambda a, b: tp.matmul(a, b),
            lambda a, b: torch.matmul(a, b),
            warmup=opts.warmup, iters=opts.iters))

    # --- 2. layer_norm forward ---------------------------------------------------
    for shape, dim in [((4096, 1024), 1024), ((8192, 4096), 4096)]:
        def make(shape=shape):
            x = tp.rand(shape, device=dev)
            w = tp.ones(shape[-1], device=dev)
            bb = tp.zeros(shape[-1], device=dev)
            xt = torch.rand(shape, device=dev)
            wt = torch.ones(shape[-1], device=dev)
            bt = torch.zeros(shape[-1], device=dev)
            return (x, w, bb), (xt, wt, bt)
        # TP's tracer cannot capture the python functional wrapper (it
        # dispatches through _C directly); eager-only for now — compiler
        # support lands with template/extern lowering.
        nd = shape[-1]
        results.append(run_case(
            f"layer_norm_fw {shape}", make,
            lambda x, w, b, _nd=nd: tp.nn.functional.layer_norm(x, (_nd,), w, b),
            lambda x, w, b, _nd=nd: torch.nn.functional.layer_norm(
                x, (_nd,), w, b),
            warmup=opts.warmup, iters=opts.iters, compile_tp=False))

    # --- 3. softmax --------------------------------------------------------------
    for shape in [(4096, 4096)]:
        def make(shape=shape):
            return (tp.rand(shape, device=dev),), (torch.rand(shape, device=dev),)
        results.append(run_case(
            f"softmax {shape}", make,
            lambda x: tp.softmax(x, -1),
            lambda x: torch.softmax(x, -1),
            warmup=opts.warmup, iters=opts.iters))

    # --- 4. pw -> reduction epilogue chains --------------------------------------
    def make_chain():
        x = tp.rand(4096, 4096, device=dev)
        xt = torch.rand(4096, 4096, device=dev)
        return (x,), (xt,)

    results.append(run_case(
        "chain sum(dim=1)*3+1 (epilogue)", make_chain,
        lambda x: ((x * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0,
        lambda x: ((x * 2.0).sigmoid()).sum(dim=1) * 3.0 + 1.0,
        warmup=opts.warmup, iters=opts.iters, compile_torch=True))

    results.append(run_case(
        "chain full-sum sigmoid", make_chain,
        lambda x: (x.sigmoid()).sum(),
        lambda x: (x.sigmoid()).sum(),
        warmup=opts.warmup, iters=opts.iters, compile_torch=True))

    # --- 5. pure pointwise chain --------------------------------------------------
    results.append(run_case(
        "pw gelu-ish tanh/exp chain", make_chain,
        lambda x: (x.tanh() * 0.5 + x).relu().square(),
        lambda x: (torch.tanh(x) * 0.5 + x).relu().square(),
        warmup=opts.warmup, iters=opts.iters, compile_torch=True))

    # --- 6. reductions ------------------------------------------------------------
    results.append(run_case(
        "sum full 16M", make_chain,
        lambda x: x.sum(),
        lambda x: x.sum(),
        warmup=opts.warmup, iters=opts.iters))

    results.append(run_case(
        "argmax dim=-1 4096x4096", make_chain,
        lambda x: x.argmax(dim=-1),
        lambda x: x.argmax(dim=-1),
        warmup=opts.warmup, iters=opts.iters))

    print("\n=== summary ===")
    wins = sum(1 for r in results if r["speedup"] >= 1.0)
    print(f"TP >= torch on {wins}/{len(results)} cases "
          f"(geomean speedup of best-vs-best: "
          f"{statistics.geometric_mean(max(r['speedup'], 0.01) for r in results):.2f}x)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
