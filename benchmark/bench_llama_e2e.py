"""

Prefill (one causal pass over prompt) + autoregressive decode loop.
Paired min-of-R; run serially on the GPU box.
"""
import argparse
import time
import torch
import tensorplay as tp

torch.manual_seed(0)

# ~0.606B parameters for this un-tied Llama-shaped checkpoint:
# 2*vocab*hidden + layers*(4*hidden**2 + 3*hidden*inter) + norm scales.
CFG = dict(vocab=24576, hidden=4096, inter=11008, heads=32, head_dim=128,
           layers=2, max_seq=4096)

# Same ~0.606B parameter model, with realistic prompt/context sizes. The
# decode loop intentionally grows the context each step, as an autoregressive
# serving request does; both implementations use the same no-KV-cache workload.
SCENARIOS = {
    "short": (128, 1),
    "medium": (512, 4),
    "long": (2048, 4),
}


def parameter_count(c):
    h = c["hidden"]
    vocab = c["vocab"]
    layers = c["layers"]
    return (2 * vocab * h
            + layers * (4 * h * h + 3 * h * c["inter"])
            + (2 * layers + 1) * h)


class TorchLlama(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c["hidden"]
        self.embed = torch.nn.Embedding(c["vocab"], h)
        self.layers = torch.nn.ModuleList()
        for _ in range(c["layers"]):
            self.layers.append(torch.nn.ModuleDict({
                "q": torch.nn.Linear(h, c["heads"] * c["head_dim"], bias=False),
                "k": torch.nn.Linear(h, c["heads"] * c["head_dim"], bias=False),
                "v": torch.nn.Linear(h, c["heads"] * c["head_dim"], bias=False),
                "o": torch.nn.Linear(c["heads"] * c["head_dim"], h, bias=False),
                "gate": torch.nn.Linear(h, c["inter"], bias=False),
                "up": torch.nn.Linear(h, c["inter"], bias=False),
                "down": torch.nn.Linear(c["inter"], h, bias=False),
                "n1": torch.nn.RMSNorm(h, eps=1e-5),
                "n2": torch.nn.RMSNorm(h, eps=1e-5),
            }))
        self.final = torch.nn.RMSNorm(h, eps=1e-5)
        self.lm_head = torch.nn.Linear(h, c["vocab"], bias=False)
        # benchmark is specifically exercising fused interleaved RoPE, so a
        half_dim = c["head_dim"] // 2
        positions = torch.arange(c["max_seq"], dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(half_dim, dtype=torch.float32) * 2.0 /
            c["head_dim"]
        ))
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(self, ids):
        x = self.embed(ids)
        B, T = ids.shape
        H, D = CFG["heads"], CFG["head_dim"]
        for L in self.layers:
            r = x
            xn = L["n1"](x)
            q = L["q"](xn).view(B, T, H, D).transpose(1, 2)
            k = L["k"](xn).view(B, T, H, D).transpose(1, 2)
            v = L["v"](xn).view(B, T, H, D).transpose(1, 2)
            cos = self.rope_cos[:T].view(1, 1, T, D // 2)
            sin = self.rope_sin[:T].view(1, 1, T, D // 2)
            q_even, q_odd = q[..., 0::2], q[..., 1::2]
            k_even, k_odd = k[..., 0::2], k[..., 1::2]
            q = torch.stack((q_even * cos - q_odd * sin,
                             q_even * sin + q_odd * cos), dim=-1).reshape_as(q)
            k = torch.stack((k_even * cos - k_odd * sin,
                             k_even * sin + k_odd * cos), dim=-1).reshape_as(k)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, H * D)
            x = r + L["o"](a)
            r = x
            xn = L["n2"](x)
            x = r + L["down"](torch.nn.functional.silu(L["gate"](xn)) * L["up"](xn))
        return self.lm_head(self.final(x))


classTPLlama_ready = None


def tp_forward(model, ids, attn_impl=2):
    import math
    c = CFG
    B, T = ids.shape
    h = c["hidden"]; H = c["heads"]; D = c["head_dim"]

    def rms(x, w):
        return tp.rms_norm(x, [x.shape[-1]], w, 1e-5)

    def lin(x, w):
        # The reshape pair is a view; it avoids the batched-matmul slice path
        # for decode's common [1,1,H] shape.
        if len(x.shape) > 2:
            out = tp.matmul(x.reshape([-1, x.shape[-1]]), w)
            return out.reshape(list(x.shape[:-1]) + [w.shape[-1]])
        return tp.matmul(x, w)

    x = tp.embedding(model["embed"], ids)
    for L in model["layers"]:
        res = x
        xn = rms(x, L["n1"])
        q = lin(xn, L["q"]).reshape([B, T, H, D]).permute([0, 2, 1, 3])
        k = lin(xn, L["k"]).reshape([B, T, H, D]).permute([0, 2, 1, 3])
        v = lin(xn, L["v"]).reshape([B, T, H, D]).permute([0, 2, 1, 3])
        # Native fused Q/K RoPE: one dispatch/launch replaces the composite
        # even/odd slice, multiply, add, stack, and reshape sequence.
        _q, _k = tp.fused_rope(q, k, model["cos"], model["sin"])
        # impl=2 is the native GEMM-backed path.  impl=5 selects the original
        # aligned standalone CUTE/CUTLASS flash schedule on CUDA FP16.
        att = tp.scaled_dot_product_attention(
            _q, _k, v, is_causal=True, impl=attn_impl
        )
        att = att.permute([0, 2, 1, 3]).reshape([B, T, H * D])
        x = res + lin(att, L["o"])
        res = x
        xn = rms(x, L["n2"])
        x = res + lin(tp.silu_mul(lin(xn, L["gate"]), lin(xn, L["up"])), L["down"])
    return lin(rms(x, model["final"]), model["lm_head"])


def _dtype_pair(dtype_name):
    if dtype_name == "float32":
        return torch.float32, __import__("numpy").float32
    if dtype_name == "float16":
        return torch.float16, __import__("numpy").float16
    if dtype_name == "bfloat16":
        return torch.bfloat16, __import__("numpy").float32
    raise ValueError(f"unsupported dtype {dtype_name!r}")


def build_tp_model(dev="cuda", dtype_name="float32"):
    import numpy as np
    rng = np.random.default_rng(0)
    _torch_dtype, np_dtype = _dtype_pair(dtype_name)
    c = CFG; h = c["hidden"]; H = c["heads"]; D = c["head_dim"]
    def to_device(t):
        # NumPy has no portable bfloat16 dtype.  Construct those weights as
        if dtype_name == "bfloat16":
            t = t.to(tp.bfloat16)
        return t.to(dev)
    def W(*s):
        return to_device(tp.from_numpy(
            (rng.standard_normal(s) * 0.02).astype(np_dtype)
        ))
    half_dim = c["head_dim"] // 2
    positions = np.arange(c["max_seq"], dtype=np.float32)[:, None]
    inv_freq = 1.0 / (10000.0 ** (
        np.arange(half_dim, dtype=np.float32) * 2.0 / c["head_dim"]
    ))
    angles = positions * inv_freq[None, :]
    rope_dtype = np.float32 if dtype_name == "bfloat16" else np_dtype
    m = {"embed": W(c["vocab"], h), "final": W(h),
         "cos": to_device(tp.from_numpy(
             np.cos(angles).astype(rope_dtype)
         )),
         "sin": to_device(tp.from_numpy(
             np.sin(angles).astype(rope_dtype)
         )),
         "layers": []}
    for _ in range(c["layers"]):
        m["layers"].append({"q": W(h, H*D), "k": W(h, H*D), "v": W(h, H*D),
                            "o": W(H*D, h), "gate": W(h, c["inter"]),
                            "up": W(h, c["inter"]), "down": W(c["inter"], h),
                            "n1": W(h), "n2": W(h)})
    m["lm_head"] = W(h, c["vocab"])
    return m


def best(fn, dev="cuda", R=5):
    fn()
    if dev == "cuda":
        torch.cuda.synchronize(); tp.cuda.synchronize()
    m = 1e18
    for _ in range(R):
        t0 = time.perf_counter(); fn()
        if dev == "cuda":
            torch.cuda.synchronize(); tp.cuda.synchronize()
        m = min(m, (time.perf_counter() - t0) * 1e3)
    return round(m, 2)


def run(fw, dev="cuda", prefill_len=128, decode_steps=24,
        dtype_name="float32", timing_runs=5, attn_impl=2):
    c = CFG
    torch_dtype, _np_dtype = _dtype_pair(dtype_name)
    if fw == "t":
        model = TorchLlama(c).to(dev, dtype=torch_dtype).eval()
        ids = torch.randint(0, c["vocab"], (1, prefill_len), device=dev)
        with torch.no_grad():
            pf = best(lambda: model(ids), dev, timing_runs)
            def dec():
                cur = ids
                for _ in range(decode_steps):
                    cur = torch.cat([cur, model(cur)[:, -1:, :].argmax(-1)], 1)
            dc = best(dec, dev, timing_runs)
    else:
        model = build_tp_model(dev, dtype_name)
        ids = tp.from_numpy(torch.randint(0, c["vocab"], (1, prefill_len)).cpu().numpy())
        ids = ids.to(dev)
        with tp.no_grad():
            pf = best(lambda: tp_forward(model, ids, attn_impl), dev, timing_runs)
            def dec():
                cur = ids
                for _ in range(decode_steps):
                    cur = tp.cat([
                        cur,
                        tp.argmax(tp_forward(model, cur, attn_impl)[:, -1:, :], dim=-1),
                    ], 1)
        with tp.no_grad():
            dc = best(dec, dev, timing_runs)
    ms_per_tok = dc / decode_steps
    return pf, ms_per_tok


def profile_once(dev, prompt_len=128, steps=10, dtype_name="float32",
                 attn_impl=2):
    """Show the existing native dispatcher profile for the TP decode path."""
    model = build_tp_model(dev, dtype_name)
    ids = tp.from_numpy(
        torch.randint(0, CFG["vocab"], (1, prompt_len)).cpu().numpy()
    ).to(dev)

    def dec(n):
        cur = ids
        for _ in range(n):
            cur = tp.cat([
                cur,
                tp.argmax(tp_forward(model, cur, attn_impl)[:, -1:, :], dim=-1),
            ], 1)

    # Warm every context length that will be captured outside the region so
    # one-time GEMM plan selection is not mistaken for steady-state profile
    # cost.  This matters for multi-step decode, where each growing context
    # is a distinct GEMM shape.
    dec(steps)
    with tp.profiler.profile(
        record_shapes=True,
        gpu_timing=(dev == "cuda"),
    ) as prof:
        dec(steps)
    gpu = [ev[8] for ev in prof.events if ev[8] >= 0]
    gpu_total = sum(gpu)
    gpu_max = max(gpu, default=-1.0)
    print(
        f"native profile [{dev}]: events={len(prof.events)} "
        f"gpu_resolved={len(gpu)}/{len(prof.events)} "
        f"gpu_ms_sum={gpu_total:.3f} gpu_ms_max={gpu_max:.3f} "
        f"stop_reclaim_ms={prof.stop_ms:.3f}"
    )
    gpu_by_name = {}
    for event in prof.events:
        gpu_ms = event[8]
        if gpu_ms < 0:
            continue
        row = gpu_by_name.setdefault(event[0], [0, 0.0, 0.0])
        row[0] += 1
        row[1] += gpu_ms
        row[2] = max(row[2], gpu_ms)
    print("GPU op detail: name                         calls   total ms   max ms")
    for name, (calls, total, maximum) in sorted(
        gpu_by_name.items(), key=lambda item: item[1][1], reverse=True
    ):
        print(f"  {name:<38} {calls:5d} {total:10.3f} {maximum:9.3f}")
    print(str(prof.key_averages(sort_by="self_cpu_time"))[:5000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "all"), default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"),
        default="float32", help="model/activation dtype (default: float32)",
    )
    parser.add_argument(
        "--attn-impl", type=int, choices=(2, 5), default=2,
        help="TensorPlay attention implementation (2=GEMM, 5=aligned native flash)",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--scenario", choices=("custom", *SCENARIOS, "all"), default="custom",
        help="sequence workload; 'all' runs short/medium/long",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--timing-runs", type=int, default=5,
        help="timed repeats inside each run after one warmup (default: 5)",
    )
    parser.add_argument("--prefill-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=24)
    parser.add_argument("--profile-steps", type=int, default=10)
    args = parser.parse_args()
    if args.rounds <= 0 or args.timing_runs <= 0:
        raise ValueError("--rounds and --timing-runs must be positive")

    if args.device == "all":
        devices = ("cpu", "cuda")
    else:
        devices = (args.device,)
    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    for dev in devices:
        if args.scenario == "custom":
            cases = [("custom", args.prefill_len, args.decode_steps)]
        elif args.scenario == "all":
            cases = [(name, *lengths) for name, lengths in SCENARIOS.items()]
        else:
            cases = [(args.scenario, *SCENARIOS[args.scenario])]

        params = parameter_count(CFG)
        print(
            f"model params={params:,} ({params / 1e9:.3f}B) "
            f"device={dev} dtype={args.dtype} attn_impl={args.attn_impl}"
        )
        for case_name, prefill_len, decode_steps in cases:
            if prefill_len + decode_steps > CFG["max_seq"]:
                raise ValueError(
                    f"scenario {case_name!r} exceeds max_seq={CFG['max_seq']}"
                )
            if args.profile:
                print(
                    f"scenario={case_name} prefill_len={prefill_len} "
                    f"decode={decode_steps}"
                )
                profile_once(
                    dev, prefill_len, args.profile_steps, args.dtype,
                    args.attn_impl,
                )
                continue

            print(
                f"scenario={case_name} prefill_len={prefill_len} "
                f"decode={decode_steps} steps (interleaved x{args.rounds})"
            )
            res = {"t": [], "p": []}
            for _ in range(args.rounds):
                for fw in ("t", "p"):
                    pf, mt = run(
                        fw, dev, prefill_len, decode_steps, args.dtype,
                        args.timing_runs, args.attn_impl
                    )
                    res[fw].append((pf, mt))
            for fw in ("t", "p"):
                name = "ref  " if fw == "t" else "tp   "
                ds = sorted(m for _, m in res[fw])
                ps = sorted(p for p, _ in res[fw])
                print(
                    f"  {name}: decode med={ds[len(ds)//2]:7.2f} "
                    f"min={ds[0]:7.2f} ms/tok | prefill med={ps[len(ps)//2]:6.2f} "
                    f"ms ({1000/ds[0]:7.1f} tok/s best)"
                )
            torch_decode_med = sorted(m for _, m in res["t"])[len(res["t"]) // 2]
            tp_decode_med = sorted(m for _, m in res["p"])[len(res["p"]) // 2]
            torch_decode_best = min(m for _, m in res["t"])
            tp_decode_best = min(m for _, m in res["p"])
            torch_prefill_med = sorted(p for p, _ in res["t"])[len(res["t"]) // 2]
            tp_prefill_med = sorted(p for p, _ in res["p"])[len(res["p"]) // 2]
            print(
                f"  speedup ref/tp: decode med={torch_decode_med / tp_decode_med:.3f}x "
                f"best={torch_decode_best / tp_decode_best:.3f}x | "
                f"prefill med={torch_prefill_med / tp_prefill_med:.3f}x"
            )


if __name__ == "__main__":
    main()
