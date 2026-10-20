#!/usr/bin/env python3
"""End-to-end toy-LLM training benchmark (the TinyLM from the README).

Trains a small decoder-only language model on the toy corpus from the README
(next-token prediction over a shifting 65-token window) and compares:

* total wall time per optimizer step for this package vs the reference runtime,
* a breakdown of forward / loss / backward / optimizer.step within a step.

Both implementations use the same architecture: embedding, GQA with per-head
query/key RMSNorm and rotary embeddings, SwiGLU feed-forward, final RMSNorm and
un-tied lm_head.  The reference side uses its native scaled-dot-product
attention so both runtimes execute fused attention kernels.

Run::

    python3 benchmark/bench_toy_llm_train.py --device cuda --steps 50
    python3 benchmark/bench_toy_llm_train.py --device cpu  --steps 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import tensorplay as tp
from tensorplay import nn as tp_nn
from tensorplay.nn.cache import DynamicCache
from tensorplay.nn import CrossEntropyLoss as TpCrossEntropyLoss
from tensorplay.optim import AdamW as TpAdamW

VOCAB = 256
HIDDEN = 128
LAYERS = 4
HEADS = 8
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS  # 16
SEQ = 65  # the README toy window is 65 tokens
MLP = 4 * HIDDEN  # SwiGLU up projection width


# ---------------------------------------------------------------------------
# TensorPlay implementation: verbatim TinyLM from the README.
# ---------------------------------------------------------------------------
class TpDecoderLayer(tp_nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, layer_idx):
        super().__init__()
        mixer = tp_nn.GroupedQueryAttention(
            hidden_size, num_heads, num_kv_heads=num_kv_heads,
            qk_norm=True, layer_idx=layer_idx,
        )
        self.block = tp_nn.TransformerBlock(hidden_size, mixer)

    def forward(self, x, position_ids=None, past_key_values=None):
        return self.block(x, position_ids=position_ids,
                          past_key_values=past_key_values)


class TpTinyLM(tp_nn.Module):
    def __init__(self, vocab_size=256, hidden_size=128, num_layers=4,
                 num_heads=8, num_kv_heads=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = tp_nn.Embedding(vocab_size, hidden_size)
        self.layers = tp_nn.ModuleList([
            TpDecoderLayer(hidden_size, num_heads, num_kv_heads, i)
            for i in range(num_layers)
        ])
        self.norm = tp_nn.RMSNorm(hidden_size)
        self.lm_head = tp_nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, past_key_values=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, position_ids=position_ids,
                      past_key_values=past_key_values)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------------------
# Reference implementation: equivalent GQA + RoPE + SwiGLU decoder.
# ---------------------------------------------------------------------------
class RefDecoderLayer(torch.nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.q = torch.nn.Linear(hidden, heads * head_dim, bias=False)
        self.k = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o = torch.nn.Linear(heads * head_dim, hidden, bias=False)
        self.gate = torch.nn.Linear(hidden, MLP, bias=False)
        self.up = torch.nn.Linear(hidden, MLP, bias=False)
        self.down = torch.nn.Linear(MLP, hidden, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        residual = x
        xn = torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=1e-5)

        q = self.q(xn).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(xn).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(xn).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        # Rotary embeddings (half-rotate, the same layout as the tp side):
        # the first half of each head rotates against the second half.
        cos, sin = _rope_cos_sin(T, self.head_dim, x.device, x.dtype)
        cos = torch.cat([cos, cos], dim=-1)[None, None, :, :]
        sin = torch.cat([sin, sin], dim=-1)[None, None, :, :]
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                             is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, self.heads * self.head_dim)
        x = residual + self.o(a)

        residual = x
        xn = torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=1e-5)
        x = residual + self.down(
            torch.nn.functional.silu(self.gate(xn)) * self.up(xn))
        return x


class RefTinyLM(torch.nn.Module):
    def __init__(self, vocab_size=256, hidden_size=128, num_layers=4,
                 num_heads=8, num_kv_heads=2):
        super().__init__()
        self.vocab_size = vocab_size
        head_dim = hidden_size // num_heads
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([
            RefDecoderLayer(hidden_size, num_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.RMSNorm(hidden_size, eps=1e-5)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def _rope_cos_sin(seq, head_dim, device, dtype):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2,
                                               device=device,
                                               dtype=torch.float32)
                                  / head_dim))
    t = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# Training step and measurement helpers.
# ---------------------------------------------------------------------------
def make_tp_model(device="cpu"):
    torch.manual_seed(0)
    model = TpTinyLM(VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS)
    return model.to(device) if device != "cpu" else model


def make_ref_model(device="cpu"):
    torch.manual_seed(0)
    model = RefTinyLM(VOCAB, HIDDEN, LAYERS, HEADS, KV_HEADS)
    return model.to(device) if device != "cpu" else model


def tp_step(model, optimizer, criterion, step, device):
    if device == "cuda":
        tokens = (tp.arange(SEQ, device="cuda").unsqueeze(0)
                  + 3 * step) % VOCAB
    else:
        tokens = (tp.arange(SEQ).unsqueeze(0) + 3 * step) % VOCAB
    logits = model(tokens[:, :-1])
    loss = criterion(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def ref_step(model, optimizer, criterion, step, device):
    tokens = (torch.arange(SEQ, device=device).unsqueeze(0)
              + 3 * step) % VOCAB
    logits = model(tokens[:, :-1])
    loss = criterion(logits.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_step(fn, device, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples)


def breakdown(step_fn, device, iters=20):
    """Return (forward, loss, backward, step) per-call ms for a step builder
    that returns (phase_times_ms, total_ms) each call."""
    best = None
    for _ in range(iters):
        phases, total = step_fn()
        if best is None or total < best[1]:
            best = (phases, total)
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    print(f"TinyLM toy-LLM training  vocab={VOCAB} hidden={HIDDEN} "
          f"layers={LAYERS} heads={HEADS} kv_heads={KV_HEADS} "
          f"seq={SEQ} device={args.device}")

    tp_model = make_tp_model(args.device)
    ref_model = make_ref_model(args.device)

    tp_crit = TpCrossEntropyLoss()
    ref_crit = torch.nn.CrossEntropyLoss()
    tp_opt = TpAdamW(tp_model.parameters(), lr=3e-4)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=3e-4)

    # Warm the GPU kernels / allocators.
    for s in range(args.warmup):
        tp_step(tp_model, tp_opt, tp_crit, s, args.device)
        ref_step(ref_model, ref_opt, ref_crit, s, args.device)
    sync(args.device)

    tp_ms = time_step(
        lambda: tp_step(tp_model, tp_opt, tp_crit, 0, args.device),
        args.device, warmup=0, iters=args.steps)
    ref_ms = time_step(
        lambda: ref_step(ref_model, ref_opt, ref_crit, 0, args.device),
        args.device, warmup=0, iters=args.steps)

    print(f"\nper optimizer step (min of {args.steps}):")
    print(f"  {'tp':>10s} {'ref':>10s} {'speedup':>8s}")
    print(f"  {tp_ms:10.3f} {ref_ms:10.3f} {ref_ms / tp_ms:7.2f}x")

    # Phase breakdown on the reference side.
    def ref_phased():
        t = {}
        t0 = time.perf_counter()
        logits = ref_model((torch.arange(SEQ, device=args.device)
                            .unsqueeze(0))[:, :-1])
        t["forward"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        loss = ref_crit(logits.reshape(-1, VOCAB),
                        torch.arange(SEQ, device=args.device)[1:]
                        .expand(1, -1).reshape(-1))
        t["loss"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        ref_opt.zero_grad()
        loss.backward()
        t["backward"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        ref_opt.step()
        t["step"] = (time.perf_counter() - t0) * 1e3
        sync(args.device)
        return t, sum(t.values())

    def tp_phased():
        t = {}
        tokens = (tp.arange(SEQ, device=args.device).unsqueeze(0))[:, :-1]
        t0 = time.perf_counter()
        logits = tp_model(tokens)
        t["forward"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        loss = tp_crit(logits.reshape(-1, VOCAB),
                       (tp.arange(SEQ, device=args.device)[1:])
                       .expand(1, -1).reshape(-1))
        t["loss"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        tp_opt.zero_grad()
        loss.backward()
        t["backward"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        tp_opt.step()
        t["step"] = (time.perf_counter() - t0) * 1e3
        sync(args.device)
        return t, sum(t.values())

    for name, phased in (("tp", tp_phased), ("ref", ref_phased)):
        phases, total = breakdown(phased, args.device)
        print(f"\n{name} phase breakdown ({total:.3f} ms/step):")
        for k, v in phases.items():
            print(f"  {k:9s} {v:8.3f} ms  {v / total * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())