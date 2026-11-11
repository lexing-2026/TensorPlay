#!/usr/bin/env python3
"""End-to-end LLM and CV workloads for the GPU backends.

The same model definitions run against either library, with weights and
inputs drawn from a fixed NumPy generator, so two processes produce
comparable numbers:

    python3 benchmark/model_smoke.py torch      --model cv     --out /tmp/cv_torch.json
    python3 benchmark/model_smoke.py tensorplay --model cv     --out /tmp/cv_tp.json
    python3 benchmark/model_smoke.py torch      --model resnet --out /tmp/rn_torch.json
    python3 benchmark/model_smoke.py tensorplay --model resnet --out /tmp/rn_tp.json
    python3 benchmark/model_smoke.py torch      --model llm --out /tmp/llm_torch.json
    python3 benchmark/model_smoke.py tensorplay --model llm --out /tmp/llm_tp.json

Each run reports the loss, the output checksum and the per-parameter gradient
norms of the first step, plus the loss after one optimizer step.  A model is
"running" when both libraries agree on those numbers and the loss falls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np


def load(name: str):
    if name == "torch":
        import torch

        return torch
    import tensorplay

    return tensorplay


def from_numpy(lib, arr: np.ndarray, device: str):
    """Move a NumPy array onto the device without the library's numpy bridge.

    One of the reference builds on the test box cannot import NumPy through
    its own bridge (the extension was compiled against the 1.x ABI while the
    environment ships 2.x), so the array travels as a byte buffer instead.
    """
    arr = np.ascontiguousarray(arr)
    if lib.__name__ == "torch":
        dtype = {
            np.dtype(np.float32): lib.float32,
            np.dtype(np.float64): lib.float64,
            np.dtype(np.int64): lib.int64,
            np.dtype(np.int32): lib.int32,
        }[arr.dtype]
        t = lib.frombuffer(bytearray(arr.tobytes()), dtype=dtype)
        return t.reshape(*arr.shape).to(device)
    return lib.from_numpy(arr).to(device)


# --------------------------------------------------------------------------
# Model definitions.  Written against the shared surface so both libraries
# execute the same graph; attention is spelled out with matmul + softmax
# instead of the fused module.
# --------------------------------------------------------------------------


def make_cv(lib):
    nn = lib.nn

    class CVNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            f = lib.nn.functional
            x = self.pool(f.relu(self.bn1(self.conv1(x))))
            x = self.pool(f.relu(self.bn2(self.conv2(x))))
            x = f.relu(self.bn3(self.conv3(x)))
            x = f.adaptive_avg_pool2d(x, 1)
            x = x.flatten(1)
            return self.fc(x)

    return CVNet()


def make_resnet(lib, width=64, blocks=(2, 2, 2, 2), classes=10):
    """ResNet-18 shaped network for 32x32 inputs.

    Four stages of two residual blocks each: 3x3 convolutions with a 1x1
    projection on the shortcuts, batch norm after every convolution, and a
    stride-2 first block in stages two to four.
    """
    nn = lib.nn
    f = lib.nn.functional

    class BasicBlock(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(cout)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(cout)
            if stride != 1 or cin != cout:
                self.short_conv = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)
                self.short_bn = nn.BatchNorm2d(cout)
            else:
                self.short_conv = None

        def forward(self, x):
            out = f.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            if self.short_conv is not None:
                x = self.short_bn(self.short_conv(x))
            return f.relu(out + x)

    class ResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, width, 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(width)
            stages = []
            cin = width
            for i, n in enumerate(blocks):
                cout = width * (2 ** i)
                stride = 1 if i == 0 else 2
                stages.append(nn.Sequential(
                    *[BasicBlock(cin if j == 0 else cout, cout,
                                 stride if j == 0 else 1) for j in range(n)]))
                cin = cout
            self.stages = nn.ModuleList(stages)
            self.fc = nn.Linear(cin, classes)

        def forward(self, x):
            x = f.relu(self.stem_bn(self.stem(x)))
            for stage in self.stages:
                x = stage(x)
            x = f.adaptive_avg_pool2d(x, 1).flatten(1)
            return self.fc(x)

    return ResNet()


def make_llm(lib, vocab=1024, dim=256, heads=8, seq=128, batch=4, attn="sdpa",
             n_blocks=2):
    nn = lib.nn
    head_dim = dim // heads
    scale = 1.0 / math.sqrt(head_dim)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
            self.ln2 = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, 4 * dim)
            self.fc2 = nn.Linear(4 * dim, dim)

        def forward(self, x):
            f = lib.nn.functional
            b, t, _ = x.shape
            h = self.ln1(x)
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            q = q.reshape(b, t, heads, head_dim).transpose(1, 2)
            k = k.reshape(b, t, heads, head_dim).transpose(1, 2)
            v = v.reshape(b, t, heads, head_dim).transpose(1, 2)
            if attn == "sdpa":
                o = f.scaled_dot_product_attention(q, k, v)
            else:
                p = f.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
                o = p @ v
            o = o.transpose(1, 2).reshape(b, t, dim)
            x = x + self.proj(o)
            x = x + self.fc2(f.gelu(self.fc1(self.ln2(x))))
            return x

    class LLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
            self.ln_f = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, vocab, bias=False)

        def forward(self, idx):
            x = self.emb(idx)
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.ln_f(x))

    return LLM(), (batch, seq), vocab


# --------------------------------------------------------------------------
# Deterministic weights and inputs
# --------------------------------------------------------------------------


def fill_weights(model, lib, device, seed=0):
    """Fill every parameter from one generator so both runs start equal."""
    rng = np.random.default_rng(seed)
    names = []
    for name, p in sorted(model.named_parameters()):
        shape = tuple(p.shape)
        low = name.lower()
        if p.dim() == 1 and low.endswith(".weight"):
            # Normalization and batch-norm scales start at one.
            arr = np.ones(shape, dtype=np.float32)
        elif p.dim() == 1:
            arr = np.zeros(shape, dtype=np.float32)
        else:
            arr = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        with lib.no_grad():
            p.copy_(from_numpy(lib, arr, device))
        names.append((name, shape))
    return names


def grad_norms(model, lib):
    out = {}
    for name, p in sorted(model.named_parameters()):
        if p.grad is None:
            out[name] = None
            continue
        out[name] = float(p.grad.detach().norm().item())
    return out


def sync(lib) -> None:
    if hasattr(lib, "cuda"):
        lib.cuda.synchronize()
    else:
        lib.accelerator.synchronize()


def run(args) -> dict:
    lib = load(args.library)
    device = args.device
    lib.manual_seed(0)

    if args.model in ("cv", "resnet"):
        model = (make_cv(lib) if args.model == "cv" else make_resnet(lib)).to(device)
        shape = (args.batch, 3, args.size, args.size)
        x = from_numpy(lib, np.random.default_rng(1).standard_normal(shape).astype(np.float32), device)
        target = from_numpy(
            lib, np.random.default_rng(2).integers(0, 10, size=(args.batch,)).astype(np.int64), device
        )
    else:
        model, shape, vocab = make_llm(lib, batch=args.batch, seq=args.seq,
                                       dim=args.dim, heads=args.heads,
                                       n_blocks=args.blocks, attn=args.attn)
        model = model.to(device)
        x = from_numpy(
            lib,
            np.random.default_rng(1).integers(0, vocab, size=shape).astype(np.int64),
            device,
        )
        target = from_numpy(
            lib,
            np.random.default_rng(2).integers(0, vocab, size=shape).astype(np.int64),
            device,
        )

    names = fill_weights(model, lib, device)
    opt = lib.optim.SGD(model.parameters(), lr=args.lr)

    f = lib.nn.functional
    if args.model in ("cv", "resnet"):
        loss_fn = lambda out: f.cross_entropy(out, target)
    else:
        loss_fn = lambda out: f.cross_entropy(out.reshape(-1, out.shape[-1]), target.reshape(-1))

    out = model(x)
    loss = loss_fn(out)
    loss.backward()
    result = {
        "library": args.library,
        "model": args.model,
        "loss": float(loss.item()),
        "out_sum": float(out.detach().sum().item()),
        "out_absmax": float(out.detach().abs().max().item()),
        "params": names,
        "grads": grad_norms(model, lib),
    }

    opt.step()
    with lib.no_grad():
        loss2 = loss_fn(model(x))
    result["loss_after_step"] = float(loss2.item())

    # Warm the step path, then time whole training steps (forward, backward
    # and the update) so the number covers what a training loop pays.
    def one_step():
        opt.zero_grad()
        step_loss = loss_fn(model(x))
        step_loss.backward()
        opt.step()

    for _ in range(args.warmup):
        one_step()
    sync(lib)
    start = lib.cuda.Event(enable_timing=True)
    end = lib.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps - 1):
        one_step()
    end.record()
    sync(lib)
    result["ms_per_step"] = start.elapsed_time(end) / max(args.steps - 1, 1)

    with lib.no_grad():
        result["loss_final"] = float(loss_fn(model(x)).item())
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("library", choices=["torch", "tensorplay"])
    p.add_argument("--model", choices=["cv", "resnet", "llm"], default="cv")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=32, help="input resolution for the cv models")
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--attn", choices=["sdpa", "manual"], default="sdpa",
                   help="attention entry point for the llm model")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out", default="")
    args = p.parse_args()

    result = run(args)
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    print(
        f"{result['library']:<10} {result['model']:<6} "
        f"loss {result['loss']:.6f} -> {result['loss_after_step']:.6f} "
        f"(after {args.steps} steps {result['loss_final']:.6f})  "
        f"out_sum {result['out_sum']:.4f}  absmax {result['out_absmax']:.4f}  "
        f"{result['ms_per_step']:.3f} ms/step"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
