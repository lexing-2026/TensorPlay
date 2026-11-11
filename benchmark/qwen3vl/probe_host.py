"""Time what each call costs on the host, before anything runs on the device.

A step that is limited by the calls it issues is limited by how long each call
takes to leave the host, not by the work it describes.  Each call is therefore
timed around the point it returns, without waiting for the device, at the
shapes a decode step uses.  The device is drained once before and once after
the batch, so the numbers are host time and not queued work.

    python probe_host.py [--repeat 2000]
"""

from __future__ import annotations

import argparse
import time

import tensorplay as tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=2000)
    args = ap.parse_args()

    import tensorplay_mlu

    tensorplay_mlu.load()

    row = tp.randn(1, 1, 4096, device="mlu", dtype=tp.float16)
    wide = tp.randn(1, 1, 12288, device="mlu", dtype=tp.float16)
    weight = tp.randn(4096, 4096, device="mlu", dtype=tp.float16)
    wide_weight = tp.randn(12288, 4096, device="mlu", dtype=tp.float16)
    heads = tp.randn(1, 32, 1, 128, device="mlu", dtype=tp.float16)
    kv_heads = tp.randn(1, 8, 1, 128, device="mlu", dtype=tp.float16)
    scores = tp.randn(1, 8, 4, 512, device="mlu", dtype=tp.float16)
    half = tp.randn(1, 1, 1, 64, device="mlu", dtype=tp.float16)
    index = tp.tensor([[20]], dtype=tp.int64, device="mlu")
    table = tp.randn(32768, 64, device="mlu", dtype=tp.float32)
    buffer = tp.empty_like(row)
    norm = tp.nn.RMSNorm(4096, eps=1e-6, device="mlu", dtype=tp.float16)
    silu = tp.nn.SiLU()
    cache_out = tp.empty(1, 8, 512, 128, device="mlu", dtype=tp.float16)
    cache_slot = cache_out[:, :, 20:21]

    def host_time(label, fn):
        for _ in range(5):
            fn()
        tp.mlu.synchronize()
        start = time.time()
        for _ in range(args.repeat):
            fn()
        elapsed = time.time() - start
        tp.mlu.synchronize()
        print(f"  {label:46} {elapsed / args.repeat * 1e6:8.2f} us")

    print("smallest possible call")
    host_time("tp.empty(1)", lambda: tp.empty(1, device="mlu"))
    host_time("x.size(0)", lambda: row.size(0))

    print("\nproducts")
    flat = row.view(1, 4096)
    weight_inner = weight.t().contiguous()
    host_time("x @ w.t()", lambda: row @ weight.t())
    host_time("flat.mm(w.t())", lambda: flat.mm(weight.t()))
    host_time("flat @ weight_inner (contracted outermost)",
              lambda: flat @ weight_inner)
    host_time("tp.mm(flat, w.t())", lambda: tp.mm(flat, weight.t()))
    host_time("w.t() alone", lambda: weight.t())
    host_time("x @ wide_weight.t()", lambda: row @ wide_weight.t())
    host_time("kv_heads @ kv_heads.transpose(-1,-2)",
              lambda: kv_heads @ kv_heads.transpose(-1, -2))

    print("\nshape operations")
    host_time("x.view(1, 1, 4096)", lambda: row.view(1, 1, 4096))
    host_time("heads.reshape(1, 32, 128)", lambda: heads.reshape(1, 32, 128))
    host_time("heads.transpose(1, 2)", lambda: heads.transpose(1, 2))
    host_time("heads.contiguous()", lambda: heads.contiguous())
    host_time("heads.transpose(1,2).reshape(1,1,4096)",
              lambda: heads.transpose(1, 2).reshape(1, 1, 4096))
    host_time("x[..., :2048]", lambda: row[..., :2048])
    host_time("x.unsqueeze(2)", lambda: half.unsqueeze(2))

    print("\nthe pieces a written rotation would need")
    quarter = tp.randn(1, 32, 1, 64, device="mlu", dtype=tp.float16)
    wide128 = tp.randn(1, 1, 1, 128, device="mlu", dtype=tp.float16)

    def guard(label, fn):
        try:
            host_time(label, fn)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:46} {type(exc).__name__}: {str(exc)[:44]}")

    guard("q[..., :64]", lambda: heads[..., :64])
    guard("q[..., 64:]", lambda: heads[..., 64:])
    guard("tp.roll(x, -64, -1)", lambda: tp.roll(wide128, -64, -1))
    guard("tp.stack([a, a], dim=-2)", lambda: tp.stack([quarter, quarter], dim=-2))
    guard("tp.cat([a, a], dim=-1)", lambda: tp.cat([quarter, quarter], dim=-1))
    guard("tp.addcmul(a, b, c)",
          lambda: tp.addcmul(wide128, tp.roll(wide128, -64, -1), wide128))
    guard("a * b + a * b", lambda: quarter * quarter + quarter * quarter)

    print("\nthe rotation on the shape a head actually has (1, 1, 32, 128)")
    live = tp.randn(1, 1, 32, 128, device="mlu", dtype=tp.float16)
    live_cos = tp.randn(1, 1, 32, 64, device="mlu", dtype=tp.float16)
    live_sin = tp.randn(1, 1, 32, 64, device="mlu", dtype=tp.float16)
    full_cos = tp.cat([live_cos, live_cos], dim=-1).unsqueeze(2)
    full_sin = tp.cat([-live_sin, live_sin], dim=-1).unsqueeze(2)
    # The pair of every element is the element one half away, so the same read
    # is a permutation of the last axis.
    permutation = tp.tensor(list(range(64, 128)) + list(range(0, 64)),
                            dtype=tp.int64, device="mlu")

    def slices_and_copy():
        half = live.shape[-1] // 2
        first, second = live[..., :half], live[..., half:]
        out = tp.empty_like(live)
        out[..., :half].copy_(first * live_cos - second * live_sin)
        out[..., half:].copy_(second * live_cos + first * live_sin)
        return out

    def in_place():
        # In the model the rotated tensor is the projection's own output, so it
        # can be written over instead of allocating a result.
        x = live * 1.0
        rolled = tp.roll(x, -64, -1)
        rolled.mul_(full_sin)
        x.mul_(full_cos)
        x.add_(rolled)
        return x

    guard("roll + addcmul", lambda: tp.addcmul(
        live * full_cos, tp.roll(live, -64, -1), full_sin))
    guard("roll + mul + add", lambda: (
        live * full_cos + tp.roll(live, -64, -1) * full_sin))
    guard("roll + mul_ + add_ (in place)",
          lambda: in_place())
    guard("slices + sub/copy", slices_and_copy)
    guard("index_select then mul + add", lambda: (
        live * full_cos + live.index_select(3, permutation) * full_sin))
    guard("gather then mul + add", lambda: (
        live * full_cos + live.gather(
            3, permutation.view(1, 1, 1, 128).expand_as(live)) * full_sin))
    guard("the two cat calls the factors need",
          lambda: tp.cat([live_cos, live_cos], dim=-1))

    print("\nbuffers and copies")
    host_time("tp.empty_like(x)", lambda: tp.empty_like(row))
    host_time("buffer.copy_(x)", lambda: buffer.copy_(row))
    host_time("buffer[..., :2048].copy_(x[:2048])",
              lambda: buffer[..., :2048].copy_(row[..., :2048]))
    host_time("cache slot copy_", lambda: cache_slot.copy_(kv_heads))
    host_time("tp.zeros(1, 1, 4096)", lambda: tp.zeros(1, 1, 4096, device="mlu",
                                                      dtype=tp.float16))

    print("\nelementwise and reductions")
    host_time("x * x", lambda: row * row)
    host_time("x + x", lambda: row + row)
    host_time("x - x", lambda: row - row)
    host_time("x * w", lambda: row * weight[0])
    host_time("x.sigmoid()", lambda: row.sigmoid())
    host_time("x.rsqrt()", lambda: row.rsqrt())
    host_time("x.mean(-1, keepdim=True)", lambda: row.mean(-1, keepdim=True))
    host_time("x.to(tp.float32)", lambda: row.to(tp.float32))
    host_time("x.to(tp.float16)", lambda: row.to(tp.float16))
    host_time("wide * wide.sigmoid()", lambda: wide * wide.sigmoid())
    host_time("functional.silu(wide)", lambda: tp.nn.functional.silu(wide))
    host_time("nn.SiLU()(wide)", lambda: silu(wide))

    print("\nnormalization and attention")
    host_time("functional.softmax(scores, -1)",
              lambda: tp.nn.functional.softmax(scores, -1))
    host_time("functional.embedding(index, table)",
              lambda: tp.nn.functional.embedding(index, table))
    host_time("RMSNorm module (kernel)", lambda: norm(row))
    host_time("functional.rms_norm (fused)", lambda: tp.nn.functional.rms_norm(
        row, (4096,), weight[0], 1e-6))
    host_time("RMSNorm module (written, single precision)", lambda: (
        lambda x: (x * ((x.to(tp.float32) * x.to(tp.float32)).mean(-1, keepdim=True)
                         + 1e-6).rsqrt().to(x.dtype)))(row))
    host_time("RMSNorm module (written, same type)", lambda: (
        lambda x: x * ((x * x).mean(-1, keepdim=True) + 1e-6).rsqrt())(row))
    host_time("tp.no_grad() context", lambda: _no_grad(row))

    print("\ndirect binding against the operator form")
    host_time("row @ weight.t()", lambda: row @ weight.t())
    host_time("tp._C.matmul(row, weight.t())",
              lambda: tp._C.matmul(row, weight.t()))
    host_time("row.mul(row)", lambda: row.mul(row))
    host_time("tp._C.mul(row, row)", lambda: tp._C.mul(row, row))

    print("\na fixed batch of ten mixed calls, as issued and with hook discovery off")
    def batch():
        first = row * row
        second = first + row
        third = (second * second).mean(-1, keepdim=True)
        scale = (third + 1e-6).rsqrt()
        scaled = row * scale
        wide = row.to(tp.float32)
        narrowed = wide.to(tp.float16)
        joined = narrowed + scaled
        scores = heads @ kv_heads.transpose(-1, -2)
        return joined @ weight.t()

    host_time("the batch as issued", batch)

    def suppressed():
        from tensorplay.overrides import _disable_tensorplay_function

        with _disable_tensorplay_function():
            return batch()

    host_time("the batch with hook discovery off", suppressed)


def _no_grad(row):
    with tp.no_grad():
        return row * row


if __name__ == "__main__":
    main()
