"""Numerical check for the fused attention schedule on CUDA.

Every case runs the fused entry point and a plain float64 softmax attention
computed with numpy, and is accepted only when the two agree to the tolerance
the dtype allows.  The reference deliberately shares no operator with the path
it checks, so a schedule that computes the wrong thing cannot pass.
"""

import math

import tensorplay as tp
from tensorplay._ops import _flash_attention_forward_adapter


def _to_numpy(t):
    """A float64 numpy array of a tensor from this library."""
    import numpy as np

    for attr in ("detach", "cpu", "numpy"):
        if hasattr(t, attr):
            t = getattr(t, attr)()
    return np.asarray(t).astype(np.float64)


def reference(query, key, value, scale, causal, window):
    """Plain softmax attention in float64, on (batch, heads, sequence, dim)."""
    import numpy as np

    q, k, v = (_to_numpy(x) for x in (query, key, value))
    tq, tk = q.shape[2], k.shape[2]
    rows = np.arange(tq).reshape(tq, 1)
    cols = np.arange(tk).reshape(1, tk)
    allowed = np.ones((tq, tk), dtype=bool)
    if causal:
        allowed &= cols <= rows + (tk - tq)
    if window is not None:
        left, right = window
        allowed &= (cols > rows - left - 1) & (cols < rows + right + 1)

    out = np.empty((q.shape[0], q.shape[1], tq, v.shape[3]), dtype=np.float64)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            scores = (q[b, h] @ k[b, h].T) * scale
            scores = np.where(allowed, scores, -np.inf)
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[b, h] = probs @ v[b, h]
    return out


def _run(q, k, v, tq, tk, causal, window=None, **extra):
    if window is None:
        return _flash_attention_forward_adapter(
            q, k, v, None, None, tq, tk, 0.0, causal, **extra
        )
    return _flash_attention_forward_adapter(
        q, k, v, None, None, tq, tk, 0.0, causal,
        window_size_left=window[0], window_size_right=window[1], **extra
    )


def _report(name, err, tol):
    import numpy as np  # noqa: F401

    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name:32s} max_err={err:.3e} tol={tol:.0e}")
    return ok


def check_dense(name, batch, heads, tq, tk, dim, dtype, causal, window, tol):
    import numpy as np

    tp.manual_seed(0)
    # The batched form of this contract is sequence-major, so build it that way.
    q = tp.randn(batch, tq, heads, dim, device="cuda", dtype=dtype)
    k = tp.randn(batch, tk, heads, dim, device="cuda", dtype=dtype)
    v = tp.randn(batch, tk, heads, dim, device="cuda", dtype=dtype)
    got = _run(q, k, v, tq, tk, causal, window)
    if isinstance(got, tuple):
        got = got[0]
    ref = reference(*(x.transpose(1, 2) for x in (q, k, v)),
                    1.0 / math.sqrt(dim), causal, window)
    err = float(np.abs(_to_numpy(got.transpose(1, 2)) - ref).max())
    return _report(name, err, tol)


def check_packed(name, lens, heads, dim, dtype, causal, tol):
    """A packed batch: several sequences in one buffer, marked only by a table.

    Nothing in the tensors marks where one sequence ends and the next begins,
    so this exercises the two parameters that are inert everywhere else:
    ``is_seqlens_k_cumulative`` and ``unpadded_lse``.
    """
    import numpy as np

    tp.manual_seed(0)
    total_q = sum(sq for sq, _ in lens)
    total_k = sum(sk for _, sk in lens)
    q = tp.randn(total_q, heads, dim, device="cuda", dtype=dtype)
    k = tp.randn(total_k, heads, dim, device="cuda", dtype=dtype)
    v = tp.randn(total_k, heads, dim, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(dim)
    max_q = max(sq for sq, _ in lens)
    max_k = max(sk for _, sk in lens)

    cq = np.concatenate([[0], np.cumsum([sq for sq, _ in lens])]).astype("int32")
    ck = np.concatenate([[0], np.cumsum([sk for _, sk in lens])]).astype("int32")
    got = _flash_attention_forward_adapter(
        q, k, v, tp.tensor(cq, device="cuda", dtype=tp.int32),
        tp.tensor(ck, device="cuda", dtype=tp.int32), max_q, max_k, 0.0, causal
    )
    if isinstance(got, tuple):
        got = got[0]

    # The reference walks the table and attends within each sequence alone.
    qn, kn, vn = (_to_numpy(x) for x in (q, k, v))
    out = np.empty((total_q, heads, dim), dtype=np.float64)
    rows = np.arange(max_q).reshape(max_q, 1)
    cols = np.arange(max_k).reshape(1, max_k)
    for i, (sq, sk) in enumerate(lens):
        qs, qe, ks, ke = int(cq[i]), int(cq[i + 1]), int(ck[i]), int(ck[i + 1])
        allow = cols[:sk, :sk] <= rows[:sq, :sk] + (sk - sq) if causal \
            else np.ones((sq, sk), dtype=bool)
        block = np.empty((heads, sq, dim), dtype=np.float64)
        for h in range(heads):
            scores = (qn[qs:qe, h] @ kn[ks:ke, h].T) * scale
            scores = np.where(allow, scores, -np.inf)
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            block[h] = probs @ vn[ks:ke, h]
        out[qs:qe] = block.transpose(1, 0, 2)
    return _report(name, float(np.abs(_to_numpy(got) - out).max()), tol)


def check_gqa(name, batch, heads_q, heads_k, tq, tk, dim, dtype, causal, tol):
    """Grouped heads: more query heads than key heads, sharing each key head."""
    import numpy as np

    tp.manual_seed(0)
    q = tp.randn(batch, tq, heads_q, dim, device="cuda", dtype=dtype)
    k = tp.randn(batch, tk, heads_k, dim, device="cuda", dtype=dtype)
    v = tp.randn(batch, tk, heads_k, dim, device="cuda", dtype=dtype)
    got = _run(q, k, v, tq, tk, causal)
    if isinstance(got, tuple):
        got = got[0]

    qn, kn, vn = (_to_numpy(x) for x in (q, k, v))
    group = heads_q // heads_k
    out = np.empty((batch, tq, heads_q, dim), dtype=np.float64)
    rows = np.arange(tq).reshape(tq, 1)
    cols = np.arange(tk).reshape(1, tk)
    allow = cols <= rows + (tk - tq) if causal else np.ones((tq, tk), dtype=bool)
    for b in range(batch):
        for h in range(heads_q):
            kh = h // group
            scores = (qn[b, :, h] @ kn[b, :, kh].T) * (1.0 / math.sqrt(dim))
            scores = np.where(allow, scores, -np.inf)
            scores = scores - scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[b, :, h] = probs @ vn[b, :, kh]
    return _report(name, float(np.abs(_to_numpy(got) - out).max()), tol)


def check_splits(name, batch, heads, tq, dim, dtype, causal, splits, tol):
    """The key axis split across ``splits`` partial results and combined.

    A different code path from ``num_splits <= 1``: the launch goes to the
    splitting leaf and the partial logsumexps have to be combined correctly.
    """
    import numpy as np

    tp.manual_seed(0)
    q = tp.randn(batch, tq, heads, dim, device="cuda", dtype=dtype)
    k = tp.randn(batch, tq, heads, dim, device="cuda", dtype=dtype)
    v = tp.randn(batch, tq, heads, dim, device="cuda", dtype=dtype)
    got = _run(q, k, v, tq, tq, causal, num_splits=splits)
    if isinstance(got, tuple):
        got = got[0]
    ref = reference(*(x.transpose(1, 2) for x in (q, k, v)),
                    1.0 / math.sqrt(dim), causal, None)
    return _report(name,
                   float(np.abs(_to_numpy(got.transpose(1, 2)) - ref).max()), tol)


def main():
    f16, bf16 = tp.float16, tp.bfloat16
    dense = [
        ("fp16 causal", 2, 4, 128, 128, 64, f16, True, None, 3e-3),
        ("fp16 non-causal", 2, 4, 128, 128, 64, f16, False, None, 3e-3),
        ("bf16 causal", 2, 4, 128, 128, 64, bf16, True, None, 2e-2),
        ("fp16 window 32/0", 2, 4, 128, 128, 64, f16, False, (32, 0), 3e-3),
        ("fp16 causal hdim32", 1, 2, 64, 64, 32, f16, True, None, 3e-3),
        ("fp16 causal hdim96", 1, 2, 64, 64, 96, f16, True, None, 4e-3),
        ("fp16 causal hdim128", 1, 2, 96, 96, 128, f16, True, None, 4e-3),
        ("fp16 causal hdim192", 1, 2, 64, 64, 192, f16, True, None, 5e-3),
        ("fp16 causal hdim256", 1, 2, 64, 64, 256, f16, True, None, 5e-3),
        ("bf16 causal hdim192", 1, 2, 64, 64, 192, bf16, True, None, 3e-2),
        ("bf16 causal hdim256", 1, 2, 64, 64, 256, bf16, True, None, 3e-2),
        ("fp16 non-square causal", 1, 2, 64, 128, 64, f16, True, None, 3e-3),
    ]
    packed = [
        ("fp16 packed causal", [(64, 64), (32, 96), (128, 128)], 4, 64, f16, True, 4e-3),
        ("bf16 packed causal", [(64, 64), (128, 128)], 2, 64, bf16, True, 3e-2),
        ("fp16 packed non-causal", [(48, 48), (80, 80)], 4, 64, f16, False, 4e-3),
    ]
    gqa = [
        ("fp16 gqa 8/2 causal", 2, 8, 2, 128, 128, 64, f16, True, 3e-3),
        ("fp16 gqa 4/1 causal", 1, 4, 1, 96, 96, 64, f16, True, 3e-3),
        ("bf16 gqa 8/4 causal", 1, 8, 4, 128, 128, 64, bf16, True, 2e-2),
    ]
    splits = [
        ("fp16 splitkv 4", 2, 4, 128, 64, f16, True, 4, 4e-3),
        ("fp16 splitkv 8", 1, 4, 256, 64, f16, True, 8, 4e-3),
    ]

    results = [check_dense(*c) for c in dense]
    results += [check_packed(*c) for c in packed]
    results += [check_gqa(*c) for c in gqa]
    results += [check_splits(*c) for c in splits]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
