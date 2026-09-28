"""The attention contracts that have no kernel of their own in this build.

Each one declared in the op schema set is served here by a composite over the
kernels that do exist, so what these check is that the composite reproduces
the contract: the axis order each one takes its inputs in, which corner a
causal mask aligns to, what a sliding window keeps, the shape and meaning of
the logsumexp it hands back, and what a query with no visible key does.

The reference throughout is the same arithmetic in float64, so a case that
disagrees with it disagrees about the contract rather than about rounding.
"""

import math

import pytest

import tensorplay as tp


def _ref_attention(query, key, value, keep=None, scale=None):
    """Attention over head-major inputs, in float64.

    A row with no visible key has no softmax; it is given a zero output here so
    that the comparison is against a defined number rather than a NaN.
    """
    query = query.double()
    key = key.double()
    value = value.double()
    factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else float(scale)
    scores = tp.matmul(query * factor, tp.transpose(key, -2, -1))
    if keep is not None:
        scores = scores + tp.where(
            keep,
            tp.zeros((), dtype=tp.float64, device=query.device),
            tp.full((), float("-inf"), dtype=tp.float64, device=query.device),
        )
    probs = tp.softmax(scores, -1)
    probs = tp.where(tp.isfinite(scores).sum(-1, keepdim=True) > 0, probs, tp.zeros_like(probs))
    return tp.matmul(probs, value)


def _triu_ish(length_q, length_k, bottom_right, device):
    q_idx = tp.arange(length_q, device=device).unsqueeze(-1)
    k_idx = tp.arange(length_k, device=device).unsqueeze(-2)
    if bottom_right:
        return q_idx >= k_idx - (length_k - length_q)
    return q_idx >= k_idx


def _flash(
    query, key, value, *, cum_q=None, cum_k=None, max_q=None, max_k=None, dropout_p=0.0, is_causal=False, **kwargs
):
    return tp.ops.tp._flash_attention_forward(
        query,
        key,
        value,
        cum_q,
        cum_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        False,
        **kwargs,
    )


def _cudnn(
    query, key, value, *, cum_q=None, cum_k=None, max_q=None, max_k=None, dropout_p=0.0, is_causal=False, **kwargs
):
    return tp.ops.tp._cudnn_attention_forward(
        query,
        key,
        value,
        None,
        cum_q,
        cum_k,
        max_q,
        max_k,
        True,
        dropout_p,
        is_causal,
        False,
        **kwargs,
    )


def _efficient(query, key, value, *, custom_mask_type=0, compute_log_sumexp=False, dropout_p=0.0, **kwargs):
    return tp.ops.tp._efficient_attention_forward(
        query,
        key,
        value,
        None,
        None,
        None,
        None,
        None,
        dropout_p,
        custom_mask_type,
        compute_log_sumexp,
        **kwargs,
    )


def _assert_close(name, got, want, tol):
    diff = (got.double() - want.double()).abs().max().item()
    assert diff <= tol, f"{name}: max difference {diff:.3e} exceeds {tol:.3e}"


# The three batched contracts disagree on which of the second and third axes
# is the sequence, which is the easiest thing to get wrong and the least
# visible when it is: a product along the wrong axis still produces a tensor
# of the right shape.  Each case below states the order it hands over.
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_orders_differ_by_axis_only(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dev = tp.device(device)
    batch, heads, length, dim = 2, 4, 16, 32
    generator = tp.Generator()
    generator.manual_seed(0)
    seq_major = [tp.randn([batch, length, heads, dim], generator=generator, device=dev) for _ in range(3)]
    head_major = [t.transpose(1, 2) for t in seq_major]
    keep = _triu_ish(length, length, False, dev)
    want = _ref_attention(*head_major, keep=keep)

    flash = _flash(*seq_major, max_q=length, max_k=length, is_causal=True)
    efficient = _efficient(*seq_major, custom_mask_type=1)
    cudnn = _cudnn(*head_major, max_q=length, max_k=length, is_causal=True)
    _assert_close("flash", flash[0], want.transpose(1, 2), 1e-5)
    _assert_close("efficient", efficient[0], want.transpose(1, 2), 1e-5)
    _assert_close("cudnn", cudnn[0], want, 1e-5)


def test_causal_on_the_flash_contracts_aligns_to_the_lower_right():
    """A causal flag is a window, and the window is measured from the diagonal.

    Both lengths differing is what tells the two corners apart, so this is the
    only way to see which one the contract picks.
    """
    generator = tp.Generator()
    generator.manual_seed(1)
    length_q, length_k, heads, dim = 12, 20, 4, 32
    shape_q = [2, length_q, heads, dim]
    shape_k = [2, length_k, heads, dim]
    q = tp.randn(shape_q, generator=generator)
    k = tp.randn(shape_k, generator=generator)
    v = tp.randn(shape_k, generator=generator)
    got = _flash(q, k, v, max_q=length_q, max_k=length_k, is_causal=True)[0]
    for name, corner in (("lower-right", True), ("upper-left", False)):
        keep = _triu_ish(length_q, length_k, corner, q.device)
        want = _ref_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), keep=keep).transpose(1, 2)
        diff = (got - want).abs().max().item()
        if corner:
            assert diff < 1e-5, f"causal did not align to the {name} corner: {diff:.3e}"
        else:
            assert diff > 1e-3, "the two corners cannot both be what causal means"


@pytest.mark.parametrize("window", [(2, 3), (0, 0), (5, 1), (99, 99), (-1, 4)])
def test_sliding_window_keeps_what_its_bounds_say(window):
    generator = tp.Generator()
    generator.manual_seed(2)
    length_q, length_k, heads, dim = 12, 20, 4, 32
    q = tp.randn([1, length_q, heads, dim], generator=generator)
    k = tp.randn([1, length_k, heads, dim], generator=generator)
    v = tp.randn([1, length_k, heads, dim], generator=generator)
    left, right = window
    got = _flash(q, k, v, max_q=length_q, max_k=length_k, window_size_left=left, window_size_right=right)[0]
    q_idx = tp.arange(length_q).unsqueeze(-1)
    k_idx = tp.arange(length_k).unsqueeze(-2)
    shifted = q_idx + (length_k - length_q)
    keep = None
    if left >= 0:
        keep = k_idx >= shifted - left
    if right >= 0:
        nearer = k_idx < shifted + right + 1
        keep = nearer if keep is None else keep & nearer
    want = _ref_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), keep=keep).transpose(1, 2)
    _assert_close(f"window {window}", got, want, 1e-5)


def test_a_query_with_no_visible_key_keeps_a_zero_and_an_infinite_constant():
    """A query further along than there are keys has an empty score row.

    Dividing by that row's zero total would make every output a NaN.  The
    reported constant is positive infinity rather than negative so that a
    backward pass reusing it reweights the scores to zero, which is the right
    derivative for a row that contributes nothing.
    """
    generator = tp.Generator()
    generator.manual_seed(3)
    heads, length_k, dim = 2, 6, 16
    length_q = 10
    q = tp.randn([1, length_q, heads, dim], generator=generator)
    k = tp.randn([1, length_k, heads, dim], generator=generator)
    v = tp.randn([1, length_k, heads, dim], generator=generator)
    out, lse, *_ = _flash(q, k, v, max_q=length_q, max_k=length_k, is_causal=True)
    assert bool(tp.isfinite(out).all()), "an empty score row leaked a NaN into the output"
    # A causal row sees the keys at or before the shifted diagonal, so with more
    # queries than keys it is the leading queries that run out, not the trailing
    # ones: row r is empty once the diagonal has moved past the first key.
    count = length_q - length_k
    empty_rows = slice(0, count)
    seen_rows = slice(count, length_q)
    _assert_close("empty rows", out[:, empty_rows], tp.zeros_like(out[:, empty_rows]), 0.0)
    # The output came back sequence-major and the constant head-major, so the
    # queries sit on a different axis in each.
    assert bool((lse[..., empty_rows] == float("inf")).all())
    assert bool(tp.isfinite(lse[..., seen_rows]).all())


def test_varlen_visits_each_sequence_on_its_own_terms():
    generator = tp.Generator()
    generator.manual_seed(4)
    heads, dim = 4, 32
    lengths = [5, 9, 1]
    total = sum(lengths)
    tensors = [tp.randn([total, heads, dim], generator=generator) for _ in range(3)]
    q, k, v = tensors
    bounds = tp.tensor([0, 5, 14, 15], dtype=tp.int32)

    out, lse, *_ = _flash(q, k, v, cum_q=bounds, cum_k=bounds, max_q=max(lengths), max_k=max(lengths))
    assert tuple(lse.shape) == (heads, total)
    pieces = []
    start = 0
    for length in lengths:
        stop = start + length
        pieces.append(
            _ref_attention(
                q[start:stop].transpose(0, 1), k[start:stop].transpose(0, 1), v[start:stop].transpose(0, 1)
            ).transpose(0, 1)
        )
        start = stop
    _assert_close("varlen", out, tp.cat(pieces, dim=0), 1e-5)


def test_varlen_causal_uses_each_sequence_length_not_the_batch_maximum():
    """A sequence shorter than the batch maximum still masks against itself.

    Masking against the longest sequence in the batch would let the tail of a
    short sequence see keys past its own end, so the two sequences below have
    to come out different.
    """
    generator = tp.Generator()
    generator.manual_seed(5)
    heads, dim, short, long = 2, 16, 3, 7
    total = short + long
    tensors = [tp.randn([total, heads, dim], generator=generator) for _ in range(3)]
    q, k, v = tensors
    bounds = tp.tensor([0, short, total], dtype=tp.int32)
    out = _flash(q, k, v, cum_q=bounds, cum_k=bounds, max_q=short, max_k=long, is_causal=True)[0]
    pieces = []
    for start, stop in ((0, short), (short, total)):
        pieces.append(
            _ref_attention(
                q[start:stop].transpose(0, 1),
                k[start:stop].transpose(0, 1),
                v[start:stop].transpose(0, 1),
                keep=_triu_ish(stop - start, stop - start, False, q.device),
            ).transpose(0, 1)
        )
    _assert_close("varlen causal", out, tp.cat(pieces, dim=0), 1e-5)


def test_varlen_echoes_the_sequence_table_it_was_given():
    generator = tp.Generator()
    generator.manual_seed(6)
    total, heads, dim = 8, 2, 16
    tensors = [tp.randn([total, heads, dim], generator=generator) for _ in range(3)]
    bounds = tp.tensor([0, 3, 8], dtype=tp.int32)
    result = _cudnn(*tensors, cum_q=bounds, cum_k=bounds, max_q=3, max_k=5)
    assert result[2] is bounds and result[3] is bounds
    assert result[4] == 3 and result[5] == 5


def test_the_inplace_form_fills_the_callers_buffer_and_returns_the_constant():
    generator = tp.Generator()
    generator.manual_seed(7)
    total, heads, dim = 8, 2, 16
    tensors = [tp.randn([total, heads, dim], generator=generator) for _ in range(3)]
    q, k, v = tensors
    bounds = tp.tensor([0, 3, 8], dtype=tp.int32)
    out, lse, *_ = _flash(q, k, v, cum_q=bounds, cum_k=bounds, max_q=3, max_k=5)
    buffer = tp.full((total, heads, dim), float("nan"))
    got_lse = tp.ops.tp._flash_attention_forward_no_dropout_inplace(
        buffer, q, k, v, bounds, bounds, 3, 5, 0.0, False, False
    )
    assert bool(tp.isfinite(buffer).all()), "the caller's buffer still holds its fill value"
    _assert_close("buffer", buffer, out, 0.0)
    _assert_close("constant", got_lse, lse, 0.0)


def test_logsumexp_is_only_computed_when_asked_for():
    generator = tp.Generator()
    generator.manual_seed(8)
    tensors = [tp.randn([2, 8, 4, 16], generator=generator) for _ in range(3)]
    q, k, v = tensors
    asked = _efficient(*tensors, custom_mask_type=1, compute_log_sumexp=True)
    skipped = _efficient(*tensors, custom_mask_type=1)
    assert tuple(asked[1].shape) == (2, 4, 8)
    assert skipped[1].numel() == 0, "an unasked-for constant is not worth computing"
    keep = _triu_ish(8, 8, False, q.device)
    _assert_close(
        "output",
        asked[0],
        _ref_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), keep=keep).transpose(1, 2),
        1e-5,
    )


def test_grouped_query_heads_take_the_run_of_queries_they_feed():
    """Each key head serves a contiguous run of query heads, not a stride.

    The two expansions a head count can be written as are a repeat of each
    head and a repeat of the whole tensor; they differ, and only one of them
    groups heads the way the contract means.
    """
    generator = tp.Generator()
    generator.manual_seed(9)
    batch, length, dim = 1, 8, 16
    q = tp.randn([batch, length, 4, dim], generator=generator)
    k = tp.randn([batch, length, 2, dim], generator=generator)
    v = tp.randn([batch, length, 2, dim], generator=generator)
    grouped = _flash(q, k, v, max_q=length, max_k=length)[0]
    expanded = _flash(
        q, tp.repeat_interleave(k, 2, dim=2), tp.repeat_interleave(v, 2, dim=2), max_q=length, max_k=length
    )[0]
    _assert_close("grouped heads", grouped, expanded, 0.0)
    interleaved = _flash(q, tp.cat([k, k], dim=2), tp.cat([v, v], dim=2), max_q=length, max_k=length)[0]
    assert (grouped - interleaved).abs().max().item() > 1e-3


def test_head_counts_that_do_not_divide_are_refused():
    generator = tp.Generator()
    generator.manual_seed(10)
    q = tp.randn([1, 8, 5, 16], generator=generator)
    k = tp.randn([1, 8, 2, 16], generator=generator)
    with pytest.raises(ValueError, match="multiple of"):
        _flash(q, k, k, max_q=8, max_k=8)


@pytest.mark.parametrize(
    "dtype,tol",
    [
        (tp.float32, 1e-5),
        # the constant is reported in single precision even from wider inputs, so
        # the output is the only thing here that can be held to a float64 tolerance
        (tp.float64, 1e-6),
        (tp.float16, 5e-3),
        (tp.bfloat16, 3e-2),
    ],
)
def test_reduced_precision_accumulates_wider_and_comes_back_narrow(dtype, tol):
    generator = tp.Generator()
    generator.manual_seed(11)
    batch, heads, length, dim = 2, 4, 32, 64
    tensors = [tp.randn([batch, length, heads, dim], generator=generator, dtype=dtype) for _ in range(3)]
    q, k, v = tensors
    out, lse, *_ = _flash(q, k, v, max_q=length, max_k=length, is_causal=True)
    assert out.dtype == dtype
    assert lse.dtype == tp.float32, "the constant is kept wider than the inputs"
    keep = _triu_ish(length, length, False, q.device)
    _assert_close(
        str(dtype),
        out,
        _ref_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), keep=keep).transpose(1, 2),
        tol,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gradients_reach_all_three_inputs(device):
    if device == "cuda" and not tp.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dev = tp.device(device)
    generator = tp.Generator()
    generator.manual_seed(12)
    batch, heads, length, dim = 2, 2, 8, 16
    shape = [batch, length, heads, dim]
    inputs = [tp.randn(shape, generator=generator, device=dev, dtype=tp.float64) for _ in range(3)]
    leaves = [t.requires_grad_(True) for t in inputs]
    replicates = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = _flash(*leaves, max_q=length, max_k=length, is_causal=True)[0]
    seed = tp.randn(list(out.shape), generator=generator, device=dev, dtype=tp.float64)
    out.backward(seed)
    keep = _triu_ish(length, length, False, dev)
    want = _ref_attention(*[t.transpose(1, 2) for t in replicates], keep=keep).transpose(1, 2)
    want.backward(seed)
    for name, got, expect in zip("qkv", leaves, replicates):
        _assert_close(f"grad {name}", got.grad, expect.grad, 1e-9)


@pytest.mark.parametrize(
    "name",
    [
        "_flash_attention_forward",
        "_cudnn_attention_forward",
        "_efficient_attention_forward",
    ],
)
def test_dropout_has_no_composite_here(name):
    """Dropout needs the counter the kernels keep, not just the scaling.

    A composite can rescale by ``1 / (1 - p)`` and get the expectation right
    while getting every individual draw wrong, so the refusal stands.
    """
    generator = tp.Generator()
    generator.manual_seed(13)
    q, k, v = (tp.randn([1, 8, 2, 16], generator=generator) for _ in range(3))
    with pytest.raises(NotImplementedError, match="dropout"):
        if name == "_flash_attention_forward":
            _flash(q, k, v, max_q=8, max_k=8, dropout_p=0.1)
        elif name == "_cudnn_attention_forward":
            _cudnn(q, k, v, max_q=8, max_k=8, dropout_p=0.1)
        else:
            _efficient(q, k, v, dropout_p=0.1)


@pytest.mark.parametrize(
    "name",
    [
        "_flash_attention_forward",
        "_cudnn_attention_forward",
        "_efficient_attention_forward",
    ],
)
def test_cached_and_paged_key_stores_have_no_composite(name):
    """A cached store is an indirection, not an argument the composite can read.

    Trimming a packed tensor to the lengths a cache says are still live is what
    a kernel does inside a decoding step; a composite handed the same tables
    would have to guess whether they name a prefix or a gather.
    """
    generator = tp.Generator()
    generator.manual_seed(14)
    tensors = [tp.randn([1, 8, 2, 16], generator=generator) for _ in range(3)]
    q, k, v = tensors
    lengths = tp.tensor([8], dtype=tp.int32)
    with pytest.raises(NotImplementedError, match="key lengths|seqused_k"):
        if name == "_flash_attention_forward":
            _flash(q, k, v, max_q=8, max_k=8, seqused_k=lengths)
        elif name == "_cudnn_attention_forward":
            _cudnn(q, k, v, max_q=8, max_k=8, seqused_k=lengths)
        else:
            _efficient(q, k, v, seqlen_k=lengths)


def test_low_precision_entry_points_name_the_kernel_they_need():
    with pytest.raises(NotImplementedError, match="low-precision attention kernel"):
        tp.ops.tp._scaled_dot_product_flash_attention.quantized(
            *[tp.zeros(1, 8, 2, 16)] * 3, None, None, None, 0.0, False, False
        )


def test_a_rank_the_contract_does_not_take_is_named_rather_than_computed():
    """A packed call needs three axes and a batched call needs four.

    Which of the second and third axes is the sequence cannot be told from a
    shape, so a swapped order is not caught here; a missing axis is, and it is
    the one mistake that would otherwise reach a product along nothing.
    """
    generator = tp.Generator()
    generator.manual_seed(15)
    batched = [tp.randn([2, 4, 8, 16], generator=generator) for _ in range(3)]
    packed = [tp.randn([8, 4, 16], generator=generator) for _ in range(3)]
    bounds = tp.tensor([0, 4, 8], dtype=tp.int32)
    with pytest.raises(ValueError, match=r"batched attention takes query as"):
        _flash(*packed, max_q=8, max_k=8)
    with pytest.raises(ValueError, match=r"packed attention takes query as"):
        _flash(*batched, cum_q=bounds, cum_k=bounds, max_q=4, max_k=4)
    with pytest.raises(ValueError, match=r"packed attention takes key as"):
        _cudnn(packed[0], batched[1], batched[2], cum_q=bounds, cum_k=bounds)
