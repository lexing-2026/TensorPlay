import math

import pytest

import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE

ops = getattr(tp.ops, NATIVE_NAMESPACE)

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
cuda_only = pytest.mark.skipif(not tp.cuda.is_available(), reason="needs CUDA")


def _close(a, b, tol=1e-5):
    return tp.allclose(a.cpu(), b.cpu(), rtol=tol, atol=tol)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("method", ["fill_", "zero_"])
def test_fill_on_strided_view_writes_only_the_view(device, method):
    base = tp.arange(12.0, device=device).reshape(3, 4)
    expected = base.clone().cpu()
    view = base.select(1, 2)
    if method == "fill_":
        view.fill_(5.0)
        expected[:, 2] = 5.0
    else:
        view.zero_()
        expected[:, 2] = 0.0
    assert tp.equal(base.cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_unsafe_index_put_pairs_values_with_positions(device, accumulate):
    x = tp.zeros(4, 5, device=device)
    rows = tp.tensor([3, 0, 2], device=device)
    cols = tp.tensor([1, 4, 2], device=device)
    values = tp.tensor([10.0, 20.0, 30.0], device=device)
    got = ops._unsafe_index_put.default(x, [rows, cols], values, accumulate)
    expected = ops.index_put.default(x, [rows, cols], values, accumulate)
    assert tp.equal(got.cpu(), expected.cpu())
    assert got[0, 4].item() == 20.0


@pytest.mark.parametrize("device", DEVICES)
def test_unsafe_index_put_accumulates_duplicates(device):
    x = tp.zeros(4, 5, device=device)
    rows = tp.tensor([3, 0, 3], device=device)
    cols = tp.tensor([1, 4, 1], device=device)
    values = tp.tensor([10.0, 20.0, 30.0], device=device)
    got = ops._unsafe_index_put.default(x, [rows, cols], values, True)
    assert got[3, 1].item() == 40.0 and got[0, 4].item() == 20.0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("align_corners", [False, True])
def test_affine_grid_identity_is_the_base_grid(device, align_corners):
    theta = tp.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], device=device)
    grid = ops.affine_grid_generator.default(theta, [1, 1, 2, 3], align_corners)
    bound_w = 1.0 if align_corners else 2.0 / 3.0
    bound_h = 1.0 if align_corners else 0.5
    xs = tp.linspace(-bound_w, bound_w, 3)
    ys = tp.linspace(-bound_h, bound_h, 2)
    assert _close(grid[0, :, :, 0], xs.unsqueeze(0).expand(2, 3))
    assert _close(grid[0, :, :, 1], ys.unsqueeze(1).expand(2, 3))


@pytest.mark.parametrize("device", DEVICES)
def test_upsample_vec_uses_spatial_sizes_and_scales(device):
    x = tp.rand(1, 2, 4, 5, device=device)
    got = ops.upsample_bilinear2d.vec(x, None, False, [1.5, 2.0])
    expected = ops.upsample_bilinear2d.default(x, [6, 10], False, 1.5, 2.0)
    assert tuple(got.shape) == (1, 2, 6, 10)
    assert _close(got, expected)
    with pytest.raises(RuntimeError):
        ops.upsample_bilinear2d.vec(x, [6, 10], False, [1.5, 2.0])


@pytest.mark.parametrize("device", DEVICES)
def test_upsample_backward_takes_the_full_input_shape(device):
    x = tp.rand(2, 3, 3, 4, device=device, requires_grad=True)
    y = ops.upsample_nearest2d.default(x, [7, 5])
    grad = tp.rand(*y.shape, device=device)
    (y * grad).sum().backward()
    direct = ops.upsample_nearest2d_backward.default(grad, [7, 5], [2, 3, 3, 4])
    assert tuple(direct.shape) == (2, 3, 3, 4)
    assert _close(direct, x.grad)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_avg_pool2d_divisor_override(device, ceil_mode):
    x = tp.rand(1, 2, 5, 5, device=device)
    got = ops.avg_pool2d.default(x, [2, 2], [2, 2], [0, 0], ceil_mode, True, 3)
    total = ops.avg_pool2d.default(x, [2, 2], [2, 2], [0, 0], ceil_mode, True, 1)
    assert _close(got * 3, total)
    with pytest.raises(RuntimeError):
        ops.avg_pool2d.default(x, [2, 2], [2, 2], [0, 0], ceil_mode, True, 0)


@cuda_only
def test_avg_pool2d_ceil_mode_divisor_matches_cpu():
    x = tp.rand(1, 1, 5, 5)
    args = ([2, 2], [2, 2], [1, 1], True, True, None)
    cpu = ops.avg_pool2d.default(x, *args)
    gpu = ops.avg_pool2d.default(x.cuda(), *args)
    assert _close(cpu, gpu)


@pytest.mark.parametrize("device", DEVICES)
def test_rms_norm_default_epsilon(device):
    x = tp.rand(3, 4, device=device)
    got = ops.rms_norm.default(x, [4], None, None)
    eps = tp.finfo(tp.float32).eps
    expected = x * tp.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    assert _close(got, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_functional_copy_takes_values_from_src(device):
    self = tp.zeros(2, 3, device=device)
    src = tp.arange(3.0, device=device)
    got = ops.copy.default(self, src, False)
    assert tuple(got.shape) == (2, 3)
    assert tp.equal(got.cpu(), src.cpu().expand(2, 3))
    assert tp.equal(self.cpu(), tp.zeros(2, 3))


@pytest.mark.parametrize("device", DEVICES)
def test_copy_into_a_permuted_destination_places_every_element(device):
    # A square destination laid out column-major over a row-major source has
    # the extents of a relabelled transpose but must still receive src[i, j]
    # at [i, j].
    src = tp.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    base = tp.zeros(2, 2, device=device)
    base.t().copy_(src)
    assert base.cpu().tolist() == [[1.0, 3.0], [2.0, 4.0]]
    for shape, perm in (((64, 64), (1, 0)), ((48, 32), (1, 0)), ((2, 5, 3, 4), (0, 2, 3, 1))):
        source = tp.randn(*shape, device=device)
        dest = tp.empty(*[shape[p] for p in perm], device=device).permute(
            *sorted(range(len(perm)), key=lambda d: perm[d]))
        dest.copy_(source)
        assert tp.equal(dest.cpu(), source.cpu())
        same = tp.empty(*[shape[p] for p in perm], device=device).permute(
            *sorted(range(len(perm)), key=lambda d: perm[d]))
        same.copy_(dest)
        assert tp.equal(same.cpu(), source.cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_copy_broadcasts_its_source_and_refuses_other_shapes(device):
    row = tp.arange(3.0, device=device)
    for source in (row, row.reshape(1, 3), row.cpu()):
        dest = tp.zeros(2, 3, device=device)
        dest.copy_(source)
        assert dest.cpu().tolist() == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
    dest = tp.zeros(2, 3, device=device)
    dest.copy_(tp.tensor([[5.0], [7.0]], device=device))
    assert dest.cpu().tolist() == [[5.0, 5.0, 5.0], [7.0, 7.0, 7.0]]
    # Equal element counts are not enough: the source must broadcast.
    for dshape, sshape in (((2, 3), (3, 2)), ((2, 3), (6,)), ((6,), (2, 3))):
        with pytest.raises(RuntimeError):
            tp.zeros(*dshape, device=device).copy_(tp.zeros(*sshape, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_nll_loss_forward_output_writes_destinations(device):
    scores = tp.log_softmax(tp.rand(4, 5, device=device), 1)
    target = tp.tensor([0, 2, 1, 4], device=device)
    output = tp.empty((), device=device)
    total_weight = tp.empty((), device=device)
    alias = output.view(())
    got = ops.nll_loss_forward.output(scores, target, None, 1, -100,
                                      output=output, total_weight=total_weight)
    expected, expected_weight = ops.nll_loss_forward.default(scores, target, None, 1, -100)
    assert isinstance(got, tuple) and len(got) == 2
    assert _close(alias, expected) and _close(total_weight, expected_weight)


@pytest.mark.parametrize("device", DEVICES)
def test_out_variant_keeps_the_destination_storage(device):
    x = tp.rand(4, 4, device=device)
    out = tp.empty(4, 4, device=device)
    view = out.view(16)
    tp.tril(x, out=out)
    assert _close(view.view(4, 4), tp.tril(x))


@pytest.mark.parametrize("device", DEVICES)
def test_index_add_out_honors_alpha(device):
    x = tp.zeros(3, 2, device=device)
    index = tp.tensor([0, 2], device=device)
    source = tp.ones(2, 2, device=device)
    out = tp.empty(3, 2, device=device)
    ops.index_add.out(x, 0, index, source, alpha=2.5, out=out)
    assert out[0, 0].item() == 2.5 and out[1, 0].item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_hardswish_backward_boundaries(device):
    x = tp.tensor([-3.0, -1.5, 0.0, 3.0, 4.0], device=device)
    grad = ops.hardswish_backward.default(tp.ones(5, device=device), x)
    expected = tp.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    assert _close(grad, expected)


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_eval_backward_applies_slope_at_zero(device):
    x = tp.tensor([-1.0, 0.0, 2.0], device=device)
    grad = ops.rrelu_with_noise_backward.default(
        tp.ones(3, device=device), x, tp.ones(3, device=device), 0.1, 0.3, False, False)
    assert _close(grad, tp.tensor([0.2, 0.2, 1.0]))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("inplace", [False, True])
def test_rrelu_with_noise_draws_and_records_noise(device, inplace):
    x = tp.tensor([-1.0, -2.0, 3.0, -4.0], device=device)
    noise = tp.full((4,), -7.0, device=device)
    if inplace:
        out = x.clone()
        ops.rrelu_with_noise_.default(out, noise, 0.1, 0.3, True)
    else:
        out = ops.rrelu_with_noise.default(x, noise, 0.1, 0.3, True)
    drawn = noise.cpu()
    assert drawn[2].item() == 1.0
    for i in (0, 1, 3):
        assert 0.1 <= drawn[i].item() <= 0.3
    assert _close(out, x.cpu() * drawn)


@pytest.mark.parametrize("device", DEVICES)
def test_rrelu_gradient_is_the_recorded_slope(device):
    tp.manual_seed(0)
    x = tp.tensor([-1.0, -2.0, 3.0, -4.0], device=device, requires_grad=True)
    y = tp.nn.functional.rrelu(x, 0.1, 0.3, training=True)
    y.sum().backward()
    slopes = (y / x).detach()
    assert _close(x.grad, slopes)


@pytest.mark.parametrize("device", DEVICES)
def test_rnn_dropout_between_layers(device):
    tp.manual_seed(0)
    T, N, H = 4, 3, 5
    x = tp.rand(T, N, H, device=device)
    hx = [tp.zeros(2, N, H, device=device)]
    params = []
    for _ in range(2):
        params += [tp.rand(H, H, device=device), tp.rand(H, H, device=device)]
    evaluated = ops.rnn_tanh.default(x, hx, params, False, 2, 0.9, False, False, False)
    plain = ops.rnn_tanh.default(x, hx, params, False, 2, 0.0, True, False, False)
    dropped = ops.rnn_tanh.default(x, hx, params, False, 2, 0.9, True, False, False)
    assert _close(evaluated[0], plain[0])
    assert not _close(dropped[0], plain[0])


def test_every_kernel_matches_the_dispatch_abi():
    assert tp._C._dispatch_abi_mismatches() == []


@pytest.mark.parametrize("device", DEVICES)
def test_linspace_splits_at_half_the_steps(device):
    got = tp.linspace(0.0, 1.0, 5, device=device)
    assert got[2].item() == 0.5
    assert math.isclose(got[-1].item(), 1.0)


# --------------------------------------------------------------------------
# Python indexing goes through index / index_put_
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
def test_integer_beside_advanced_index_selects_its_dimension(device):
    x = tp.arange(120.0, device=device).reshape(2, 3, 4, 5)
    got = x[0, :, [0, 1]]
    assert tuple(got.shape) == (3, 2, 5)
    assert tp.equal(got.cpu(), tp.stack([x[0, :, 0], x[0, :, 1]], dim=1).cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_python_bool_in_tuple_adds_an_indexed_dimension(device):
    x = tp.arange(6.0, device=device).reshape(2, 3)
    assert tuple(x[True].shape) == (1, 2, 3)
    assert tuple(x[False].shape) == (0, 2, 3)
    # The bool's index broadcasts with the list index beside it.
    assert tp.equal(x[True, [0, 1]].cpu(), x[[0, 1]].cpu())
    with pytest.raises(IndexError):
        x[False, [0, 1]]
    assert tuple(x[None, ..., [2]].shape) == (1, 2, 1)


@pytest.mark.parametrize("device", DEVICES)
def test_setitem_with_skipped_dimension(device):
    x = tp.zeros(3, 4, device=device)
    x[:, tp.tensor([3, -4], device=device)] = tp.tensor([1.0, 2.0], device=device)
    expected = tp.zeros(3, 4)
    expected[:, 3] = 1.0
    expected[:, 0] = 2.0
    assert tp.equal(x.cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_setitem_gradient_flows_through_skipped_dimension(device):
    base = tp.randn(3, 4, device=device, requires_grad=True)
    values = tp.randn(3, 2, device=device, requires_grad=True)
    y = base * 1
    y[:, tp.tensor([1, 3], device=device)] = values
    y.sum().backward()
    expected = tp.ones(3, 4)
    expected[:, 1] = 0.0
    expected[:, 3] = 0.0
    assert tp.equal(base.grad.cpu(), expected)
    assert tp.equal(values.grad.cpu(), tp.ones(3, 2))


@pytest.mark.parametrize("device", DEVICES)
def test_mask_assignment_on_strided_view(device):
    base = tp.arange(12.0, device=device).reshape(3, 4)
    view = base.t()
    view[view > 5] = -1.0
    expected = tp.arange(12.0).reshape(3, 4)
    expected[expected > 5] = -1.0
    assert tp.equal(base.cpu(), expected)


@pytest.mark.parametrize("device", DEVICES)
def test_index_put_accumulates_duplicates(device):
    x = tp.zeros(2, 3, device=device)
    rows = tp.tensor([1, 1, 0, 1], device=device)
    values = tp.tensor([[1.0, 2.0, 3.0]] * 4, device=device)
    x.index_put_([rows], values, accumulate=True)
    assert tp.equal(x.cpu(), tp.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]]))


@pytest.mark.parametrize("device", DEVICES)
def test_index_put_non_adjacent_indices(device):
    x = tp.zeros(2, 3, 4, device=device)
    i = tp.tensor([0, 1], device=device)
    k = tp.tensor([3, 2], device=device)
    x.index_put_([i, None, k], tp.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device))
    expected = tp.zeros(2, 3, 4)
    expected[0, :, 3] = tp.tensor([1.0, 2.0, 3.0])
    expected[1, :, 2] = tp.tensor([4.0, 5.0, 6.0])
    assert tp.equal(x.cpu(), expected)


def test_layout_and_memory_format_print_as_module_attributes():
    assert repr(tp.contiguous_format) == "tensorplay.contiguous_format"
    assert repr(tp.channels_last) == "tensorplay.channels_last"
    assert repr(tp.strided) == "tensorplay.strided"
    assert str(tp.per_tensor_affine) == "tensorplay.per_tensor_affine"


@cuda_only
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_nll_loss2d_backward_normalizes_only_mean(reduction):
    x = tp.randn(2, 3, 4, 5).log_softmax(1)
    target = tp.randint(0, 3, (2, 4, 5))
    grads = []
    for device in ("cpu", "cuda"):
        inp = x.to(device).detach().requires_grad_(True)
        tp.nn.functional.nll_loss(inp, target.to(device), reduction=reduction).backward()
        grads.append(inp.grad)
    assert _close(grads[0], grads[1])


@pytest.mark.parametrize("inplace", [False, True])
def test_scatter_add_sums_every_repeat_of_an_index(inplace):
    # Thousands of index rows land on eight destinations per column; with ones
    # as the source each destination must hold exactly its repeat count.
    tp.manual_seed(0)
    idx = tp.randint(0, 8, (4000, 512))
    ones = tp.ones(4000, 512)
    out = tp.zeros(8, 512)
    got = out.scatter_add_(0, idx, ones) if inplace else out.scatter_add(0, idx, ones)
    expected = tp.stack([(idx == k).sum(0).float() for k in range(8)])
    assert tp.equal(got, expected)


def test_scatter_keeps_the_last_write_along_its_axis():
    # Along the scatter axis the later index row wins, call after call.
    idx = tp.zeros(4000, 512, dtype=tp.int64)
    src = tp.arange(4000.0).unsqueeze(1).expand(4000, 512).contiguous()
    got = tp.zeros(2, 512).scatter(0, idx, src)
    assert tp.equal(got[0], tp.full((512,), 3999.0))


@pytest.mark.parametrize("dim,shape,n", [(0, (64, 256), 4000), (1, (999, 77), 77)])
def test_index_add_sums_every_repeat_of_an_index(dim, shape, n):
    tp.manual_seed(0)
    idx = tp.randint(0, shape[dim], (n,))
    src_shape = list(shape)
    src_shape[dim] = n
    got = tp.zeros(*shape).index_add(dim, idx, tp.ones(*src_shape))
    counts = tp.stack([(idx == k).sum().float() for k in range(shape[dim])])
    view = [1, 1]
    view[dim] = shape[dim]
    assert tp.equal(got, counts.reshape(view).expand(*shape))


def test_index_add_with_a_scalar_source_adds_it_everywhere_indexed():
    got = tp.zeros(3, 4).index_add(0, tp.tensor([0, 0, 2]), tp.tensor(1.0))
    assert tp.equal(got, tp.tensor([[2.0] * 4, [0.0] * 4, [1.0] * 4]))


@pytest.mark.parametrize("device", DEVICES)
def test_bitwise_with_a_number_answers_in_the_pair_type(device):
    # An integer number widens a boolean tensor to int64; against an integer
    # tensor it, like a zero-dim tensor, keeps the tensor's own width.
    b = tp.tensor([True, False, True], device=device)
    i8 = tp.tensor([6, 3, 1], dtype=tp.int8, device=device)
    cases = [
        (b & 5, tp.int64, [1, 0, 1]),
        (5 | b, tp.int64, [5, 5, 5]),
        (b ^ 2, tp.int64, [3, 2, 3]),
        (b << 2, tp.int64, [4, 0, 4]),
        (b & True, tp.bool, [True, False, True]),
        (2 << i8, tp.int8, [-128, 16, 4]),
        (i8 & 5, tp.int8, [4, 1, 1]),
        (tp.tensor(5, device=device) & i8, tp.int8, [4, 1, 1]),
    ]
    for got, dtype, values in cases:
        assert got.dtype == dtype
        assert got.cpu().tolist() == values


@cuda_only
def test_cuda_binary_math_broadcasts_like_cpu():
    # atan2, lerp and pow meet their operands in the broadcast shape, and
    # atan2 promotes the pair, as on the CPU.
    tp.manual_seed(0)
    a = tp.randn(3, 5)
    b = tp.randn(5)
    wide = tp.randn(3, 1, dtype=tp.float64)
    cases = [
        lambda a, b, w: tp.atan2(a, b),
        lambda a, b, w: tp.atan2(a, w),
        lambda a, b, w: tp.lerp(a, b, 0.3),
        lambda a, b, w: tp.pow(a.abs(), b),
    ]
    for fn in cases:
        want = fn(a, b, wide)
        got = fn(a.cuda(), b.cuda(), wide.cuda())
        assert tuple(got.shape) == tuple(want.shape) and got.dtype == want.dtype
        assert _close(got, want)


@cuda_only
def test_cuda_depthwise_spelling_takes_an_absent_bias_and_one_group_per_channel():
    # The depthwise spelling answers like the grouped convolution it names --
    # without a bias, and for a weight that widens each input channel into
    # several outputs.
    tp.manual_seed(0)
    x = tp.randn(2, 3, 7, 7, device="cuda", requires_grad=True)
    w = tp.randn(6, 1, 3, 3, device="cuda", requires_grad=True)
    got = ops._conv_depthwise2d.default(x, w, [3, 3], None, [1, 1], [1, 1], [1, 1])
    want = tp.nn.functional.conv2d(x, w, padding=1, groups=3)
    assert _close(got, want)
    gx, gw = tp.autograd.grad(got.square().sum(), [x, w])
    ex, ew = tp.autograd.grad(want.square().sum(), [x, w])
    assert _close(gx, ex, 1e-4) and _close(gw, ew, 1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_dilated_and_transposed_spellings_answer_like_the_convolution(device):
    tp.manual_seed(0)
    x = tp.randn(2, 3, 7, 7, device=device, requires_grad=True)
    w = tp.randn(4, 3, 3, 3, device=device)
    got = ops.slow_conv_dilated2d.default(x, w, [3, 3], None, [1, 1], [2, 2], [2, 2])
    want = tp.nn.functional.conv2d(x, w, padding=2, dilation=2)
    assert _close(got, want, 1e-4)
    (gx,) = tp.autograd.grad(got.sum(), [x])
    (ex,) = tp.autograd.grad(want.sum(), [x])
    assert _close(gx, ex, 1e-4)
    x3 = tp.randn(1, 2, 5, 5, 5, device=device)
    w3 = tp.randn(2, 3, 3, 3, 3, device=device)
    got = ops.slow_conv_dilated3d.default(x3, w3.transpose(0, 1).contiguous(), [3, 3, 3], None, [1, 1, 1], [1, 1, 1], [1, 1, 1])
    assert _close(got, tp.nn.functional.conv3d(x3, w3.transpose(0, 1).contiguous(), padding=1), 1e-4)
    got = ops.slow_conv_transpose3d.default(x3, w3, [3, 3, 3], None, [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1])
    assert _close(got, tp.nn.functional.conv_transpose3d(x3, w3), 1e-4)


@cuda_only
def test_cuda_fused_dropout_keeps_with_the_probability_given():
    tp.manual_seed(0)
    x = tp.ones(20000, device="cuda")
    out, mask = tp._C._fused_dropout(x, 0.9)
    assert mask.dtype == tp.uint8
    kept = mask.float().mean().item()
    assert 0.88 < kept < 0.92
    assert _close(out, mask.float() / 0.9)
