"""The Ascend NPU backend against the host, kernel by kernel and end to end.

Needs an Ascend device and the CANN toolkit; skipped otherwise.  Every check
compares the device result with the same computation on the host.
"""

import os

import pytest

tp = pytest.importorskip("tensorplay")
tensorplay_npu = pytest.importorskip("tensorplay_npu")

if not os.path.isfile(os.path.join(tensorplay_npu.ascend_home(), "include", "acl", "acl.h")):
    pytest.skip("CANN toolkit not installed", allow_module_level=True)

tensorplay_npu.load()
if not tp.npu.is_available():
    pytest.skip("no NPU device", allow_module_level=True)

DTYPES = [tp.float32, tp.float16, tp.bfloat16]
TOL = {tp.float32: 1e-5, tp.float16: 2e-3, tp.bfloat16: 2e-2}


def close(actual, expected, dtype=tp.float32, scale=1.0):
    assert actual.device.type == "npu", actual.device
    got = actual.cpu().to(tp.float64)
    want = expected.to(tp.float64)
    assert tuple(got.shape) == tuple(want.shape), (tuple(got.shape), tuple(want.shape))
    if got.numel() == 0:
        return
    err = (got - want).abs().max().item()
    ref = max(want.abs().max().item(), 1.0) * scale
    assert err <= TOL[dtype] * ref * 10, f"max error {err} (tolerance {TOL[dtype] * ref * 10})"


def rand(*shape, dtype=tp.float32, seed=0):
    tp.manual_seed(seed)
    return tp.randn(*shape).to(dtype)


# -- placement, copies, casts ------------------------------------------------

def test_device_is_named_and_counted():
    assert tp.device("npu:0").type == "npu"
    assert tp.npu.device_count() >= 1
    assert tp.accelerator.current_accelerator().type == "npu"


@pytest.mark.parametrize("dtype", DTYPES + [tp.float64, tp.int64, tp.int32, tp.bool])
def test_round_trip(dtype):
    x = (rand(5, 7) * 10).to(dtype)
    y = x.to("npu")
    assert y.device.type == "npu" and y.dtype == dtype
    assert tp.equal(y.cpu(), x)


def test_strided_copies_both_ways():
    x = rand(6, 8)
    assert tp.equal(x.t().to("npu").cpu(), x.t())
    d = x.to("npu")
    assert tp.equal(d.t().cpu(), x.t())
    assert tp.equal(d[:, 1::3].cpu(), x[:, 1::3])
    out = tp.zeros(8, 6, device="npu")
    out.copy_(d.t())
    assert tp.equal(out.cpu(), x.t())


def test_broadcasting_copy_on_device():
    row = rand(1, 5).to("npu")
    out = tp.empty(4, 5, device="npu")
    out.copy_(row)
    assert tp.equal(out.cpu(), row.cpu().expand(4, 5))


@pytest.mark.parametrize("src,dst", [(tp.float32, tp.float16), (tp.float16, tp.float32),
                                     (tp.float32, tp.bfloat16), (tp.float16, tp.bfloat16),
                                     (tp.float32, tp.int32), (tp.int64, tp.float32),
                                     (tp.float64, tp.float32), (tp.float32, tp.float64)])
def test_casts_on_device(src, dst):
    x = (rand(3, 9) * 4).to(src)
    y = x.to("npu").to(dst)
    assert y.dtype == dst
    assert tp.equal(y.cpu(), x.to(dst))


@pytest.mark.parametrize("dtype", DTYPES + [tp.int32, tp.int64, tp.float64])
def test_fill_and_factories(dtype):
    assert tp.equal(tp.full((3, 4), 2.5 if dtype.is_floating_point else 3,
                            dtype=dtype, device="npu").cpu(),
                    tp.full((3, 4), 2.5 if dtype.is_floating_point else 3, dtype=dtype))
    assert tp.equal(tp.zeros(5, dtype=dtype, device="npu").cpu(), tp.zeros(5, dtype=dtype))
    assert tp.equal(tp.ones(2, 3, dtype=dtype, device="npu").cpu(), tp.ones(2, 3, dtype=dtype))


def test_views_share_device_memory():
    x = rand(4, 6)
    d = x.to("npu")
    v = d.view(6, 4)
    v.mul_(2.0)
    assert tp.equal(d.cpu(), x * 2)
    assert tp.equal(d.permute(1, 0).reshape(-1).cpu(), (x * 2).t().reshape(-1))
    assert tp.equal(d.unsqueeze(0).squeeze(0).cpu(), x * 2)
    parts = d.split(2, dim=1)
    assert len(parts) == 3 and tp.equal(parts[1].cpu(), (x * 2)[:, 2:4])
    assert tp.equal(d.narrow(0, 1, 2).cpu(), (x * 2)[1:3])


def test_resize_keeps_contents():
    d = tp.arange(6, dtype=tp.float32).to("npu")
    d.resize_(12)
    assert tp.equal(d[:6].cpu(), tp.arange(6, dtype=tp.float32))


# -- arithmetic --------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_binary_with_broadcasting(dtype):
    a, b = rand(4, 1, 5, dtype=dtype), rand(3, 1, dtype=dtype, seed=1)
    da, db = a.to("npu"), b.to("npu")
    close(da + db, a.float() + b.float(), dtype)
    close(da - 0.5 * db, a.float() - 0.5 * b.float(), dtype)
    close(tp.sub(da, db, alpha=2), a.float() - 2 * b.float(), dtype)
    close(da * db, a.float() * b.float(), dtype)
    close(da / (db.abs() + 1), a.float() / (b.float().abs() + 1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_scalar_forms_and_host_scalars(dtype):
    a = rand(8, 3, dtype=dtype)
    d = a.to("npu")
    af = a.float()
    close(d + 2, af + 2, dtype)
    close(d * 3, af * 3, dtype)
    close(d / 4, af / 4, dtype)
    close(1 - d, 1 - af, dtype)
    close(d * tp.tensor(2.0), af * 2, dtype)
    close(tp.tensor(2.0) + d, af + 2, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_in_place(dtype):
    a, b = rand(6, 4, dtype=dtype), rand(4, dtype=dtype, seed=2)
    d = a.to("npu")
    d.add_(b.to("npu"), alpha=0.5)
    d.mul_(2)
    d.sub_(b.to("npu"))
    d.div_(3)
    close(d, ((a.float() + 0.5 * b.float()) * 2 - b.float()) / 3, dtype)


def test_strided_operands():
    a = rand(5, 9)
    d = a.to("npu")
    close(d.t() + d.t(), a.t() * 2)
    close(d[:, :4] * d[:, 1::2][:, :4], a[:, :4] * a[:, 1::2][:, :4])
    close(d.t().sum(dim=1), a.t().sum(dim=1), scale=5)
    close(tp.softmax(d.t(), dim=1), tp.softmax(a.t(), dim=1))
    d.t().add_(1.0)
    close(d, a + 1.0)


def test_integer_math_runs_on_host_fallback():
    x = tp.arange(10).to("npu")
    assert tp.equal((x * 3 + 1).cpu(), tp.arange(10) * 3 + 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name", ["exp", "log", "sqrt", "rsqrt", "neg", "abs", "relu",
                                  "sigmoid", "tanh"])
def test_unary(dtype, name):
    a = rand(33, 17, dtype=dtype)
    if name in ("log", "sqrt", "rsqrt"):
        a = a.abs() + 0.1
    got = getattr(tp, name)(a.to("npu"))
    close(got, getattr(tp, name)(a.float()), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_activations(dtype):
    a = rand(16, 32, dtype=dtype)
    d, af = a.to("npu"), a.float()
    F = tp.nn.functional
    close(F.gelu(d), F.gelu(af), dtype)
    close(F.gelu(d, approximate="tanh"), F.gelu(af, approximate="tanh"), dtype)
    close(F.silu(d), F.silu(af), dtype)


# -- products, reductions, softmax -------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ta,tb", [(False, False), (True, False), (False, True), (True, True)])
def test_mm_transposes(dtype, ta, tb):
    a, b = rand(37, 19, dtype=dtype), rand(19, 23, dtype=dtype, seed=3)
    da = a.t().contiguous().to("npu").t() if ta else a.to("npu")
    db = b.t().contiguous().to("npu").t() if tb else b.to("npu")
    close(da @ db, a.float() @ b.float(), dtype, scale=5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_addmm_bmm_matmul(dtype):
    bias, a, b = rand(23, dtype=dtype), rand(37, 19, dtype=dtype, seed=1), rand(19, 23, dtype=dtype, seed=2)
    close(tp.addmm(bias.to("npu"), a.to("npu"), b.to("npu"), beta=0.5, alpha=2.0),
          0.5 * bias.float() + 2.0 * (a.float() @ b.float()), dtype, scale=10)
    x, y = rand(4, 9, 6, dtype=dtype), rand(4, 6, 5, dtype=dtype, seed=4)
    close(tp.bmm(x.to("npu"), y.to("npu")), tp.bmm(x.float(), y.float()), dtype, scale=5)
    w = rand(6, 5, dtype=dtype, seed=5)
    close(x.to("npu") @ w.to("npu"), x.float() @ w.float(), dtype, scale=5)
    v = rand(2, 1, 6, 3, dtype=dtype, seed=6)
    close(x.to("npu") @ v.to("npu"), x.float() @ v.float(), dtype, scale=5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_reductions(dtype):
    a = rand(6, 7, 8, dtype=dtype)
    d, af = a.to("npu"), a.float()
    close(d.sum(), af.sum(), dtype, scale=20)
    close(d.sum(dim=1), af.sum(dim=1), dtype, scale=5)
    close(d.sum(dim=(0, 2), keepdim=True), af.sum(dim=(0, 2), keepdim=True), dtype, scale=20)
    close(d.mean(), af.mean(), dtype)
    close(d.mean(dim=-1), af.mean(dim=-1), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_softmax_and_log_softmax(dtype, dim):
    a = rand(5, 6, 7, dtype=dtype)
    d, af = a.to("npu"), a.float()
    close(tp.softmax(d, dim=dim), tp.softmax(af, dim=dim), dtype)
    close(tp.log_softmax(d, dim=dim), tp.log_softmax(af, dim=dim), dtype)


# -- whole programs ----------------------------------------------------------

def test_operator_without_npu_kernel_runs_on_host():
    a = rand(4, 5)
    close(tp.cumsum(a.to("npu"), dim=1), tp.cumsum(a, dim=1))
    vals, idx = tp.sort(a.to("npu"), dim=1)
    ref_vals, ref_idx = tp.sort(a, dim=1)
    assert vals.device.type == "npu" and tp.equal(idx.cpu(), ref_idx)


def test_training_step_matches_host():
    tp.manual_seed(0)
    model = tp.nn.Sequential(tp.nn.Linear(12, 32), tp.nn.GELU(), tp.nn.Linear(32, 3))
    x, target = tp.randn(16, 12), tp.randint(0, 3, (16,))
    grads = {}
    for device in ("cpu", "npu"):
        m = model.to(device)
        m.zero_grad()
        loss = tp.nn.functional.cross_entropy(m(x.to(device)), target.to(device))
        loss.backward()
        grads[device] = (loss.detach().cpu(), [p.grad.cpu() for p in m.parameters()])
    ref_loss, ref_grads = grads["cpu"]
    loss, got = grads["npu"]
    assert abs(loss.item() - ref_loss.item()) < 1e-4
    for g, r in zip(got, ref_grads):
        assert (g - r).abs().max().item() < 1e-4


def test_memory_statistics():
    tp.npu.empty_cache()
    before = tp.npu.memory_allocated()
    x = tp.empty(1 << 20, device="npu")
    assert tp.npu.memory_allocated() >= before + (4 << 20)
    assert tp.npu.max_memory_allocated() >= tp.npu.memory_allocated()
    del x
    assert tp.npu.memory_allocated() == before
    assert tp.npu.memory_reserved() >= before
    free, total = tp.npu.mem_get_info()
    assert 0 < free <= total
