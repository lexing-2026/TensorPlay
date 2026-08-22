"""AMP alignment tests: tensorplay.amp vs torch.amp semantics.

Runs against the C++ dispatcher implementation (Autocast dispatch keys,
thread-local state in p10, native _amp_* kernels).  Torch-parity cases are
skipped when torch is not importable.
"""

import pickle
import warnings

import pytest

import tensorplay as tp
from tensorplay.amp import GradScaler, autocast


HAS_TORCH = False
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Autocast state API (torch top-level binding surface)
# ---------------------------------------------------------------------------

def test_top_level_state_api():
    for name in (
        "autocast",
        "GradScaler",
        "is_autocast_available",
        "is_autocast_enabled",
        "get_autocast_dtype",
        "set_autocast_enabled",
        "set_autocast_dtype",
        "autocast_increment_nesting",
        "autocast_decrement_nesting",
        "clear_autocast_cache",
        "is_autocast_cache_enabled",
        "set_autocast_cache_enabled",
        "get_autocast_gpu_dtype",
        "get_autocast_cpu_dtype",
    ):
        assert hasattr(tp, name), name


def test_state_defaults_and_availability():
    assert tp.is_autocast_enabled() is False  # device_type defaults to 'cuda'
    assert tp.get_autocast_dtype() == tp.float16
    assert tp.get_autocast_dtype("cpu") == tp.bfloat16
    assert tp.get_autocast_cpu_dtype() == tp.bfloat16
    assert tp.get_autocast_gpu_dtype() == tp.float16
    assert tp.is_autocast_available("cpu")
    assert tp.is_autocast_available("cuda")
    assert not tp.is_autocast_available("xpu")


def test_validation_parity():
    with pytest.raises(ValueError):
        autocast(123)
    with pytest.raises(RuntimeError):
        autocast("xpu")
    with pytest.raises(TypeError):
        autocast("cpu")(123)


# ---------------------------------------------------------------------------
# Context manager / decorator semantics
# ---------------------------------------------------------------------------

def test_enter_exit_restores_state():
    prev_dtype = tp.get_autocast_dtype("cpu")
    prev_cache = tp.is_autocast_cache_enabled()
    assert tp.autocast_increment_nesting() >= 1
    base = tp.autocast_increment_nesting()
    with autocast(device_type="cpu", dtype=tp.float16):
        assert tp.is_autocast_enabled("cpu") is True
        assert tp.get_autocast_dtype("cpu") == tp.float16
        assert tp.is_autocast_cache_enabled() is True
    assert tp.is_autocast_enabled("cpu") is False
    assert tp.get_autocast_dtype("cpu") == prev_dtype
    assert tp.is_autocast_cache_enabled() == prev_cache
    assert tp.autocast_decrement_nesting() == base - 1
    tp.autocast_decrement_nesting()


def test_cache_flag_is_inherited_not_forced():
    # Mirrors torch: cache_enabled=None inherits the ambient flag.
    tp.set_autocast_cache_enabled(False)
    try:
        with autocast(device_type="cpu"):
            assert tp.is_autocast_cache_enabled() is False
        with autocast(device_type="cpu", cache_enabled=True):
            assert tp.is_autocast_cache_enabled() is True
        assert tp.is_autocast_cache_enabled() is False
    finally:
        tp.set_autocast_cache_enabled(True)


def test_nested_disable_region():
    with autocast(device_type="cpu"):
        with autocast(device_type="cpu", enabled=False):
            assert tp.is_autocast_enabled("cpu") is False
        assert tp.is_autocast_enabled("cpu") is True
    assert tp.is_autocast_enabled("cpu") is False


def test_decorator_form():
    @autocast(device_type="cpu")
    def fn(x):
        return tp.matmul(x, x)

    a = tp.randn(4, 4)
    out = fn(a)
    assert out.dtype == tp.bfloat16


def test_unsupported_dtype_warns_and_disables():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with autocast(device_type="cpu", dtype=tp.float32):
            assert tp.is_autocast_enabled("cpu") is False
    assert len(w) == 1
    assert "not supported" in str(w[0].message)


# ---------------------------------------------------------------------------
# Op-level casting (CPU backend)
# ---------------------------------------------------------------------------

def test_lower_precision_ops_cast_to_autocast_dtype():
    a = tp.randn(4, 4)
    b = tp.randn(4, 4)
    with autocast(device_type="cpu"):
        assert tp.matmul(a, b).dtype == tp.bfloat16
        assert tp.mm(a, b).dtype == tp.bfloat16
        import tensorplay.nn.functional as F

        assert F.linear(a, b).dtype == tp.bfloat16
    assert tp.matmul(a, b).dtype == tp.float32


def test_fp32_ops_stay_float32():
    a = tp.randn(4, 4)
    with autocast(device_type="cpu"):
        for op in (tp.exp, tp.log, tp.rsqrt, tp.acos, tp.asin, tp.cosh, tp.sinh, tp.tan):
            out = op(a.abs().clamp_min(0.5))
            assert out.dtype == tp.float32, op
        assert tp.softmax(a, dim=1).dtype == tp.float32
        assert a.softmax(dim=1).dtype == tp.float32
        assert a.sum().dtype == tp.float32
        assert a.sum(dtype=tp.float16).dtype == tp.float16  # explicit dtype respected
        import tensorplay.nn.functional as F

        x = F.layer_norm(a, [4])
        assert x.dtype == tp.float32


def test_promote_ops():
    a = tp.randn(4, 4)
    half_a = a.to(tp.float16)
    with autocast(device_type="cpu"):
        # fp32 present -> promote to fp32
        assert tp.atan2(a, half_a).dtype == tp.float32
        # lower-precision pair -> stays at autocast dtype family
        out = tp.atan2(half_a, half_a)
        assert out.dtype in (tp.float16, tp.bfloat16)


def test_grad_flows_in_original_dtype():
    lin = tp.nn.Linear(4, 2)
    x = tp.randn(8, 4, requires_grad=True)
    with autocast(device_type="cpu"):
        loss = lin(x).sum()
    loss.backward()
    assert lin.weight.grad is not None
    assert lin.weight.grad.dtype == tp.float32
    assert x.grad.dtype == tp.float32


def test_no_grad_autocast_still_casts():
    a = tp.randn(4, 4)
    with tp.no_grad():
        with autocast(device_type="cpu"):
            assert tp.matmul(a, a).dtype == tp.bfloat16


# ---------------------------------------------------------------------------
# custom_fwd / custom_bwd
# ---------------------------------------------------------------------------

def test_custom_fwd_bwd_contract():
    from tensorplay.autograd import Function
    from tensorplay.amp import custom_bwd, custom_fwd

    class MyFn(Function):
        @staticmethod
        @custom_fwd(device_type="cpu")
        def forward(ctx, x):
            ctx._fwd_used_autocast_seen = tp.is_autocast_enabled("cpu")
            return x * 2.0

        @staticmethod
        @custom_bwd(device_type="cpu")
        def backward(ctx, g):
            return g * 2.0

    with autocast(device_type="cpu"):
        x = tp.randn(4, requires_grad=True)
        y = MyFn.apply(x)
        assert y.requires_grad
        y.sum().backward()
    assert x.grad is not None


def test_custom_fwd_cast_inputs():
    from tensorplay.autograd import Function
    from tensorplay.amp import custom_fwd

    seen = {}

    class CastFn(Function):
        @staticmethod
        @custom_fwd(device_type="cpu", cast_inputs=tp.float16)
        def forward(ctx, x):
            seen["dtype"] = x.dtype
            seen["autocast"] = tp.is_autocast_enabled("cpu")
            return x

        @staticmethod
        def backward(ctx, g):
            return g

    with autocast(device_type="cpu"):
        CastFn.apply(tp.randn(3))
    assert seen["dtype"] == tp.float16
    assert seen["autocast"] is False  # autocast disabled inside


# ---------------------------------------------------------------------------
# GradScaler
# ---------------------------------------------------------------------------

def test_scaler_lazy_init_tensor_props():
    scaler = GradScaler("cpu")
    assert scaler.get_scale() == 2.0**16
    out = scaler.scale(tp.tensor([1.0, 2.0]))
    assert isinstance(out, tp.Tensor)
    assert out.shape == (2,) and out[0].item() == 65536.0
    assert scaler._scale.dim() == 0 and scaler._scale.dtype == tp.float32
    assert scaler._growth_tracker.dtype == tp.int32


def test_scaler_growth_and_backoff():
    lin = tp.nn.Linear(4, 2)
    opt = tp.optim.SGD(lin.parameters(), lr=0.1)
    scaler = GradScaler("cpu", growth_interval=3)
    for _ in range(7):
        opt.zero_grad()
        loss = lin(tp.randn(8, 4)).sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    assert scaler.get_scale() == 65536.0 * 4

    scaler2 = GradScaler("cpu")
    before = lin.weight.clone()
    loss = lin(tp.randn(4, 4)).sum()
    scaler2.scale(loss).backward()
    lin.bias.grad.fill_(float("nan"))
    ret = scaler2.step(opt)
    scaler2.update()
    assert ret is None
    assert tp.allclose(lin.weight, before)
    assert scaler2.get_scale() == 65536.0 * 0.5


def test_unscale_math_and_errors():
    lin = tp.nn.Linear(3, 1)
    opt = tp.optim.SGD(lin.parameters(), lr=0.0)
    scaler = GradScaler("cpu")
    loss = lin(tp.randn(5, 3)).sum()
    scaler.scale(loss).backward()
    scaled = lin.weight.grad.clone()
    scaler.unscale_(opt)
    assert tp.allclose(lin.weight.grad, scaled / 65536.0)
    with pytest.raises(RuntimeError):
        scaler.unscale_(opt)

    w16 = tp.tensor([1.0], dtype=tp.float16)
    w16.grad = tp.tensor([1.0], dtype=tp.float16)

    class FakeOpt:
        param_groups = [{"params": [w16]}]

        def step(self):
            pass

    with pytest.raises(ValueError):
        scaler._unscale_grads_(FakeOpt(), tp.full((), 1.0), tp.full((), 0.0), False)

    fresh = GradScaler("cpu")
    with pytest.raises(RuntimeError):
        fresh.step(opt, closure=lambda: None)
    with pytest.raises(AssertionError):
        fresh.update()


def test_disabled_scaler_passthrough():
    scaler = GradScaler("cpu", enabled=False)
    t = tp.tensor([3.0])
    assert scaler.scale(t) is t
    assert scaler.get_scale() == 1.0
    assert scaler.state_dict() == {}
    scaler.load_state_dict({})  # no-op when disabled


def test_state_dict_roundtrip_and_manual_update():
    lin = tp.nn.Linear(4, 2)
    opt = tp.optim.SGD(lin.parameters(), lr=0.1)
    scaler = GradScaler("cpu", growth_interval=2)
    for _ in range(5):
        opt.zero_grad()
        scaler.scale(lin(tp.randn(8, 4)).sum()).backward()
        scaler.step(opt)
        scaler.update()

    sd = scaler.state_dict()
    other = GradScaler("cpu")
    other.scale(tp.tensor(1.0))
    other.load_state_dict(sd)
    assert other.get_scale() == sd["scale"]
    assert other.state_dict()["_growth_tracker"] == sd["_growth_tracker"]

    other.update(123.0)
    assert other.get_scale() == 123.0
    other.update(tp.full((), 77.0))
    assert other.get_scale() == 77.0
    with pytest.raises(AssertionError):
        other.update(tp.full((2,), 1.0))


def test_step_supports_amp_scaling_contract():
    lin = tp.nn.Linear(3, 1)
    calls = {}

    class AmpOpt:
        _step_supports_amp_scaling = True

        def __init__(self, p):
            self.param_groups = [{"params": [p]}]

        def step(self):
            calls["grad_scale"] = getattr(self, "grad_scale", None)
            calls["found_inf"] = getattr(self, "found_inf", None)

    opt = AmpOpt(lin.weight)
    scaler = GradScaler("cpu")
    scaler.scale(lin(tp.randn(3, 3)).sum()).backward()
    scaler.step(opt)
    scaler.update()
    assert not hasattr(opt, "grad_scale") and not hasattr(opt, "found_inf")
    assert calls["grad_scale"].item() == 65536.0
    assert calls["found_inf"].item() == 0.0


def test_pickle_roundtrip():
    scaler = GradScaler("cpu")
    scaler.scale(tp.tensor(1.0))
    restored = pickle.loads(pickle.dumps(scaler))
    assert restored.get_scale() == scaler.get_scale()


# ---------------------------------------------------------------------------
# Torch parity (CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_parity_with_torch_cpu():
    import torch

    def run_tp():
        lin = tp.nn.Linear(6, 4)
        opt = tp.optim.SGD(lin.parameters(), lr=0.1)
        scaler = GradScaler("cpu", growth_interval=3)
        scales = []
        steps_taken = []
        for i in range(9):
            opt.zero_grad()
            x = tp.randn(4, 6)
            loss = lin(x).sum()
            if i == 4:
                # poison one grad to force a skip
                pass
            scaler.scale(loss).backward()
            if i == 4:
                list(lin.parameters())[1].grad.fill_(float("inf"))
            ret = scaler.step(opt)
            scaler.update()
            scales.append(scaler.get_scale())
            steps_taken.append(ret is not None)
        return scales, steps_taken

    def run_torch():
        lin = torch.nn.Linear(6, 4)
        opt = torch.optim.SGD(lin.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", growth_interval=3)
        scales = []
        steps_taken = []
        for i in range(9):
            opt.zero_grad()
            x = torch.randn(4, 6)
            loss = lin(x).sum()
            scaler.scale(loss).backward()
            if i == 4:
                list(lin.parameters())[1].grad.fill_(float("inf"))
            ret = scaler.step(opt)
            scaler.update()
            scales.append(scaler.get_scale())
            steps_taken.append(ret is not None)
        return scales, steps_taken

    tp_scales, tp_steps = run_tp()
    th_scales, th_steps = run_torch()
    assert tp_scales == th_scales, (tp_scales, th_scales)
    assert tp_steps == th_steps


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_parity_autocast_dtypes_cpu():
    import torch

    ta = tp.randn(8, 8)
    tb = tp.randn(8, 8)
    xa = torch.from_numpy(ta.numpy())
    xb = torch.from_numpy(tb.numpy())

    with tp.autocast(device_type="cpu"):
        tp_out = tp.matmul(ta, tb)
        tp_sm = tp.softmax(ta, dim=1)
    with torch.autocast(device_type="cpu"):
        th_out = torch.matmul(xa, xb)
        th_sm = torch.softmax(xa, dim=1)

    assert tp_out.dtype == tp.bfloat16
    assert th_out.dtype == torch.bfloat16
    assert tp_sm.dtype == tp.float32
    assert th_sm.dtype == torch.float32
