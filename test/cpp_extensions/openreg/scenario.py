"""Bring-up scenario for the openreg reference backend.

Run as a child process by test_openreg.py: naming the out-of-tree device is
irreversible for a process, and a registered backend becomes the process's
accelerator, so these checks must not share a process with the rest of the
suite.  Each case runs independently; the result is one JSON object mapping
case name to ``"ok"`` or the failure's traceback, printed on the last line.
"""

import json
import os
import sys
import traceback

import tensorplay as tp
from tensorplay.utils import cpp_extension

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


def assert_close(actual, expected, tol=1e-5):
    actual = actual.cpu() if actual.device.type != "cpu" else actual
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    diff = (actual.to(tp.float64) - expected.to(tp.float64)).abs()
    worst = diff.max().item() if diff.numel() else 0.0
    if worst > tol:
        raise AssertionError(f"max difference {worst} > {tol}")


def expect_raises(exc_type, fn, match=None):
    try:
        fn()
    except exc_type as exc:
        if match is not None and match not in str(exc):
            raise AssertionError(f"expected {match!r} in {str(exc)!r}") from exc
        return
    raise AssertionError(f"{exc_type.__name__} not raised")


@case
def naming_before_load():
    # The slot parses under its default spelling before any backend names it.
    assert tp.device("privateuseone:1").type == "privateuseone"
    assert tp.utils.get_privateuse1_backend_name() == "privateuseone"
    # In-tree device names and non-identifiers are refused.
    expect_raises(ValueError, lambda: tp.utils.rename_privateuse1_backend("cuda"))
    expect_raises(ValueError, lambda: tp.utils.rename_privateuse1_backend("1npu"))
    expect_raises(ValueError, lambda: tp.device("openreg"))


def load_backend():
    cpp_extension.load(
        "openreg",
        [os.path.join(HERE, "openreg.cpp")],
        build_directory=os.environ.get("OPENREG_BUILD_DIR"),
    )
    tp.utils.rename_privateuse1_backend("openreg")


@case
def naming_after_rename():
    assert tp.utils.get_privateuse1_backend_name() == "openreg"
    dev = tp.device("openreg:1")
    assert (dev.type, dev.index, str(dev)) == ("openreg", 1, "openreg:1"), dev
    assert tp.device("openreg") == tp.device("openreg:0")
    assert tp.device("privateuseone:1") == dev
    assert dev.is_privateuse1()
    # Renaming again to the same name is a no-op; to another name an error.
    tp.utils.rename_privateuse1_backend("openreg")
    expect_raises(ValueError, lambda: tp.utils.rename_privateuse1_backend("other"))


@case
def factories_and_copies():
    x = tp.arange(6, dtype=tp.float32).reshape(2, 3)
    y = x.to("openreg")
    assert y.device == tp.device("openreg:0"), y.device
    assert_close(y.cpu(), x)
    # Strided sources, dtype conversion on the device, device-to-device.
    assert_close(x.t().to("openreg").cpu(), x.t())
    assert y.to(tp.float64).dtype == tp.float64
    assert_close(y.to(tp.float64), x)
    z = y.to("openreg:1")
    assert z.device.index == 1
    assert_close(z, x)
    e = tp.empty(4, 5, device="openreg:1")
    assert (e.device.index, tuple(e.shape)) == (1, (4, 5))
    assert_close(tp.ones(3, device="openreg"), tp.ones(3))
    assert_close(tp.zeros(2, 2, device="openreg"), tp.zeros(2, 2))
    assert y.sum().item() == 15.0


@case
def views_need_no_kernels():
    x = tp.arange(12, dtype=tp.float32).reshape(3, 4)
    y = x.to("openreg")
    assert_close(y.t(), x.t())
    assert_close(y[1], x[1])
    assert_close(y.view(4, 3), x.view(4, 3))
    assert_close(y.reshape(2, 6)[:, 1:4], x.reshape(2, 6)[:, 1:4])


@case
def arithmetic_matches_host():
    tp.manual_seed(0)
    a, b = tp.randn(4, 5), tp.randn(5, 3)
    da, db = a.to("openreg"), b.to("openreg")
    assert_close(da @ db, a @ b)
    assert_close(da + 1.5 * da, a + 1.5 * a)
    assert_close((da - da * da) / (da * da + 1), (a - a * a) / (a * a + 1))
    assert_close(tp.relu(da).sum(dim=0), tp.relu(a).sum(dim=0))
    out = tp.zeros(4, 5, device="openreg")
    out.add_(da)
    assert out.device.type == "openreg"
    assert_close(out, a)


@case
def autograd_through_backend_kernels():
    tp.manual_seed(1)
    x = tp.randn(8, 4)
    w1, w2 = tp.randn(4, 6), tp.randn(6, 1)

    def loss_and_grads(device):
        p1 = w1.to(device).detach().requires_grad_()
        p2 = w2.to(device).detach().requires_grad_()
        loss = (tp.relu(x.to(device) @ p1) @ p2).sum()
        loss.backward()
        return loss, p1, p2

    ref_loss, ref1, ref2 = loss_and_grads("cpu")
    loss, g1, g2 = loss_and_grads("openreg")
    assert loss.device.type == "openreg"
    assert g1.grad.device.type == "openreg" and g2.grad.device.type == "openreg"
    assert_close(loss, ref_loss.detach())
    assert_close(g1.grad, ref1.grad)
    assert_close(g2.grad, ref2.grad)


@case
def dispatch_keys_and_abi():
    y = tp.ones(2, device="openreg")
    keys = tp._C._dispatch_keys(y)
    assert "PrivateUse1" in keys and "AutogradPrivateUse1" in keys, keys
    dump = tp._C._dispatch_dump("copy_")
    assert dump.get("PrivateUse1") == "registered", dump
    assert tp._C._dispatch_dump("mm").get("AutogradPrivateUse1") == "registered"
    bad = [row for row in tp._C._dispatch_abi_mismatches() if "PrivateUse1" in row[1]]
    assert not bad, bad


@case
def accelerator_and_device_module():
    acc = tp.accelerator.current_accelerator()
    assert acc is not None and acc.type == "openreg", acc
    assert tp.accelerator.device_count() == 2
    assert tp.accelerator.is_available()
    module = tp.get_device_module("openreg")
    assert module is tp.openreg and module.device_count() == 2
    tp.accelerator.set_device_index(1)
    try:
        assert tp.accelerator.current_device_index() == 1
        index = tp.accelerator.current_device_index()
        assert tp.empty(1, device=f"openreg:{index}").device.index == 1
        with tp.accelerator.device_index(0):
            assert tp.accelerator.current_device_index() == 0
        assert tp.accelerator.current_device_index() == 1
    finally:
        tp.accelerator.set_device_index(0)
    tp.accelerator.synchronize()
    expect_raises(RuntimeError, tp.accelerator.current_stream, match="stream")


@case
def generated_methods():
    tp.utils.generate_methods_for_privateuse1_backend()
    y = tp.ones(3).openreg()
    assert y.is_openreg and not tp.ones(1).is_openreg
    assert tp.ones(1).openreg(1).device.index == 1
    layer = tp.nn.Linear(3, 2).openreg()
    assert layer.weight.device.type == "openreg"
    assert layer(y).device.type == "openreg"
    expect_raises(RuntimeError, tp.utils.generate_methods_for_privateuse1_backend)


@case
def custom_op_kernel_for_the_device():
    @tp.library.custom_op("openreg_test::twice", mutates_args=())
    def twice(x: tp.Tensor) -> tp.Tensor:
        return x * 2

    seen = []

    @twice.register_kernel("PrivateUse1")
    def _(x):
        seen.append(x.device.type)
        return x + x

    y = twice(tp.ones(2, device="openreg"))
    assert seen == ["openreg"], seen
    assert_close(y, tp.full((2,), 2.0))
    assert_close(twice(tp.ones(2)), tp.full((2,), 2.0))


@case
def autocast_on_the_device():
    assert tp.amp.is_autocast_available("openreg")
    a = tp.randn(4, 4, device="openreg")
    with tp.autocast("openreg", dtype=tp.float16):
        assert tp.is_autocast_enabled("openreg")
        out = a @ a
    assert out.dtype == tp.float16, out.dtype
    assert out.device.type == "openreg"
    assert not tp.is_autocast_enabled("openreg")


@case
def vmap_over_device_tensors():
    x = tp.arange(6, dtype=tp.float32).reshape(3, 2)
    out = tp.func.vmap(lambda row: row * 2 + 1)(x.to("openreg"))
    assert out.device.type == "openreg"
    assert_close(out, x * 2 + 1)


@case
def profiler_reports_device_time():
    from tensorplay.profiler import ProfilerActivity, profile, supported_activities

    assert ProfilerActivity.PrivateUse1 in supported_activities()
    a = tp.randn(64, 64).to("openreg")
    for request in ({"activities": [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]},
                    {"use_device": "openreg"}):
        with profile(**request) as prof:
            tp.mm(a, a).relu()
        rows = {row.key: row for row in prof.key_averages()}
        assert rows["mm"].device_time_total > 0, request
        assert rows["relu"].device_time_total > 0, request
        assert "OPENREG" in prof.key_averages().table(sort_by="device_time_total"), request
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        a @ a
    assert all(row.device_time_total == 0 for row in prof.key_averages())


@case
def host_fallbacks_are_counted():
    a = tp.randn(3, 4).to("openreg")
    with tp.utils.count_host_fallbacks() as calls:
        tp.relu(a + a).sum()
    assert calls == {"add.Tensor": 1, "relu": 1, "sum": 1}, calls
    with tp.utils.count_host_fallbacks() as calls:
        a.t()  # a view: no kernel runs
    assert calls == {}, calls
    tp.relu(a)  # outside a counting block
    assert tp._C._host_fallback_counts() == {}, tp._C._host_fallback_counts()


@case
def missing_kernel_is_reported():
    # An operator with neither a host kernel (which the fallback could run)
    # nor a composite one.
    x = tp.ones(1, 1, 2, 2, device="openreg")
    grid = tp.zeros(1, 2, 2, 2, device="openreg")
    expect_raises(NotImplementedError, lambda: tp.cudnn_grid_sampler(x, grid), match="PrivateUse1")


def main():
    results = {}
    loaded = False
    for name, fn in CASES.items():
        if name != "naming_before_load" and not loaded:
            try:
                load_backend()
            except Exception:
                results["load_backend"] = traceback.format_exc()
                break
            loaded = True
        try:
            fn()
            results[name] = "ok"
        except Exception:
            results[name] = traceback.format_exc()
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
