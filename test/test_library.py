"""Tests for tensorplay.library user-operator integration."""

import math
import unittest

import tensorplay as tp
from tensorplay import library

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_TRITON = False


class CustomOpTest(unittest.TestCase):
    def test_decorated_body_is_default_kernel(self):
        @library.custom_op("testns::mul_add", mutates_args=())
        def mul_add(a, b):
            return tp.add(tp.mul(a, b), b)

        x = tp.tensor([2.0, 3.0])
        y = tp.tensor([4.0, 5.0])
        expected = [2.0 * 4.0 + 4.0, 3.0 * 5.0 + 5.0]
        self.assertEqual(mul_add(x, y).tolist(), expected)
        # The decorated body covers every device until overridden.
        self.assertIs(library.get_op("testns::mul_add"), mul_add)

    def test_device_specific_kernel_overrides(self):
        @library.custom_op("testns::scaled", mutates_args=(), device_types="cpu")
        def scaled(x):
            return tp.mul(x, 2.0)

        @scaled.register_kernel("cpu")
        def cpu_kernel(x):
            return tp.mul(x, 3.0)

        self.assertEqual(scaled(tp.tensor([1.0])).tolist(), [3.0])

    def test_missing_kernel_raises(self):
        @library.custom_op("testns::orphan", mutates_args=())
        def orphan(x):
            return x

        op = library.get_op("testns::orphan")
        # Drop the default kernel to simulate an unregistered device.
        op._kernels.clear()
        with self.assertRaises(NotImplementedError):
            op(tp.tensor([1.0]))

    def test_name_validation(self):
        for bad in ("noscope", "ns::1bad", "", "a::b::c"):
            with self.assertRaises(ValueError):
                library.custom_op(bad)

    def test_duplicate_registration_rejected(self):
        @library.custom_op("testns::dup", mutates_args=())
        def dup(x):
            return x

        with self.assertRaises(RuntimeError):
            @library.custom_op("testns::dup", mutates_args=())
            def _(x):  # noqa: F811 - deliberate duplicate
                return x

    def test_mutates_args_validation(self):
        with self.assertRaises(TypeError):
            library.custom_op("testns::badmut", mutates_args="x")
        with self.assertRaises(TypeError):
            library.custom_op("testns::badmut2", mutates_args=[1])

    def test_register_fake_and_autograd_api_shape(self):
        @library.custom_op("testns::fakeop", mutates_args=())
        def fakeop(x):
            return tp.mul(x, 1.0)

        @fakeop.register_fake
        def _(x):
            return tp.empty_like(x)

        self.assertIsNotNone(fakeop._fake_fn)


class CustomOpAutogradTest(unittest.TestCase):
    def test_backward_via_register_autograd(self):
        @library.custom_op("autons::square", mutates_args=())
        def square(x):
            return tp.mul(x, x)

        def backward(ctx, grad_out):
            (x,) = ctx.saved_tensors
            return (grad_out * x * 2.0,)

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)

        square.register_autograd(backward, setup_context=setup_context)

        x = tp.tensor([1.0, 2.0], requires_grad=True)
        y = square(x)
        y.sum().backward()
        expected = [2.0 * 1.0, 2.0 * 2.0]
        self.assertEqual([float(g) for g in x.grad.tolist()], expected)
        self.assertTrue(y.requires_grad)

    def test_backward_without_setup_context_uses_grad_only(self):
        @library.custom_op("autons::negate", mutates_args=())
        def negate(x):
            return tp.neg(x)

        negate.register_autograd(lambda ctx, g: (tp.neg(g),))

        x = tp.tensor([1.0, -2.0], requires_grad=True)
        negate(x).sum().backward()
        self.assertEqual([float(g) for g in x.grad.tolist()], [-1.0, -1.0])

    def test_no_grad_bypasses_history(self):
        @library.custom_op("autons::idn", mutates_args=())
        def idn(x):
            return tp.mul(x, 1.0)

        idn.register_autograd(lambda ctx, g: (g,))

        x = tp.tensor([1.0], requires_grad=True)
        with tp.no_grad():
            y = idn(x)
        self.assertFalse(y.requires_grad)


class LibraryClassTest(unittest.TestCase):
    def test_define_and_impl(self):
        lib = library.Library("implns")
        lib.define("implns::add_one(Tensor self) -> Tensor")

        lib.impl("add_one", lambda x: tp.add(x, 1.0))

        x = tp.tensor([1.0])
        result = library.get_op("implns::add_one")(x)
        self.assertEqual(result.tolist(), [2.0])

    def test_impl_decorator_form(self):
        lib = library.Library("decns")
        lib.define("decns::twice(Tensor self) -> Tensor")

        @lib.impl("twice")
        def twice(x):
            return tp.mul(x, 2.0)

        self.assertEqual(
            tp.ops.decns.twice(tp.tensor([1.5])).tolist(), [3.0]
        )

    def test_composite_impl_covers_devices(self):
        lib = library.Library("compns")
        lib.define("compns::plus_two(Tensor self) -> Tensor")
        lib.impl(
            "plus_two",
            lambda x: tp.add(x, 2.0),
            device_type="CompositeExplicitAutograd",
        )
        self.assertEqual(
            library.get_op("compns::plus_two")(tp.tensor([0.0])).tolist(),
            [2.0],
        )

    def test_double_def_library_rejected(self):
        library.Library("duplens")
        with self.assertRaises(RuntimeError):
            library.Library("duplens")

    def test_fragment_extends_namespace(self):
        library.Library("fragdef").define("fragdef::a(Tensor) -> Tensor")
        frag = library.Library("fragdef", kind="FRAGMENT")
        frag.define("fragdef::b(Tensor) -> Tensor")
        self.assertTrue(library.has_op("fragdef::b"))


class OpsPacketTest(unittest.TestCase):
    def test_packet_resolves_registered_ops(self):
        @library.custom_op("packetns::hello", mutates_args=())
        def hello(x):
            return tp.add(x, 41.0)

        resolved = tp.ops.packetns.hello
        self.assertIs(resolved, hello)
        self.assertEqual(hello(tp.tensor([1.0])).tolist(), [42.0])

    def test_unknown_packet_attr_raises(self):
        with self.assertRaises(AttributeError):
            tp.ops.no_such_namespace_xyz.op


class NativeBridgeTest(unittest.TestCase):
    """Exercises the C++ Dispatcher bridge when the build carries it."""

    def setUp(self):
        self.bridge = getattr(tp._C, "_call_native_op", None)
        if self.bridge is None:
            self.skipTest("native custom-op bridge unavailable in this build")

    def test_native_roundtrip_matches_python_dispatch(self):
        @library.custom_op("bridgens::weighted", mutates_args=())
        def weighted(x, w):
            return tp.add(tp.mul(x, w), 0.5)

        x = tp.randn(4, 4)
        w = tp.randn(4, 4)
        native = weighted.run_native([x, w])
        eager = weighted(x, w)
        self.assertIsInstance(native, list)
        self.assertTrue(bool(tp.allclose(native[0], eager)))

    def test_has_native_kernel_reflects_registration(self):
        has = getattr(tp._C, "_has_native_kernel", None)
        if has is None:
            self.skipTest("native custom-op bridge unavailable in this build")

        @library.custom_op("bridgens::probe", mutates_args=())
        def probe(x):
            return x

        self.assertTrue(has("bridgens::probe"))
        self.assertFalse(has("bridgens::missing_op"))

    def test_native_mirror_outlives_python_kernels(self):
        # the Python-side kernel table must not corrupt the native mirror;
        # run_native keeps dispatching the registered callable.
        @library.custom_op("bridgens::bare", mutates_args=())
        def bare(x):
            return tp.mul(x, 3.0)

        bare._kernels.clear()
        native = self.bridge("bridgens::bare", [tp.tensor([2.0])])
        self.assertEqual(native[0].tolist(), [6.0])
        with self.assertRaises(Exception):
            self.bridge("bridgens::never_registered", [tp.tensor([1.0])])


class CompilerCaptureTest(unittest.TestCase):
    def test_custom_op_captures_as_opaque_node(self):
        @library.custom_op("capns::square_plus", mutates_args=())
        def square_plus(x):
            return tp.add(tp.mul(x, x), 1.0)

        x = tp.tensor([1.0, 2.0])
        gm = tp.compiler.Tracer().trace(
            lambda a: square_plus(a),
            sample_inputs={"a": x},
        )
        node_targets = [
            getattr(n.target, "__name__", n.target)
            for n in gm.graph.nodes
            if n.op == "call_function"
        ]
        self.assertIn("capns::square_plus", node_targets)
        self.assertEqual(gm(x).tolist(), [2.0, 5.0])

    def test_compiled_matches_eager_across_barrier(self):
        @library.custom_op("capns::shifted", mutates_args=())
        def shifted(x):
            return tp.add(x, 10.0)

        x = tp.tensor([1.0, 2.0])
        compiled = tp.compile(lambda a: tp.mul(shifted(a), 2.0))
        self.assertEqual(compiled(x).tolist(), [22.0, 24.0])

    def test_custom_op_lowers_into_native_graph_not_interpreter(self):
        # dispatcher bridge inside the compiled artifact, never via the
        # Python GraphModule interpreter.
        from tensorplay.backends.stax import stax

        @library.custom_op("capns::native_shift", mutates_args=())
        def native_shift(x):
            return tp.add(x, 10.0)

        x = tp.tensor([1.0, 2.0])
        gm = tp.compiler.Tracer().trace(
            lambda a: tp.mul(native_shift(a), 2.0),
            sample_inputs={"a": x},
        )
        backend_inputs = [x]
        compiled = stax(gm, backend_inputs)
        self.assertIsNotNone(
            getattr(gm, "_stax_native_graph", None),
            "custom-op graph fell back to the interpreter executor",
        )
        self.assertEqual(compiled(x).tolist(), [22.0, 24.0])

    def test_autograd_flows_through_native_graph(self):
        from tensorplay.backends.stax import stax

        @library.custom_op("capns::nat_sq", mutates_args=())
        def nat_sq(x):
            return tp.mul(x, x)

        def backward(ctx, grad_out):
            (saved,) = ctx.saved_tensors
            return (grad_out * saved * 2.0,)

        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)

        nat_sq.register_autograd(backward, setup_context=setup_context)

        x = tp.tensor([1.0, 3.0], requires_grad=True)
        gm = tp.compiler.Tracer().trace(
            lambda a: tp.add(nat_sq(a), 1.0),
            sample_inputs={"a": x},
        )
        compiled = stax(gm, [x])
        out = compiled(x)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertEqual(
            [float(g) for g in x.grad.tolist()], [2.0, 6.0]
        )

    def test_triton_op_is_flagged(self):
        @library.triton_op("tritns::noop", mutates_args=())
        def noop(x):
            return x

        self.assertTrue(noop.is_triton_op)
        self.assertIn("<triton_op tritns::noop>", repr(noop))


class WrapTritonTest(unittest.TestCase):
    class _MockKernel:
        def __init__(self):
            self.launches = []

        def __getitem__(self, grid):
            def launcher(*args, **kwargs):
                self.launches.append((grid, args, kwargs))
                return "launched"

            return launcher

    def test_eager_passthrough(self):
        kernel = WrapTritonTest._MockKernel()
        wrapped = library.wrap_triton(kernel)
        result = wrapped[(32,)](1, 2, BLOCK=64)
        self.assertEqual(result, "launched")
        self.assertEqual(kernel.launches, [((32,), (1, 2), {"BLOCK": 64})])

    def test_wrap_is_idempotent(self):
        kernel = WrapTritonTest._MockKernel()
        once = library.wrap_triton(kernel)
        self.assertIs(library.wrap_triton(once), once)

    def test_non_kernel_rejected(self):
        with self.assertRaises(TypeError):
            library.wrap_triton(object())

    def test_capture_of_raw_launch_raises_graph_error(self):
        kernel = WrapTritonTest._MockKernel()
        wrapped = library.wrap_triton(kernel)

        x = tp.tensor([1.0])

        def broken(a):
            # Simulate a raw launch reaching the tracer: proxies inside.
            return wrapped[(1,)](a)

        try:
            tp.compiler.Tracer().trace(broken, sample_inputs={"x": x})
        except Exception as exc:  # noqa: BLE001
            from tensorplay.compiler.graph import GraphCaptureError

            self.assertIsInstance(exc, GraphCaptureError)
        else:
            self.fail("expected GraphCaptureError for raw triton launch")


class CustomOpSignatureParityTest(unittest.TestCase):

    def test_fn_positional(self):
        def body(x):
            return tp.add(x, 1.0)

        op = library.custom_op("signs::plusk", body, mutates_args=())
        self.assertIs(library.get_op("signs::plusk"), op)
        self.assertEqual(op(tp.tensor([1.0])).tolist(), [2.0])

    def test_schema_attachment(self):
        @library.custom_op(
            "signs::docd",
            mutates_args=(),
            schema="signs::docd(Tensor self) -> Tensor",
        )
        def docd(x):
            return x

        self.assertEqual(docd.schema, "signs::docd(Tensor self) -> Tensor")

    def test_triton_and_tilelang_accept_fn_positional(self):
        op = library.triton_op("signs::trop", lambda x: x, mutates_args=())
        self.assertTrue(op.is_triton_op)
        tl = library.tile_lang_op("signs::tlop", lambda x: x, mutates_args=())
        self.assertTrue(tl.is_tile_lang_op)
        self.assertIn("<tile_lang_op signs::tlop>", repr(tl))
        self.assertFalse(tl.is_triton_op)


class KwargsPassThroughTest(unittest.TestCase):
    """Regression: eager dispatch must forward keyword arguments."""

    def test_kwargs_reach_kernel(self):
        @library.custom_op("kwns::scale_by", mutates_args=())
        def scale_by(x, alpha=1.0):
            return tp.mul(x, alpha)

        x = tp.tensor([2.0])
        self.assertEqual(scale_by(x, alpha=3.0).tolist(), [6.0])
        self.assertEqual(scale_by(x).tolist(), [2.0])

    def test_kwargs_reach_autograd_path(self):
        @library.custom_op("kwns::powk", mutates_args=())
        def powk(x, p=2.0):
            return tp.pow(x, p)

        powk.register_autograd(lambda ctx, g: (g,))
        x = tp.tensor([3.0], requires_grad=True)
        y = powk(x, p=2.0)
        self.assertEqual(y.tolist(), [9.0])
        y.sum().backward()
        self.assertIsNotNone(x.grad)


class SetKernelEnabledTest(unittest.TestCase):
    def test_disable_falls_back_to_composite(self):
        @library.custom_op("enabens::dual", mutates_args=())
        def dual(x):
            return tp.mul(x, 1.0)  # composite slot

        @dual.register_kernel("cpu")
        def _(x):
            return tp.mul(x, 10.0)  # concrete slot

        x = tp.tensor([1.0])
        self.assertEqual(dual(x).tolist(), [10.0])
        with dual.set_kernel_enabled("cpu", enabled=False):
            self.assertEqual(dual(x).tolist(), [1.0])
        self.assertEqual(dual(x).tolist(), [10.0])

    def test_double_toggle_warns(self):
        @library.custom_op("enabens::one", mutates_args=())
        def one(x):
            return x

        @one.register_kernel("cpu")
        def _(x):
            return x

        # Enter the first disable outside the recorder: the *second* disable
        # must warn "already disabled" (nested assertWarns would swallow it,
        # since only the innermost catch_warnings records).
        disable_ctx = one.set_kernel_enabled("cpu", enabled=False)
        with disable_ctx, self.assertWarns(UserWarning):
            with one.set_kernel_enabled("cpu", enabled=False):
                pass
        self.assertEqual(one._disabled_kernels, set())


class GetKernelTest(unittest.TestCase):
    def test_get_kernel_resolves_slots(self):
        @library.custom_op("gkerns::probe", mutates_args=())
        def probe(x):
            return x  # composite/default slot

        @probe.register_kernel("cpu")
        def cpu_fn(x):
            return x

        self.assertIs(library.get_kernel(probe, "cpu"), cpu_fn)
        self.assertIs(library.get_kernel(probe, "default"), probe._kernels[None])
        self.assertIs(library.get_kernel(probe, "CompositeExplicitAutograd"), probe._kernels[None])

    def test_missing_kernel_raises_lookup_error(self):
        @library.custom_op("gkerns::lonely", mutates_args=())
        def lonely(x):
            return x

        with self.assertRaises(LookupError):
            library.get_kernel(lonely, "cuda")


class RegisterVmapTest(unittest.TestCase):
    def test_method_and_toplevel_store_formula(self):
        @library.custom_op("vmapns::f", mutates_args=())
        def f(x):
            return x

        formula = lambda info, in_dims, *args: (args[0], in_dims[0])  # noqa: E731
        ret = f.register_vmap(formula)
        self.assertIs(ret, formula)
        self.assertIs(f._vmap_fn, formula)
        library.register_vmap(f, formula)
        # Autograd class exposes the hook once a backward exists too.
        f.register_autograd(lambda ctx, g: (g,))
        self.assertTrue(hasattr(f._autograd_cls, "vmap"))

    def test_non_callable_rejected(self):
        @library.custom_op("vmapns::g", mutates_args=())
        def g(x):
            return x

        with self.assertRaises(TypeError):
            g.register_vmap(42)


class RegisterAutocastTest(unittest.TestCase):
    def test_autocast_casts_floating_inputs(self):
        seen = []

        @library.custom_op("acns::dtype_probe", mutates_args=())
        def probe(x):
            seen.append(str(x.dtype))
            return x

        probe.register_autocast("cpu", tp.float16)

        x = tp.tensor([1.0])
        probe(x)  # autocast disabled: fp32 flows through untouched
        self.assertEqual(seen[-1], str(tp.float32))
        with tp.autocast("cpu", dtype=tp.float16):
            self.assertTrue(tp.is_autocast_enabled("cpu"))
            probe(x)
        self.assertEqual(seen[-1], str(tp.float16))
        probe(x)  # exited: back to fp32
        self.assertEqual(seen[-1], str(tp.float32))

    def test_toplevel_registration(self):
        @library.custom_op("acns2::probe", mutates_args=())
        def probe(x):
            return x

        library.register_autocast(probe, "cpu", tp.bfloat16)
        self.assertIn("cpu", probe._autocast_rules)

    def test_invalid_rule_rejected(self):
        @library.custom_op("acns3::probe", mutates_args=())
        def probe(x):
            return x

        with self.assertRaises(TypeError):
            probe.register_autocast("cpu", tp.int32)


class InferSchemaTest(unittest.TestCase):
    def test_typed_prototype(self):
        from tensorplay import Tensor

        def weighted(x: Tensor, n: int, out: Tensor) -> Tensor:
            raise NotImplementedError

        schema = library.infer_schema(
            weighted, mutates_args=("out",), op_name="mylib::weighted"
        )
        self.assertEqual(schema, "mylib::weighted(Tensor, SymInt, Tensor(a!)) -> Tensor")

    def test_unannotated_defaults_to_tensor(self):
        def bare(x, y):
            raise NotImplementedError

        schema = library.infer_schema(bare, mutates_args=())
        self.assertEqual(schema, "bare(Tensor, Tensor) -> Tensor")


class OpcheckTest(unittest.TestCase):
    def _make_good_op(self, name):
        @library.custom_op(name, mutates_args=())
        def square(x):
            return tp.mul(x, x)

        @square.register_fake
        def _(x):
            return tp.empty_like(x)

        square.register_autograd(lambda ctx, g: (g * 2.0,))
        return square

    def test_well_registered_op_passes_all(self):
        op = self._make_good_op("opchkns::square_ok")
        failures = library.opcheck(
            op, (tp.tensor([1.0, 2.0]),), raise_exception=False
        )
        self.assertEqual(failures, {})

    def test_undeclared_mutation_detected(self):
        @library.custom_op("opchkns::mutator", mutates_args=())
        def mutator(x):
            y = tp.mul(x, 2.0)  # writes through x's storage? no: fresh buf;
            x.copy_(y)  # actual in-place write
            return y

        failures = library.opcheck(
            mutator, (tp.tensor([1.0]),), raise_exception=False
        )
        self.assertIn("test_schema", failures)

    def test_input_aliasing_output_detected(self):
        @library.custom_op("opchkns::aliaser", mutates_args=())
        def aliaser(x):
            return x  # returns the input itself

        failures = library.opcheck(
            aliaser, (tp.tensor([1.0]),), raise_exception=False
        )
        self.assertIn("test_schema", failures)

    def test_missing_fake_reported(self):
        @library.custom_op("opchkns::nofake", mutates_args=())
        def nofake(x):
            return tp.add(x, 1.0)

        failures = library.opcheck(nofake, (tp.tensor([1.0]),), raise_exception=False)
        self.assertIn("test_faketensor", failures)

    def test_fake_shape_mismatch_detected(self):
        @library.custom_op("opchkns::badfake", mutates_args=())
        def badfake(x):
            return tp.add(x, 1.0)

        @badfake.register_fake
        def _(x):
            return tp.empty([x.shape[0] + 1])

        failures = library.opcheck(
            badfake, (tp.tensor([1.0]),), raise_exception=False
        )
        self.assertIn("test_faketensor", failures)

    def test_autograd_gap_detected(self):
        # TensorPlay kernels differentiate implicitly through composed ops,
        # so a missing register_autograd is legal; what opcheck flags is a
        # kernel that BREAKS the graph or drops the gradient.
        @library.custom_op("opchkns::gradbreaker", mutates_args=())
        def gradbreaker(x):
            with tp.no_grad():
                return tp.mul(x, 2.0)

        @gradbreaker.register_fake
        def _(x):
            return tp.empty_like(x)

        failures = library.opcheck(
            gradbreaker,
            (tp.tensor([1.0], requires_grad=True),),
            raise_exception=False,
        )
        self.assertIn("test_autograd_registration", failures)

    def test_implicit_differentiation_passes(self):
        @library.custom_op("opchkns::implicit", mutates_args=())
        def implicit(x):
            return tp.mul(x, 2.0)  # composite-implicit: no formula needed

        failures = library.opcheck(
            implicit,
            (tp.tensor([1.0], requires_grad=True),),
            raise_exception=False,
            test_utils=("test_schema", "test_autograd_registration"),
        )
        self.assertEqual(failures, {})

    def test_raise_exception_mode(self):
        op = self._make_good_op("opchkns::raiseok")
        self.assertEqual(library.opcheck(op, (tp.tensor([2.0]),)), {})

    def test_unknown_test_utils_rejected(self):
        op = self._make_good_op("opchkns::unknownutils")
        with self.assertRaises(ValueError):
            library.opcheck(op, (tp.tensor([1.0]),), test_utils="test_nope")


class TopLevelDefineImplTest(unittest.TestCase):
    def test_define_then_impl(self):
        library.define("topdefs::bump(Tensor self) -> Tensor")
        library.impl("topdefs::bump", "cpu", lambda x: tp.add(x, 1.0))
        self.assertEqual(
            library.get_op("topdefs::bump")(tp.tensor([1.0])).tolist(), [2.0]
        )

    def test_impl_decorator_form(self):
        library.define("topdefs::twice2(Tensor) -> Tensor")

        @library.impl("topdefs::twice2", "cpu")
        def twice2(x):
            return tp.mul(x, 2.0)

        self.assertEqual(twice2(tp.tensor([1.0])).tolist(), [2.0])

    def test_impl_abstract_alias_registers_fake(self):
        library.define("topdefs::meta(Tensor) -> Tensor")
        library.impl_abstract("topdefs::meta", lambda x: tp.empty_like(x))
        self.assertIsNotNone(library.get_op("topdefs::meta")._fake_fn)


class LibraryImplDispatchKeyAliasTest(unittest.TestCase):
    def test_dispatch_key_kwarg(self):
        lib = library.Library("dkalias")
        lib.define("dkalias::go(Tensor) -> Tensor")
        lib.impl("go", lambda x: tp.add(x, 5.0), dispatch_key="CPU")
        self.assertEqual(
            library.get_op("dkalias::go")(tp.tensor([0.0])).tolist(), [5.0]
        )


class _FakeTileLangKernel:
    """Duck-typed stand-in for tilelang.jit.kernel.JITKernel."""

    _tilelang_kernel = True

    def __init__(self):
        self.launches = []

    def __call__(self, *args, **kwargs):
        self.launches.append((args, kwargs))
        return "tl-launched"


class _FakeTileLangFactory:
    """Duck-typed stand-in for lazy-mode tilelang.jit.JITImpl."""

    def get_tir(self, *args, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        return _FakeTileLangKernel()


class WrapTileLangTest(unittest.TestCase):
    def test_eager_passthrough(self):
        kernel = _FakeTileLangKernel()
        wrapped = library.wrap_tilelang(kernel)
        self.assertEqual(wrapped(1, 2, BLOCK=64), "tl-launched")
        self.assertEqual(kernel.launches, [((1, 2), {"BLOCK": 64})])

    def test_lazy_factory_accepted(self):
        wrapped = library.wrap_tilelang(_FakeTileLangFactory())
        bound = wrapped.compile(1024, 1024, 1024)
        self.assertIsInstance(bound, library.TileLangKernelWrapper)
        self.assertEqual(bound(), "tl-launched")

    def test_wrap_is_idempotent(self):
        kernel = _FakeTileLangKernel()
        once = library.wrap_tilelang(kernel)
        self.assertIs(library.wrap_tilelang(once), once)

    def test_plain_callable_rejected(self):
        with self.assertRaises(TypeError):
            library.wrap_tilelang(lambda: None)
        with self.assertRaises(TypeError):
            library.wrap_tilelang(object())

    def test_capture_of_raw_launch_raises_graph_error(self):
        wrapped = library.wrap_tilelang(_FakeTileLangKernel())

        def broken(a):
            return wrapped(a)

        x = tp.tensor([1.0])
        with self.assertRaises(Exception) as ctx:
            tp.compiler.Tracer().trace(broken, sample_inputs={"a": x})
        from tensorplay.compiler.graph import GraphCaptureError

        self.assertIsInstance(ctx.exception, GraphCaptureError)


class TileLangOpCaptureTest(unittest.TestCase):
    def test_tile_lang_op_captures_single_opaque_node(self):
        launched = []
        wrapped = library.wrap_tilelang(_FakeTileLangKernel())

        @library.tile_lang_op("tlcapns::shift", mutates_args=())
        def shift(x):
            # Eager path would launch here; tracing never runs the body.
            launched.append(True)
            del wrapped
            return tp.add(x, 1.0)

        x = tp.tensor([1.0, 2.0])
        gm = tp.compiler.Tracer().trace(lambda a: shift(a), sample_inputs={"a": x})

        self.assertEqual(launched, [])  # body skipped under trace
        op_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, library.CustomOpDef)
        ]
        self.assertEqual(len(op_nodes), 1)
        self.assertTrue(op_nodes[0].target.is_tile_lang_op)
        compute_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(len(compute_nodes), 1)  # fusion barrier held

    def test_compiled_matches_eager(self):
        @library.tile_lang_op("tlcapns::plus_ten", mutates_args=())
        def plus_ten(x):
            return tp.add(x, 10.0)

        x = tp.tensor([1.0, 2.0])
        compiled = tp.compile(lambda a: tp.mul(plus_ten(a), 2.0))
        self.assertEqual(compiled(x).tolist(), [22.0, 24.0])


@unittest.skipIf(not _HAS_TRITON, "triton is unavailable")
class RealTritonJITFunctionTest(unittest.TestCase):
    """Level-2 plumbing against a genuine ``@triton.jit`` kernel object.

    JITFunctions; an actual launch additionally needs a supported GPU
    (sm_70+), which neither the local box nor the sm_61 remote provides.
    Everything that is launch-free is covered here with the real object:
    wrapper typing/idempotence and the ``triton_op`` capture contract --
    exactly one opaque fusion-barrier node whose body is never executed
    during tracing (the launch stays inside the triton_op boundary).
    """

    def _jit_kernel(self):
        @triton.jit
        def shift_kernel(x_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1.0,
                     mask=mask)

        return shift_kernel

    def test_wrap_accepts_real_jit_function(self):
        from triton.runtime.jit import JITFunction

        kernel = self._jit_kernel()
        self.assertIsInstance(kernel, JITFunction)
        wrapped = library.wrap_triton(kernel)
        self.assertIs(wrapped.kernel, kernel)

    def test_wrap_idempotent_on_real_jit(self):
        once = library.wrap_triton(self._jit_kernel())
        self.assertIs(library.wrap_triton(once), once)

    def test_triton_op_captures_single_opaque_node_without_running_body(self):
        wrapped = library.wrap_triton(self._jit_kernel())
        body_ran = []

        @library.triton_op("tritns::shift", mutates_args=())
        def shift(x):
            # Eager path: a real launch would go through `wrapped` here.
            body_ran.append(True)
            del wrapped
            return tp.add(x, 1.0)

        x = tp.tensor([1.0, 2.0])
        gm = tp.compiler.Tracer().trace(lambda a: shift(a), sample_inputs={"a": x})

        # The body never executed under the tracer: no launch, no proxy leak.
        self.assertEqual(body_ran, [])

        op_nodes = [
            n for n in gm.graph.nodes
            if n.op == "call_function" and isinstance(n.target, library.CustomOpDef)
        ]
        self.assertEqual(len(op_nodes), 1)
        target = op_nodes[0].target
        self.assertTrue(target.is_triton_op)
        self.assertIn("<triton_op tritns::shift>", repr(target))

        # The opaque node is a fusion barrier: it is the only compute node --
        # the inner add never leaked into the graph as separate ops.
        compute_nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
        self.assertEqual(len(compute_nodes), 1)


if __name__ == "__main__":
    unittest.main()
