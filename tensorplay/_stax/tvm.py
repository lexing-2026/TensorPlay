"""TVM compiler backend selected by ``tensorplay.compile(backend="tvm")``.


* Apache-TVM is an optional dependency; selecting this backend without it
  raises an actionable error instead of silently falling back.
* The canonical :class:`GraphModule` is translated into ``tvm.topi``
  compute DAGs; maximal pointwise chains are then fused into one kernel
  via ``compute_inline`` using the same fusibility whitelist the
  Stax/Triton generators share.
* Regions outside the supported subset fall back to the interpreter
  executor unless ``strict_native=True`` — identical contract to the Stax
  Triton path.  Training regions also keep the native path: TVM lowering
  currently covers inference.

Data crosses through DLPack without copies: TensorPlay tensors implement
``__dlpack__``/``from_dlpack``, TVM consumes both directions natively.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ..graph import GraphModule, Node
from ..graph._utils import _iter_nodes
from .stax import _is_scalar

__all__ = ["tvm", "has_tvm"]


def has_tvm() -> bool:
    """Return whether Apache-TVM is importable without importing it."""

    return importlib.util.find_spec("tvm") is not None


def _require_tvm():
    try:
        import tvm
        from tvm import topi

        return tvm, topi
    except ImportError as exc:
        raise RuntimeError(
            "Please install apache-tvm to use the tvm backend. "
            "See https://tvm.apache.org/docs/install/index.html for "
            "installation instructions."
        ) from exc


_DTYPE_TO_TVM = {
    "bool": "bool",
    "uint8": "uint8",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "float64": "float64",
}


def _dtype_name(dtype: Any) -> str | None:
    name = getattr(dtype, "name", None)
    if isinstance(name, str) and name in _DTYPE_TO_TVM:
        return name
    text = str(dtype)
    candidate = text.rsplit(".", 1)[-1]
    return candidate if candidate in _DTYPE_TO_TVM else None


def _shape_of(tensor: Any) -> tuple[int, ...]:
    """Read a tensor's shape across builds (callable or Size attribute)."""

    shape = tensor.shape
    if callable(shape):
        shape = shape()
    return tuple(int(d) for d in shape)


def _target_name(node_target: Any) -> str:
    return str(getattr(node_target, "__name__", node_target))


_BINARY = {"add", "sub", "mul", "div", "pow"}
_ALPHA_FORM = {"add", "sub"}


def _apply_binary(te: Any, name: str, lhs: Any, rhs: Any) -> Any:
    """Lower one binary pointwise op onto raw TIR expressions.

    Deliberately avoids ``topi``: the unity-line wheels lower several topi
    transcendentals through TVM's workspace pool, which segfaults when
    another OpenMP runtime (TensorPlay's) is already loaded.  Plain TIR
    intrinsics keep every kernel workspace-free.
    """

    import operator

    table = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "div": operator.truediv,
        "pow": te.power,
    }
    return table[name](lhs, rhs)


def _apply_unary(te: Any, name: str, x: Any, dtype: str) -> Any:
    if name == "neg":
        return -x
    if name == "pos":
        return x
    if name == "square":
        return x * x
    if name == "relu":
        return te.max(x, te.const(0, dtype))
    table = {
        "abs": te.abs,
        "sin": te.sin,
        "cos": te.cos,
        "exp": te.exp,
        "log": te.log,
        "sqrt": te.sqrt,
        "tanh": te.tanh,
    }
    if name in table:
        return table[name](x)
    if name == "sigmoid":
        return te.div(te.const(1, dtype), te.const(1, dtype) + te.exp(-x))
    raise ValueError(f"unsupported TVM lowering for op {name!r}")


def _input_spec(sample: Any) -> dict[str, Any] | None:
    """Capture the specialization contract of one example input."""

    try:
        shape = _shape_of(sample)
    except (TypeError, ValueError):
        return None
    dtype_name = _dtype_name(sample.dtype)
    if dtype_name is None or not sample.is_contiguous():
        return None
    return {
        "shape": shape,
        "dtype": dtype_name,
        "device": sample.device,
        "requires_grad": bool(sample.requires_grad),
    }


class _TvmKernel:
    """One built TVM function plus its runtime guards."""

    def __init__(
        self,
        func: Any,
        placeholders: list[Node],
        specs: list[dict[str, Any]],
        output_specs: list[dict[str, Any]],
        num_outputs: int,
    ) -> None:
        self.func = func
        self.placeholders = placeholders
        self.specs = specs
        self.output_specs = output_specs
        self.num_outputs = num_outputs

    def inputs_supported(self, tensors: list[Any]) -> bool:
        for spec, tensor in zip(self.specs, tensors):
            try:
                if (
                    _shape_of(tensor) != spec["shape"]
                    or _dtype_name(tensor.dtype) != spec["dtype"]
                    or tensor.device != spec["device"]
                    or not tensor.is_contiguous()
                ):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def run(self, tensors: list[Any]) -> Any:
        import tensorplay

        tvm_module, _ = _require_tvm()
        # Wheel-layout variance: classic TVM exposes ``tvm.nd.from_dlpack``;
        # newer wheels put ``from_dlpack`` on ``tvm.runtime``.  Both consume
        # any object implementing the DLPack protocol (tp.Tensor does).
        to_tvm = getattr(tvm_module, "nd", None)
        if to_tvm is not None:
            to_tvm = to_tvm.from_dlpack
        else:
            to_tvm = tvm_module.runtime.from_dlpack
        arrays_in = [to_tvm(tensor) for tensor in tensors]
        arrays_out = []
        outputs = []
        for spec in self.output_specs:
            dtype = getattr(tensorplay, spec["dtype"])
            out = tensorplay.empty(
                spec["shape"], dtype=dtype, device=spec["device"]
            )
            outputs.append(out)
            arrays_out.append(to_tvm(out))
        self.func(*(arrays_in + arrays_out))
        if self.num_outputs == 1:
            return outputs[0]
        return outputs


def _parallelize_cpu(tvm_module: Any, mod: Any) -> Any:
    """Fuse + parallelize each compute block over its flattened domain.

    Handles scheduler dialect variance: mainstream TVM names the accessor
    ``get_block``/``Schedule`` under ``tvm.tir``, while the 0.26 pip wheel
    ships ``get_sblock``/``Schedule`` under ``tvm.s_tir``.  Blocks that do
    not support flattening (pure loads, scalar stores) are skipped.
    """

    tir_mod = getattr(tvm_module, "tir", None) or tvm_module.s_tir
    schedule_mod = getattr(tir_mod, "schedule", tir_mod)
    try:
        sch = schedule_mod.Schedule(mod)
    except AttributeError:
        return mod

    def root_block():
        if hasattr(sch, "get_block"):
            return sch.get_block("root")
        return sch.get_sblock("root")

    try:
        blocks = sch.get_child_blocks(root_block())
    except Exception:  # noqa: BLE001 - scheduling is an optimization only
        return mod
    for block in blocks:
        try:
            loops = sch.get_loops(block)
            if len(loops) >= 1:
                sch.parallel(sch.fuse(*loops))
        except Exception:  # noqa: BLE001 - per-block tolerance
            continue
    try:
        return sch.mod
    except Exception:  # noqa: BLE001
        return mod


def _lower_pointwise(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    target: str,
    parallel: bool,
) -> _TvmKernel | None:
    """Translate the pointwise subset into one fused TVM kernel."""

    import tensorplay

    tvm_module, _topi = _require_tvm()
    from tvm import te

    placeholders = graph_module.graph.placeholders
    if len(placeholders) != len(example_inputs):
        return None
    specs = [_input_spec(sample) for sample in example_inputs]
    if any(spec is None for spec in specs):
        return None
    first = specs[0]
    # Shape/dtype/device-uniform inputs keep the v1 lowering honest: every
    # op in the whitelist is shape-preserving, so outputs inherit them.
    if any(
        spec["shape"] != first["shape"] or spec["dtype"] != first["dtype"]
        for spec in specs[1:]
    ):
        return None

    dtype = first["dtype"]
    shape = first["shape"]
    ph_tensors = {
        placeholder: te.placeholder(shape, dtype=dtype, name=placeholder.name)
        for placeholder in placeholders
    }
    refs: dict[Node, Any] = dict(ph_tensors)

    def value_expr(value: Any, idx: tuple[Any, ...]) -> Any:
        """Bind iteration indices: producer loads or scalar constants."""

        if isinstance(value, Node):
            if value not in refs:
                raise ValueError("node consumed before definition")
            return refs[value][idx]
        if _is_scalar(value):
            return te.const(float(value), dtype)
        raise ValueError(f"unsupported constant {value!r}")

    output_values = [
        node
        for out_node in graph_module.graph.outputs
        for node in _iter_nodes(out_node.args)
    ]
    if not output_values:
        return None

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op not in {"call_function", "call_method"} or node.kwargs:
            return None
        op_name = _target_name(node.target)
        args = list(node.args)

        def build(idx: tuple[Any, ...], *, _op: str = op_name, _args: list[Any] = args) -> Any:
            if _op in _ALPHA_FORM and len(_args) == 3:
                lhs, rhs, alpha = _args
                if not _is_scalar(alpha):
                    raise ValueError("alpha must be scalar")
                rhs_expr = value_expr(rhs, idx)
                if alpha != 1:
                    rhs_expr = rhs_expr * te.const(float(alpha), dtype)
                return _apply_binary(te, _op, value_expr(lhs, idx), rhs_expr)
            if _op in _BINARY:
                if len(_args) != 2:
                    raise ValueError(f"{_op} expects two operands")
                return _apply_binary(
                    te, _op, value_expr(_args[0], idx), value_expr(_args[1], idx)
                )
            if len(_args) != 1:
                raise ValueError(f"{_op} expects one operand")
            return _apply_unary(te, _op, value_expr(_args[0], idx), dtype)

        # One te.compute per node over raw TIR intrinsics; create_prim_func
        # inlines the pure chain into a single kernel; the fusion scheduler
        # performs this transformation. (No topi: its transcendentals route through
        # TVM's workspace pool, which segfaults next to TensorPlay's OpenMP.)
        # fcompute runs eagerly, so closing over ``build`` is safe here.
        refs[node] = te.compute(shape, lambda *idx: build(idx), name=node.name)

    outs = [refs[value] for value in output_values]

    # Modern TVM (>=0.16): create_prim_func lowers the elementwise DAG into
    # one TIR prim func, inlining pure producers — the compute_inline fusion
    # the legacy schedule API expressed manually.  Parallelize CPU kernels
    # over their flattened domain via the block-level tir scheduler.
    mod = te.create_prim_func([*ph_tensors.values(), *outs])
    if parallel and target.startswith("llvm"):
        mod = _parallelize_cpu(tvm_module, mod)
    func = tvm_module.build(mod, target=target)
    return _TvmKernel(
        func=func,
        placeholders=placeholders,
        specs=list(specs),
        output_specs=[dict(first) for _ in outs],
        num_outputs=len(outs),
    )


def tvm(
    graph_module: GraphModule,
    example_inputs: list[Any],
    **options: Any,
):
    """Compile ``graph_module`` with Apache-TVM (backend entry point).

    Options:
        ``target``: TVM target string override (default ``"cuda"`` for CUDA
        inputs, ``"llvm"`` otherwise).
        ``parallel``: parallelize CPU kernels over the flattened domain
        repeated in-process builds with TVM's thread pool can crash).

    Unsupported graphs return an interpreter-backed callable unless
    ``strict_native=True``; the strict mode contract matches the Stax Triton
    backend.
    """

    if not example_inputs:
        raise ValueError("the tvm backend needs at least one input")
    tvm_module, _ = _require_tvm()

    strict_native = bool(options.pop("strict_native", False))
    target = options.pop("target", None)
    parallel = options.pop("parallel", False)
    if options:
        raise RuntimeError(f"Unexpected tvm backend option(s): {sorted(options)!r}")

    fallback = None if strict_native else graph_module.recompile()
    try:
        first_device = example_inputs[0].device
        resolved_target = target or (
            "cuda" if first_device.is_cuda() else "llvm"
        )
        any_grad = any(
            bool(getattr(sample, "requires_grad", False))
            for sample in example_inputs
        )
        kernel = (
            None
            if any_grad
            else _lower_pointwise(
                graph_module,
                example_inputs,
                target=resolved_target,
                parallel=parallel,
            )
        )
    except (ValueError, KeyError, NotImplementedError):
        kernel = None

    if kernel is None:
        if strict_native:
            raise RuntimeError(
                "strict_native=True could not lower the captured graph to "
                "TVM"
            )
        assert fallback is not None
        return fallback

    placeholders = kernel.placeholders

    def compiled(*args: Any, **kwargs: Any) -> Any:
        if not kwargs and len(args) == len(placeholders):
            inputs = list(args)
        else:
            bound = graph_module.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            inputs = [
                bound.arguments[node.target if isinstance(node.target, str) else node.name]
                for node in placeholders
            ]
        if not kernel.inputs_supported(inputs):
            if strict_native:
                raise RuntimeError(
                    "TVM strict_native lowering received inputs outside "
                    "its compiled specialization"
                )
            assert fallback is not None
            return fallback(*args, **kwargs)
        return kernel.run(inputs)

    compiled._tensorplay_codegen = "tvm"  # type: ignore[attr-defined]
    compiled._tensorplay_target = resolved_target  # type: ignore[attr-defined]
    del tvm_module
    return compiled
