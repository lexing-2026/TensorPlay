"""ONNX export for TensorPlay.

Export happens in two stages: capture a model into an
:class:`tensorplay.export.ExportedProgram` (via :func:`tensorplay.export`), then
translate the resulting graph into an ONNX ``ModelProto``.

The translation runs the captured graph once on the recorded example inputs so
every intermediate value carries a shape and dtype.  Handlers registered in
:mod:`tensorplay.onnx._composite_ops` use that metadata to choose between
lowerings that differ only by rank or shape (``Gemm`` vs ``MatMul``,
``GlobalAveragePool`` vs ``AveragePool``, ``perm`` vectors, ...).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping, Sequence

from onnx import TensorProto, checker, helper, numpy_helper, shape_inference

import tensorplay as tp

from ..export import ExportedProgram, export as tp_export
from ..export.dynamic_shapes import _DimHint, _DimHintType
from ..graph._utils import _iter_nodes
from . import _external_data, _passes, errors, testing, utils, verification
from ._composite_ops import (
    GraphBuilder,
    OpContext,
    Value,
    lookup_function_handler,
    lookup_method_handler,
)
from ._type_mapping import (
    _dtype_to_numpy,
    _np_dtype_to_onnx,
    _size_to_tuple,
    _to_numpy,
)
from ._verify import VerificationError, VerificationResult, verify_model
from .errors import (
    OnnxExporterError,
    OnnxExporterWarning,
    UnsupportedOperatorError,
)

__all__ = [
    "DEFAULT_OPSET_VERSION",
    "MIN_OPSET_VERSION",
    "OnnxExporterError",
    "OnnxExporterWarning",
    "UnsupportedOperatorError",
    "VerificationError",
    "VerificationResult",
    "errors",
    "export",
    "is_supported",
    "testing",
    "utils",
    "verification",
]

DEFAULT_OPSET_VERSION = 18
MIN_OPSET_VERSION = 13

#: Factories whose result is fixed by their arguments alone.  The ones that
#: leave their contents unset or draw them at random are not among them.
_SETTLED_FACTORIES = frozenset(
    {
        "arange",
        "eye",
        "full",
        "full_like",
        "linspace",
        "logspace",
        "new_full",
        "new_ones",
        "new_zeros",
        "ones",
        "ones_like",
        "zeros",
        "zeros_like",
    }
)


# ---------------------------------------------------------------------------
# Shape / dtype propagation
# ---------------------------------------------------------------------------


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "numpy")


def _propagate_metadata(
    graph_module: Any, example_inputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the captured graph once to learn every node's shape and dtype."""

    from ..graph.interpreter import Interpreter

    try:
        interpreter = Interpreter(graph_module, garbage_collect_values=False)
        interpreter.run(**dict(example_inputs))
    except Exception as exc:  # noqa: BLE001 - metadata is best effort
        warnings.warn(
            "ONNX export could not evaluate the captured graph on the example "
            f"inputs ({type(exc).__name__}: {exc}); shape-dependent operators "
            "will fail to lower.",
            RuntimeWarning,
            stacklevel=3,
        )
        return {}
    return {node.name: value for node, value in interpreter.env.items()}


def _annotate(result: Any, sample: Any) -> Any:
    """Attach the sampled shape/dtype to the value(s) a handler produced.

    A handler names an ONNX value with a string; anything else it returns
    (a :class:`Value`, or a plain Python value such as a fixed extent) is
    already what later nodes consume.
    """

    if isinstance(result, (list, tuple)):
        samples = sample if isinstance(sample, (list, tuple)) else ()
        annotated = [
            _annotate(item, samples[index] if index < len(samples) else None)
            for index, item in enumerate(result)
        ]
        return tuple(annotated) if isinstance(result, tuple) else annotated
    if not isinstance(result, str):
        return result
    if _is_tensor(sample):
        return Value(result, _size_to_tuple(sample.shape), _numpy_dtype(sample))
    if _is_scalar(sample):
        return Value(result, (), _to_numpy(sample).dtype)
    return Value(result)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _holds(value: Any, predicate: Any) -> bool:
    """Whether ``value`` or anything nested in its containers satisfies ``predicate``."""

    if predicate(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds(item, predicate) for item in value)
    if isinstance(value, dict):
        return any(_holds(item, predicate) for item in value.values())
    if isinstance(value, slice):
        return any(
            _holds(item, predicate) for item in (value.start, value.stop, value.step)
        )
    return False


def _numpy_dtype(tensor: Any) -> Any:
    # Annotate from the dtype metadata first: a device-resident sample cannot
    # be materialized on the host without a transfer, and reading an input's
    # type must never force one.  Exotic dtypes fall back to host
    # materialization (bfloat16 rounds through float32 there).
    dtype = getattr(tensor, "dtype", None)
    if dtype is not None:
        try:
            return _dtype_to_numpy(dtype)
        except TypeError:
            pass
    try:
        return _to_numpy(tensor).dtype
    except Exception:  # noqa: BLE001 - exotic dtypes stay unannotated
        return None


#: Initializers at most this large keep their data in the graph shape inference
#: sees: shape vectors, axes and scalars, which inference reads.  Larger ones
#: (weights) are only typed, which is all inference needs from them.
_INFERENCE_DATA_LIMIT = 64


def _infer_partial_shapes(
    builder: GraphBuilder, graph_inputs: Sequence[Any]
) -> dict[str, list[int | None]]:
    """Per-value extents ONNX shape inference finds in the graph built so far.

    An extent comes back as an ``int`` where inference settles it and ``None``
    where it stays symbolic or unknown.
    """

    initializers: list[Any] = []
    typed: list[Any] = []
    for tensor in builder.initializers:
        if math.prod(tensor.dims) <= _INFERENCE_DATA_LIMIT:
            initializers.append(tensor)
        else:
            typed.append(
                helper.make_tensor_value_info(
                    tensor.name, tensor.data_type, list(tensor.dims)
                )
            )
    graph = helper.make_graph(
        builder.nodes,
        builder.name,
        list(graph_inputs) + typed,
        [],
        initializer=initializers,
        value_info=builder.value_info,
    )
    opset = [helper.make_opsetid("", builder.opset)]
    model = helper.make_model(graph, opset_imports=opset)
    model.ir_version = helper.find_min_ir_version_for(opset, ignore_unknown=True)
    try:
        inferred = shape_inference.infer_shapes(
            model, check_type=False, strict_mode=False, data_prop=True
        )
    except Exception:  # noqa: BLE001 - every extent then counts as dynamic
        return {}
    shapes: dict[str, list[int | None]] = {}
    for info in (*inferred.graph.input, *inferred.graph.value_info):
        tensor_type = info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        shapes[info.name] = [
            dim.dim_value if dim.HasField("dim_value") else None
            for dim in tensor_type.shape.dim
        ]
    return shapes


# ---------------------------------------------------------------------------
# Graph conversion
# ---------------------------------------------------------------------------


class _Converter:
    """Walks the captured graph and emits the equivalent ONNX nodes."""

    def __init__(
        self,
        graph_module: Any,
        example_inputs: Mapping[str, Any],
        *,
        opset_version: int,
        input_names: Sequence[str] | None,
        output_names: Sequence[str] | None,
        dynamic_axes: Mapping[str, Any] | None,
        state_values: Mapping[str, Any] | None = None,
        num_mutations: int = 0,
    ) -> None:
        self.graph_module = graph_module
        self.example_inputs = dict(example_inputs)
        self.state_values = dict(state_values or {})
        self.num_mutations = int(num_mutations or 0)
        self.input_names = list(input_names) if input_names else None
        self.output_names = list(output_names) if output_names else None
        self.dynamic_axes = dict(dynamic_axes or {})
        self.builder = GraphBuilder(opset_version)
        self.env: dict[str, Any] = {}
        self.samples = _propagate_metadata(
            graph_module, {**self.state_values, **self.example_inputs}
        )
        self.graph_inputs: list[Any] = []
        self.eager_outputs: Any = None
        self._inferred: tuple[int, dict[str, list[int | None]]] | None = None
        self.builder.static_dims = self._static_dims

    # -- extents ------------------------------------------------------------

    def _static_dims(self, value: Value) -> list[int | None] | None:
        """Which extents of ``value`` the exported graph fixes.

        Without a dynamic input axis every sampled extent is fixed.  With one,
        ONNX shape inference over the graph built so far decides: an extent it
        settles to a number is fixed in the exported model, and any other
        (symbolic, or beyond what inference can follow) is read at runtime.
        """

        if value.shape is None:
            return None
        if not any(
            dim.HasField("dim_param")
            for info in self.graph_inputs
            for dim in info.type.tensor_type.shape.dim
        ):
            return list(value.shape)
        return self._inferred_shapes().get(value.name)

    def _inferred_shapes(self) -> dict[str, list[int | None]]:
        count = len(self.builder.nodes)
        if self._inferred is None or self._inferred[0] != count:
            self._inferred = (count, _infer_partial_shapes(self.builder, self.graph_inputs))
        return self._inferred[1]

    # -- helpers ------------------------------------------------------------

    def _resolve(self, value: Any) -> Any:
        from ..graph.node import Node

        if isinstance(value, Node):
            try:
                return self.env[value.name]
            except KeyError:  # pragma: no cover - lint guarantees ordering
                raise UnsupportedOperatorError(
                    f"value {value.name!r} is used before it is produced"
                ) from None
        if isinstance(value, tuple):
            return tuple(self._resolve(item) for item in value)
        if isinstance(value, list):
            return [self._resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                self._resolve(value.start),
                self._resolve(value.stop),
                self._resolve(value.step),
            )
        return value

    @staticmethod
    def _target_id(target: Any) -> tuple[str, str]:
        module = getattr(target, "__module__", "") or ""
        name = (
            getattr(target, "__qualname__", "")
            or getattr(target, "__name__", "")
            or repr(target)
        )
        return module, name.rsplit(".", 1)[-1]

    # -- node kinds ---------------------------------------------------------

    def _placeholder(self, node: Any, index: int) -> Any:
        name = (
            self.input_names[index]
            if self.input_names is not None and index < len(self.input_names)
            else node.name
        )
        self.builder.reserve(name)
        sample = self.example_inputs.get(node.name)
        if sample is None:
            raise UnsupportedOperatorError(
                f"input {node.name!r} has no example value; export the program "
                "with example inputs for every argument"
            )
        if _is_tensor(sample):
            value = Value(name, _size_to_tuple(sample.shape), _numpy_dtype(sample))
        else:
            array = _to_numpy(sample)
            value = Value(name, tuple(array.shape), array.dtype)
        info = self._value_info(name, value)
        if info is None:
            raise UnsupportedOperatorError(
                f"input {node.name!r} has an unsupported example value of type "
                f"{type(sample).__name__}"
            )
        self.graph_inputs.append(info)
        return value

    def _get_attr(self, node: Any) -> Any:
        attribute = self.graph_module._get_attr(str(node.target))
        if not _is_tensor(attribute):
            return attribute
        name = self.builder.unique(str(node.target).replace(".", "_"))
        array = _to_numpy(attribute)
        self.builder.initializers.append(numpy_helper.from_array(array, name))
        return Value(name, tuple(array.shape), array.dtype)

    def _state_initializer(self, node: Any) -> Any:
        """Emit a lifted state placeholder as a constant initializer."""

        value = self.state_values[node.name]
        if not _is_tensor(value):
            return Value(str(value))
        name = self.builder.unique(node.name)
        array = _to_numpy(value)
        self.builder.initializers.append(numpy_helper.from_array(array, name))
        return Value(name, tuple(array.shape), array.dtype)

    def _settled_factory(self, node: Any) -> Any:
        """The tensor a factory call makes, when its arguments settle it.

        A capture records ``zeros((2,), dtype=int64)`` as a call so each run
        of the program makes its own tensor.  An exported model holds values,
        not storage, and a factory whose arguments are all plain values makes
        the same values every time: those go into the model as an initializer.
        A factory that draws at random, leaves its contents unset, or takes
        its extent from the graph is not settled and is lowered as a call.
        """

        if node.op != "call_function":
            return None
        name = getattr(node.target, "__name__", "")
        if name not in _SETTLED_FACTORIES:
            return None
        if any(True for _ in _iter_nodes((node.args, node.kwargs))):
            return None
        sample = self.samples.get(node.name)
        return sample if _is_tensor(sample) else None

    def _call(self, node: Any) -> Any:
        settled = self._settled_factory(node)
        if settled is not None:
            name = self.builder.unique(node.name)
            array = _to_numpy(settled)
            self.builder.initializers.append(numpy_helper.from_array(array, name))
            return Value(name, tuple(array.shape), array.dtype)

        args = [self._resolve(arg) for arg in node.args]
        kwargs = {key: self._resolve(value) for key, value in node.kwargs.items()}
        sample = self.samples.get(node.name)

        if (
            sample is not None
            and not _holds(sample, _is_tensor)
            and not _holds((args, kwargs), lambda item: isinstance(item, Value))
        ):
            # Python arithmetic on fixed extents (``c // heads``, indexing a
            # shape tuple, ...) touches no value the model computes, so what
            # it produced on the example inputs is what it always produces.
            return sample

        if node.op == "call_function":
            module, name = self._target_id(node.target)
            entry = lookup_function_handler(module, name)
            description = f"{module}.{name}" if module else name
        else:
            name = str(node.target)
            entry = lookup_method_handler(name)
            description = f"Tensor.{name}"

        if entry is None:
            raise UnsupportedOperatorError(
                f"{description} has no ONNX lowering; register one in "
                "tensorplay/onnx/_composite_ops.py or rewrite the model to use "
                "a supported operator"
            )
        handler, params = entry
        out_shape = out_dtype = None
        if _is_tensor(sample):
            out_shape, out_dtype = _size_to_tuple(sample.shape), _numpy_dtype(sample)
        elif _is_scalar(sample):
            out_shape, out_dtype = (), _to_numpy(sample).dtype
        context = OpContext(
            self.builder,
            node.name,
            params,
            args,
            kwargs,
            out_shape=out_shape,
            out_dtype=out_dtype,
            sample=sample,
        )
        try:
            result = handler(context)
        except UnsupportedOperatorError:
            raise
        except Exception as exc:  # noqa: BLE001 - point at the offending node
            raise UnsupportedOperatorError(
                f"failed to lower {description} (node {node.name!r}): {exc}"
            ) from exc
        return _annotate(result, sample)

    # -- outputs ------------------------------------------------------------

    def _value_info(self, name: str, value: Value) -> Any:
        if value.dtype is None:
            return None
        shape = list(value.shape) if value.shape is not None else None
        spec = self.dynamic_axes.get(name)
        if shape is not None and spec is not None:
            if isinstance(spec, Mapping):
                for axis, axis_name in spec.items():
                    if 0 <= int(axis) < len(shape):
                        shape[int(axis)] = str(axis_name)
            else:
                for axis in spec:
                    if 0 <= int(axis) < len(shape):
                        shape[int(axis)] = f"{name}_dim_{int(axis)}"
        return helper.make_tensor_value_info(
            name, _np_dtype_to_onnx(value.dtype), shape
        )

    def _flatten_outputs(self, value: Any) -> list[Value]:
        if isinstance(value, (list, tuple)):
            flattened: list[Value] = []
            for item in value:
                flattened.extend(self._flatten_outputs(item))
            return flattened
        if isinstance(value, Value):
            return [value]
        return [Value(str(value))]

    # -- driver -------------------------------------------------------------

    def convert(self) -> Any:
        placeholder_index = 0
        outputs: list[Value] = []
        checks = _runtime_checks(self.graph_module.graph)
        for node in self.graph_module.graph.nodes:
            if node in checks:
                continue
            if node.op == "placeholder":
                if node.name in self.state_values:
                    # lifted state becomes a constant, not a graph input
                    self.env[node.name] = self._state_initializer(node)
                    continue
                self.env[node.name] = self._placeholder(node, placeholder_index)
                placeholder_index += 1
            elif node.op == "get_attr":
                self.env[node.name] = self._get_attr(node)
            elif node.op in ("call_function", "call_method"):
                self.env[node.name] = self._call(node)
            elif node.op == "output":
                flattened = self._flatten_outputs([self._resolve(node.args[0])])
                outputs = flattened[self.num_mutations:]
                self.eager_outputs = self.samples.get(node.name)
            elif node.op == "call_module":
                raise UnsupportedOperatorError(
                    f"call_module node {node.target!r} reached the ONNX exporter; "
                    "export inlines submodules, so this graph was captured with a "
                    "tracer that keeps module boundaries"
                )
            else:  # pragma: no cover - Graph.lint rejects other kinds
                raise UnsupportedOperatorError(f"unsupported node kind {node.op!r}")

        graph_outputs = []
        seen: set[str] = set()
        input_names = {info.name for info in self.graph_inputs}
        for index, value in enumerate(outputs):
            if self.output_names is not None and index < len(self.output_names):
                requested = self.builder.reserve(self.output_names[index])
                self.builder.op("Identity", [value.name], outputs=[requested])
                value = Value(requested, value.shape, value.dtype)
            elif value.name in seen or value.name in input_names:
                # A value returned twice (or returned unchanged) still needs a
                # distinct graph output name.
                copied = self.builder.unique(f"{value.name}_out")
                self.builder.op("Identity", [value.name], outputs=[copied])
                value = Value(copied, value.shape, value.dtype)
            seen.add(value.name)
            info = self._value_info(value.name, value)
            if info is None:
                info = helper.make_tensor_value_info(
                    value.name, TensorProto.UNDEFINED, None
                )
            graph_outputs.append(info)

        return helper.make_graph(
            self.builder.nodes,
            self.builder.name,
            self.graph_inputs,
            graph_outputs,
            initializer=self.builder.initializers,
            value_info=self.builder.value_info,
        )


_RUNTIME_CHECKS = frozenset(
    {"_assert_dim_range", "_assert_dims_equal", "_assert_shape_guard", "_assert_dim_relation"}
)


def _runtime_checks(graph: Any) -> set[Any]:
    """The program's runtime checks, and what only they consume.

    The checks guard eager runs of the program; an ONNX model declares its
    dimensions instead, and ONNX has no operator that fails a run, so
    neither they nor the computation feeding them become ONNX nodes.
    """

    checks: set[Any] = set()
    for node in reversed(list(graph.nodes)):
        if node.op not in ("call_function", "call_method"):
            continue
        is_check = node.target is tp._assert or (
            getattr(node.target, "__module__", None) == "tensorplay.export._trace"
            and getattr(node.target, "__name__", None) in _RUNTIME_CHECKS
        )
        if is_check or (node.users and all(user in checks for user in node.users)):
            checks.add(node)
    return checks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _to_exported_program(
    model: Any,
    dynamic_axes: Any,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> tuple[ExportedProgram, Any]:
    if isinstance(model, ExportedProgram):
        declared = _dynamic_shapes_to_dynamic_axes(model, input_names)
        if dynamic_axes is None:
            return model, declared
        _check_declared_axes(model, dynamic_axes, declared or {}, input_names)
        return model, dynamic_axes
    if isinstance(model, (list, tuple)) and model:
        callable_, *rest = model
        kwargs: dict[str, Any] = {}
        if rest and isinstance(rest[-1], dict):
            kwargs = dict(rest.pop())
        if dynamic_axes and "dynamic_shapes" not in kwargs:
            # The capture has to know which input extents vary, or a program
            # that reads them would keep the example's sizes.
            dynamic_shapes = _dynamic_axes_to_dynamic_shapes(
                callable_, rest, kwargs, dynamic_axes, input_names, output_names
            )
            if dynamic_shapes:
                kwargs["dynamic_shapes"] = dynamic_shapes
        program = tp_export(callable_, *rest, **kwargs)
        if dynamic_axes is None:
            dynamic_axes = _dynamic_shapes_to_dynamic_axes(program, input_names)
        return program, dynamic_axes
    raise TypeError(
        "expected an ExportedProgram or a (model, *args, kwargs) sequence, got "
        f"{type(model).__name__}"
    )


def _program_state_values(program: Any) -> dict[str, Any]:
    """Resolve lifted state placeholder names to their tensor values."""

    from ..export.graph_signature import InputKind

    root = program.graph_module.root
    values: dict[str, Any] = {}
    for spec in program.graph_signature.input_specs:
        if spec.kind is InputKind.USER_INPUT or not isinstance(spec.target, str):
            continue
        value: Any = root
        try:
            for atom in spec.target.split("."):
                value = getattr(value, atom)
        except AttributeError:
            continue
        values[spec.arg.name] = value
    return values


def export(
    exported_program: ExportedProgram | Any,
    f: Any = None,
    *,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    opset_version: int | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str] | Sequence[int]] | None = None,
    do_constant_folding: bool = True,
    verify: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    external_data: bool | None = None,
    external_data_location: str | None = None,
    check_model: bool = True,
) -> Any:
    """Export a TensorPlay model to ONNX.

    Args:
        exported_program: an :class:`~tensorplay.export.ExportedProgram`, or a
            ``(model, *args, kwargs)`` sequence captured on the fly.
        f: file path or writable binary file object.  When omitted the
            ``ModelProto`` is returned instead of being written.
        input_names: names for the graph inputs, in placeholder order.
        output_names: names for the graph outputs.
        opset_version: target ONNX opset (default 18, minimum 13).
        dynamic_axes: ``{value_name: {axis: axis_name}}`` (or a list of axis
            indices) marking dimensions that vary at runtime.  Applies to both
            inputs and outputs.
        do_constant_folding: fold subgraphs whose inputs are all constants.
        verify: run the exported model under onnxruntime and compare against
            eager execution of ``exported_program``.
        rtol/atol: tolerances used by ``verify``.
        external_data: store initializers in a side-car file.  ``None`` decides
            from the model size (models at or above the 2 GiB protobuf limit).
        external_data_location: side-car file name for ``external_data``.
        check_model: run ``onnx.checker`` over the finished model.

    Returns:
        The :class:`onnx.ModelProto` when ``f`` is ``None``, else ``None``.
    """

    program, dynamic_axes = _to_exported_program(
        exported_program, dynamic_axes, input_names, output_names
    )
    opset = DEFAULT_OPSET_VERSION if opset_version is None else int(opset_version)
    if opset < MIN_OPSET_VERSION:
        raise ValueError(
            f"opset_version must be >= {MIN_OPSET_VERSION}, got {opset}"
        )

    converter = _Converter(
        program.graph_module,
        program.example_inputs,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        state_values=_program_state_values(program),
        num_mutations=int(
            getattr(program.graph_module, "meta", {}).get("num_mutations", 0) or 0
        ),
    )
    graph = converter.convert()

    model = helper.make_model(
        graph,
        producer_name="tensorplay",
        producer_version=_producer_version(),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    # Declaring a newer IR version than the opset needs makes older runtimes
    # reject an otherwise valid model.
    model.ir_version = helper.find_min_ir_version_for(
        [helper.make_opsetid("", opset)], ignore_unknown=True
    )

    _passes.optimize(model, do_constant_folding=do_constant_folding)

    try:
        model = shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:  # noqa: BLE001 - inference is advisory
        pass

    if check_model and not _external_data.needs_external_data(model):
        checker.check_model(model)

    if verify:
        expected = converter.eager_outputs
        if expected is None:
            expected = program(**program.example_inputs)
        verify_model(
            model,
            expected=expected,
            input_names=[value.name for value in model.graph.input],
            example_inputs=program.example_inputs,
            rtol=rtol,
            atol=atol,
        )

    if f is not None:
        _external_data.save_model(
            model,
            f,
            external_data=external_data,
            location=external_data_location,
        )
        return None
    return model


def is_supported(target: Any) -> bool:
    """Whether a captured ``call_function`` target has an ONNX lowering."""

    module = getattr(target, "__module__", "") or ""
    name = getattr(target, "__qualname__", "") or getattr(target, "__name__", "")
    return lookup_function_handler(module, name.rsplit(".", 1)[-1]) is not None


def _producer_version() -> str:
    try:
        from ..version import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - version metadata is optional
        return "dev"


def _dynamic_axes_to_dynamic_shapes(
    model: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    dynamic_axes: Mapping[str, Any],
    input_names: Sequence[str] | None,
    output_names: Sequence[str] | None,
) -> dict[str, dict[int, Any]] | None:
    """Translate ONNX ``dynamic_axes`` into ``export(dynamic_shapes=...)``.

    Every axis named for an input becomes ``Dim.DYNAMIC``: dynamic, with the
    range left for export to settle, and an error if the program fixes it.
    Inputs are named by ``input_names`` in argument order, or by their
    argument names.
    """

    from ..export import Dim
    from ..export.dynamic_shapes import _combine_args

    arguments = list(_combine_args(model, tuple(args), kwargs))
    by_name = {
        (input_names[index] if input_names and index < len(input_names) else name): name
        for index, name in enumerate(arguments)
    }
    outputs = set(output_names or ())
    shapes: dict[str, dict[int, Any]] = {}
    for name, axes in dynamic_axes.items():
        argument = by_name.get(name, name if name in arguments else None)
        if name in outputs or argument is None or axes is None:
            continue
        shapes[argument] = {int(axis): Dim.DYNAMIC for axis in axes}
    return shapes or None


def _onnx_input_names(
    program: ExportedProgram, input_names: Sequence[str] | None
) -> dict[str, str]:
    """Each argument of the program by the name of the ONNX input it becomes:
    ``input_names`` in order, else the input placeholder's name."""

    user_inputs = set(program.graph_signature.user_inputs)
    placeholders = [
        node for node in program.graph_module.graph.placeholders if node.name in user_inputs
    ]
    # User placeholders follow the callable's arguments in order.
    signature = program.graph_module.meta.get("user_signature")
    arguments = list(signature.parameters) if signature is not None else []
    onnx_names: dict[str, str] = {}
    for index, node in enumerate(placeholders):
        argument = arguments[index] if index < len(arguments) else str(node.target)
        named = input_names is not None and index < len(input_names)
        onnx_names[argument] = input_names[index] if named else node.name
    return onnx_names


def _check_declared_axes(
    program: ExportedProgram,
    dynamic_axes: Mapping[str, Any],
    declared: Mapping[str, Mapping[int, str]],
    input_names: Sequence[str] | None,
) -> None:
    """Reject ``dynamic_axes`` that let an input vary where the program fixed it.

    A program reads the shapes of the values it computes at the sizes its
    declaration allows: an extent of an input it did not declare dynamic is the
    example's in everything computed from it.  Marking that ONNX input dynamic
    would export a model that is wrong at every other size -- unless the
    program never read such an extent, as a stack of layers that only maps its
    input does not.
    """

    read = program.graph_module.meta.get("fixed_inputs")
    if read is None:
        return
    fixed: list[str] = []
    for argument, onnx_name in _onnx_input_names(program, input_names).items():
        axes = dynamic_axes.get(onnx_name)
        if axes is None or argument not in read:
            continue
        allowed = declared.get(onnx_name, {})
        for axis in axes:
            if int(axis) not in allowed:
                fixed.append(f"{onnx_name!r} axis {int(axis)} (argument {argument!r})")
    if fixed:
        raise ValueError(
            "dynamic_axes marks "
            + ", ".join(fixed)
            + " as varying, but the program was exported with those dimensions fixed "
            "and reads shapes that depend on them, so it holds only at the example's "
            "sizes. Export it with "
            "dynamic_shapes declaring them (Dim.DYNAMIC, or a named Dim), or pass "
            "(model, *args) to this export to capture it with these axes."
        )


def _dynamic_shapes_to_dynamic_axes(
    program: ExportedProgram, input_names: Sequence[str] | None
) -> dict | None:
    """The ONNX ``dynamic_axes`` a program's ``dynamic_shapes`` declare.

    The specification names the program's arguments; the axes name the ONNX
    inputs those arguments become (``input_names`` in order, else the input
    placeholders' names).
    """

    dynamic_shapes = program.dynamic_shapes
    if not dynamic_shapes:
        return None
    onnx_names = _onnx_input_names(program, input_names)
    result: dict[str, dict[int, str]] = {}
    for argument, dims in dynamic_shapes.items():
        axes: dict[int, str] = {}
        entries = dims.items() if isinstance(dims, Mapping) else enumerate(dims or ())
        for axis, spec in entries:
            if hasattr(spec, "name") and not isinstance(spec, int):
                axes[int(axis)] = str(spec.name)
            elif isinstance(spec, _DimHint) and spec.type is not _DimHintType.STATIC:
                axes[int(axis)] = f"{argument}_dim_{int(axis)}"
        if axes:
            result[onnx_names.get(argument, argument)] = axes
    return result or None
