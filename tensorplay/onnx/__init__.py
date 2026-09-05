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

import warnings
from typing import Any, Mapping, Sequence

from onnx import TensorProto, checker, helper, numpy_helper, shape_inference

from ..export import ExportedProgram, export as tp_export
from . import _external_data, _passes
from ._composite_ops import (
    GraphBuilder,
    OpContext,
    UnsupportedOperatorError,
    Value,
    lookup_function_handler,
    lookup_method_handler,
)
from ._type_mapping import _np_dtype_to_onnx, _size_to_tuple, _to_numpy
from ._verify import VerificationError, VerificationResult, verify_model

__all__ = [
    "DEFAULT_OPSET_VERSION",
    "MIN_OPSET_VERSION",
    "UnsupportedOperatorError",
    "VerificationError",
    "VerificationResult",
    "export",
    "is_supported",
]

DEFAULT_OPSET_VERSION = 18
MIN_OPSET_VERSION = 13


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
    """Attach the sampled shape/dtype to the value(s) a handler produced."""

    if isinstance(result, (list, tuple)):
        samples = sample if isinstance(sample, (list, tuple)) else ()
        return [
            _annotate(item, samples[index] if index < len(samples) else None)
            for index, item in enumerate(result)
        ]
    if isinstance(result, Value):
        return result
    if not _is_tensor(sample):
        return Value(result)
    return Value(result, _size_to_tuple(sample.shape), _numpy_dtype(sample))


def _numpy_dtype(tensor: Any) -> Any:
    try:
        return _to_numpy(tensor).dtype
    except Exception:  # noqa: BLE001 - exotic dtypes stay unannotated
        return None


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

    def _call(self, node: Any) -> Any:
        args = [self._resolve(arg) for arg in node.args]
        kwargs = {key: self._resolve(value) for key, value in node.kwargs.items()}

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
        sample = self.samples.get(node.name)
        context = OpContext(
            self.builder,
            node.name,
            params,
            args,
            kwargs,
            out_shape=_size_to_tuple(sample.shape) if _is_tensor(sample) else None,
            out_dtype=_numpy_dtype(sample) if _is_tensor(sample) else None,
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
        for node in self.graph_module.graph.nodes:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _to_exported_program(model: Any, dynamic_axes: Any) -> tuple[ExportedProgram, Any]:
    if isinstance(model, ExportedProgram):
        if dynamic_axes is None and model.dynamic_shapes:
            dynamic_axes = _dynamic_shapes_to_dynamic_axes(
                model.dynamic_shapes, model.graph_signature.user_inputs
            )
        return model, dynamic_axes
    if isinstance(model, (list, tuple)) and model:
        callable_, *rest = model
        kwargs: dict[str, Any] = {}
        if rest and isinstance(rest[-1], dict):
            kwargs = dict(rest.pop())
        if "dynamic_shapes" in kwargs and dynamic_axes is None:
            dynamic_axes = _dynamic_shapes_to_dynamic_axes(
                kwargs["dynamic_shapes"], None
            )
        program = tp_export(callable_, *rest, **kwargs)
        return program, dynamic_axes
    raise TypeError(
        "expected an ExportedProgram or a (model, *args, kwargs) sequence, got "
        f"{type(model).__name__}"
    )


def _program_state_values(program: Any) -> dict[str, Any]:
    """Resolve lifted state placeholder names to their tensor values."""

    from .export.graph_signature import InputKind

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

    program, dynamic_axes = _to_exported_program(exported_program, dynamic_axes)
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


def _dynamic_shapes_to_dynamic_axes(
    dynamic_shapes: Mapping[str, Mapping[int, Any]],
    user_inputs: Sequence[str] | None,
) -> dict | None:
    """Translate ``export(dynamic_shapes=...)`` into ONNX ``dynamic_axes``."""

    if not dynamic_shapes:
        return None
    names = list(user_inputs or dynamic_shapes.keys())
    result: dict[str, dict[int, str]] = {}
    for index, (argument, dims) in enumerate(dynamic_shapes.items()):
        axes: dict[int, str] = {}
        for axis, spec in dims.items():
            if hasattr(spec, "name"):
                axes[int(axis)] = str(spec.name)
        if axes:
            result[names[index] if index < len(names) else argument] = axes
    return result or None
