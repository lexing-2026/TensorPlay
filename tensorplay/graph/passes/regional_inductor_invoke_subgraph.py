"""Compile marked nested regions into native custom-op call sites."""

from __future__ import annotations

import itertools
import threading
from typing import Any, Callable

from ..graph_module import GraphModule
from ..node import Node

__all__ = ["regional_inductor_invoke_subgraph"]


_op_ids = itertools.count()
_op_ids_lock = threading.Lock()


def _config(node: Node) -> Any:
    custom = node.meta.get("custom")
    return custom.get("nested_region_config") if isinstance(custom, dict) else None


def _metadata(node: Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _compiler_for(
    node: Node,
    default: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    config = _config(node)
    if config is None:
        return default
    if node.meta.get("partitioner_tag") == "is_backward":
        compiler = getattr(config, "bw_compiler", None)
    else:
        compiler = getattr(config, "fw_compiler", None)
    return compiler if callable(compiler) else default


def _template_flatten(value: Any, template: Any) -> list[Any]:
    kind = template[0]
    if kind == "tensor":
        try:
            import tensorplay

            is_tensor = isinstance(value, tensorplay.Tensor)
        except ImportError:
            is_tensor = False
        if not is_tensor:
            raise RuntimeError("nested region returned a non-tensor value")
        return [value]
    if kind == "tuple":
        if not isinstance(value, tuple) or len(value) != len(template[1]):
            raise RuntimeError("nested region returned an invalid tuple")
        result: list[Any] = []
        for item, item_template in zip(value, template[1]):
            result.extend(_template_flatten(item, item_template))
        return result
    if kind == "list":
        if not isinstance(value, list) or len(value) != len(template[1]):
            raise RuntimeError("nested region returned an invalid list")
        result = []
        for item, item_template in zip(value, template[1]):
            result.extend(_template_flatten(item, item_template))
        return result
    if kind == "dict":
        if not isinstance(value, dict):
            raise RuntimeError("nested region returned an invalid mapping")
        result = []
        for key, item_template in template[1]:
            if key not in value:
                raise RuntimeError(f"nested region omitted output key {key!r}")
            result.extend(_template_flatten(value[key], item_template))
        return result
    raise RuntimeError(f"unknown nested output template {kind!r}")


def _encode(value: Any, inputs: list[Node]) -> Any:
    if isinstance(value, Node):
        index = len(inputs)
        inputs.append(value)
        return ("input", index)
    if isinstance(value, tuple):
        return ("tuple", tuple(_encode(item, inputs) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_encode(item, inputs) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _encode(item, inputs)) for key, item in value.items()),
        )
    if isinstance(value, slice):
        return (
            "slice",
            _encode(value.start, inputs),
            _encode(value.stop, inputs),
            _encode(value.step, inputs),
        )
    return ("constant", value)


def _decode(template: Any, inputs: tuple[Any, ...]) -> Any:
    kind = template[0]
    if kind == "input":
        return inputs[template[1]]
    if kind == "constant":
        return template[1]
    if kind == "tuple":
        return tuple(_decode(item, inputs) for item in template[1])
    if kind == "list":
        return [_decode(item, inputs) for item in template[1]]
    if kind == "dict":
        return {
            key: _decode(item, inputs) for key, item in template[1]
        }
    if kind == "slice":
        return slice(
            _decode(template[1], inputs),
            _decode(template[2], inputs),
            _decode(template[3], inputs),
        )
    raise RuntimeError(f"unknown nested input template {kind!r}")


def _new_op_name() -> str:
    with _op_ids_lock:
        return f"tensorplay::nested_region_{next(_op_ids)}"


def _native_region_op(
    compiled: Callable[..., Any],
    args_template: Any,
    kwargs_template: Any,
    output_template: Any,
    output_count: int,
) -> Any:
    from tensorplay.library import custom_op

    def invoke(*inputs: Any) -> Any:
        args = _decode(args_template, inputs)
        kwargs = _decode(kwargs_template, inputs)
        result = compiled(*args, **kwargs)
        outputs = _template_flatten(result, output_template)
        if len(outputs) != output_count:
            raise RuntimeError(
                f"nested region returned {len(outputs)} outputs; "
                f"expected {output_count}"
            )
        return outputs[0] if output_count == 1 else tuple(outputs)

    return custom_op(_new_op_name(), invoke, mutates_args=())


def _sample_inputs(subgraph: GraphModule) -> list[Any]:
    samples = subgraph.meta.get("sample_inputs")
    if not isinstance(samples, dict):
        raise RuntimeError("nested region has no bound sample inputs")
    result = []
    for node in subgraph.graph.placeholders:
        if node.name not in samples:
            raise RuntimeError(
                f"nested region sample is missing for input {node.name!r}"
            )
        result.append(samples[node.name])
    return result


def regional_inductor_invoke_subgraph(
    gm: GraphModule,
    *example_args: object,
    compiler: Callable[..., Any] | None = None,
    compiler_kwargs: dict[str, Any] | None = None,
) -> GraphModule:
    """Replace nested graph calls with compiled native custom operators."""

    del example_args
    compile_kwargs = dict(compiler_kwargs or {})
    compiled_regions: dict[tuple[int, int], Callable[..., Any]] = {}
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        custom = _metadata(node)
        if not (
            custom.get("nested_region_config") or custom.get("opaque_region")
        ) or custom.get("nested_region_compiled"):
            continue
        if custom.get("opaque_region"):
            compiled = custom.get("opaque_callable")
            if not callable(compiled):
                raise TypeError("opaque region has no callable runtime")
            call_args = node.args
            call_kwargs = node.kwargs
        else:
            selected_compiler = _compiler_for(node, compiler)
            if selected_compiler is None:
                raise RuntimeError("nested region has no compiler")
            if not node.args or not isinstance(node.args[0], Node):
                raise ValueError("nested region call must name its graph attribute")
            graph_attr = node.args[0].target
            subgraph = gm._get_attr(graph_attr)
            if not isinstance(subgraph, GraphModule):
                raise TypeError("nested region target is not a GraphModule")

            regional_inductor_invoke_subgraph(
                subgraph,
                compiler=selected_compiler,
                compiler_kwargs=compile_kwargs if selected_compiler is compiler else {},
            )
            cache_key = (id(subgraph), id(selected_compiler))
            if cache_key in compiled_regions:
                compiled = compiled_regions[cache_key]
            else:
                inputs = _sample_inputs(subgraph)
                if selected_compiler is compiler:
                    compiled = selected_compiler(subgraph, inputs, **compile_kwargs)
                else:
                    compiled = selected_compiler(subgraph, inputs)
                if not callable(compiled):
                    raise TypeError("nested compiler must return a callable artifact")
                compiled_regions[cache_key] = compiled
            call_args = node.args[1:]
            call_kwargs = node.kwargs

        output_template = custom.get("nested_output_template")
        output_count = custom.get("nested_output_count")
        if output_template is None or not isinstance(output_count, int):
            raise RuntimeError("nested region output metadata is incomplete")

        op_inputs: list[Node] = []
        args_template = _encode(call_args, op_inputs)
        kwargs_template = _encode(call_kwargs, op_inputs)
        node.target = _native_region_op(
            compiled,
            args_template,
            kwargs_template,
            output_template,
            output_count,
        )
        node.args = tuple(op_inputs)
        node.kwargs = {}
        custom["nested_region_compiled"] = True
        custom["nested_custom_op"] = node.target.name
        node.meta["custom"] = custom
    gm.graph.lint()
    gm.recompile()
    return gm
