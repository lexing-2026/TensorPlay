"""Lowering of a captured graph into the native graph.

Walks the traced operators once, folds what can be folded, routes
elementwise runs through the planners, and records the values the
reverse pass will ask for.
"""
from __future__ import annotations

import numbers
import operator
from typing import Any

from ....graph import GraphModule, Node
from ....library import CustomOpDef as _CustomOpDef
from .ir import (
    _NativeLowering,
    _is_scalar,
    _int_list,
    _nodes,
    _set_int_list_attr,
    _set_named_scalar_attr,
    _set_scalar_attr,
    _spatial_int_list,
    _target_name,
    _traced_value,
)
from .pointwise import (
    _ForwardPointwiseFuser,
    _native_fused_pointwise_plans,
)


def _native_value_leaves(value: Any, values: dict[Node, Any]) -> list[Any]:
    if isinstance(value, Node):
        native = values.get(value)
        if isinstance(native, tuple):
            return list(native)
        return [] if native is None else [native]
    if isinstance(value, tuple | list):
        result: list[Any] = []
        for item in value:
            result.extend(_native_value_leaves(item, values))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_native_value_leaves(item, values))
        return result
    if isinstance(value, slice):
        result = []
        for item in (value.start, value.stop, value.step):
            result.extend(_native_value_leaves(item, values))
        return result
    return []

def _native_output_spec(value: Any, values: dict[Node, Any]) -> Any:
    if isinstance(value, Node):
        native = values.get(value)
        if isinstance(native, tuple):
            custom = value.meta.get("custom")
            template = custom.get("nested_output_template") if isinstance(custom, dict) else None
            if template is not None:
                return ("nested", template)
            # Native view operators expose a flat tuple of Tensor values.
            # Their graph output is already the public tuple, so no custom
            # pytree template is needed to rebuild it.
            return ("tuple", tuple(("leaf",) for _ in native))
        if native is None:
            raise RuntimeError(f"native graph has no value for output {value.name!r}")
        return ("leaf",)
    if isinstance(value, tuple):
        return ("tuple", tuple(_native_output_spec(item, values) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_native_output_spec(item, values) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _native_output_spec(item, values)) for key, item in value.items()),
        )
    raise RuntimeError("native graph outputs must be tensor values")


_NATIVE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "matmul",
    "t",
    "linear",
    "neg",
    "pos",
    "abs",
    "sin",
    "cos",
    "exp",
    "log",
    "sigmoid",
    "sqrt",
    "rsqrt",
    "square",
    "tanh",
    "sign",
    "relu",
    "silu",
    "conj",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cosh",
    "erf",
    "erfc",
    "exp2",
    "expm1",
    "floor",
    "log1p",
    "log2",
    "reciprocal",
    "round",
    "sinh",
    "tan",
    "trunc",
    "erfinv",
    "erfcx",
    "lgamma",
    "i0",
    "tanhshrink",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    "clamp",
    "clamp_min",
    "clamp_max",
    "gelu",
    "softmax",
    "log_softmax",
    "layer_norm",
    "transpose",
    "select",
    "slice",
    "stack",
    "repeat",
    "index_select",
    "gather",
    "embedding",
    "constant_pad_nd",
    "pad",
    "mm",
    # Tensor kernels used by the ResNet inference graph.  These are kept in
    # the native graph instead of falling back to the generated Python
    # executor; the latter still calls every functional wrapper through the
    # interpreter and is not a compiled path in any meaningful sense.
    "conv2d",
    "conv2d_relu",
    "add_relu",
    "batch_norm",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "avg_pool1d",
    "avg_pool3d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "flatten",
    "view",
    "reshape",
    "permute",
    "permute_backward",
    "contiguous",
    "unsqueeze",
    "squeeze",
    "zeros_like",
    "float",
    "group_norm",
    "avg_pool2d",
    "interpolate",
    "scaled_dot_product_attention",
    "dropout",
    "sum",
    "mean",
    "cat",
    "expand",
    "chunk",
    "split",
    "split_with_sizes",
    "unbind",
}

def _fold_eval_conv_batch_norm(
    graph_module: GraphModule,
    example_inputs: list[Any],
) -> dict[Node, tuple[Node, Any, Any]]:
    """Precompute inference BatchNorm parameters for Conv2d users.

    ResNet inference contains the stable pattern ``conv2d -> batch_norm``.
    Folding the running-statistics transform into the convolution removes one
    full feature-map kernel and its intermediate write.  The optimization is
    intentionally restricted to eval-mode BatchNorm with a single Conv2d
    user, so training graphs and branched tensors retain the ordinary native
    operators.

    The returned tensors are compile-time constants owned by the native
    lowering.  TensorPlay's public compile contract currently has no
    parameter-version guard, therefore this pass is only enabled for the
    inference lowering path; callers that mutate parameters must recompile.
    """

    # Do not fold a graph that is being differentiated with respect to its
    # runtime inputs.  The folded parameters are inference constants, while
    # eval-mode autograd still needs the original parameter edges.
    if any(getattr(value, "requires_grad", False) for value in example_inputs):
        return {}

    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return {}

    # Folding changes parameter dataflow, so it is valid only for the
    # no-grad inference specialization that the benchmark requests.  A
    # grad-enabled compile must retain the ordinary Conv/BN autograd edges.
    if tensorplay.is_grad_enabled():
        return {}

    def tensor_attr(value: Any) -> Any | None:
        if not isinstance(value, Node) or value.op != "get_attr":
            return None
        attribute = graph_module._get_attr(value.target)
        return attribute if isinstance(attribute, tensor_type) else None

    folded: dict[Node, tuple[Node, Any, Any]] = {}
    for batch_norm in graph_module.graph.nodes:
        if (
            batch_norm.op != "call_function"
            or _target_name(batch_norm.target) != "batch_norm"
            or len(batch_norm.args) != 8
            or batch_norm.kwargs
            or batch_norm.args[5] is not False
        ):
            continue

        conv = batch_norm.args[0]
        if (
            not isinstance(conv, Node)
            or conv.op != "call_function"
            or _target_name(conv.target) != "conv2d"
            or len(conv.args) != 7
            or conv.kwargs
            or conv.users != {batch_norm}
        ):
            continue

        running_mean = tensor_attr(batch_norm.args[1])
        running_var = tensor_attr(batch_norm.args[2])
        bn_weight = tensor_attr(batch_norm.args[3])
        bn_bias = tensor_attr(batch_norm.args[4])
        conv_weight = tensor_attr(conv.args[1])
        conv_bias = tensor_attr(conv.args[2])
        eps = batch_norm.args[7]
        if (
            running_mean is None
            or running_var is None
            or conv_weight is None
            or not isinstance(eps, numbers.Real)
            or conv.args[2] is not None and conv_bias is None
            or batch_norm.args[3] is not None and bn_weight is None
            or batch_norm.args[4] is not None and bn_bias is None
        ):
            continue

        try:
            with tensorplay.no_grad():
                running_mean = running_mean.detach()
                running_var = running_var.detach()
                conv_weight = conv_weight.detach()
                weight_coeff = tensorplay.rsqrt(running_var + float(eps))
                scale = (
                    bn_weight.detach()
                    if bn_weight is not None
                    else tensorplay.ones_like(running_var)
                ) * weight_coeff
                folded_weight = conv_weight * scale.reshape((-1, 1, 1, 1))
                base_bias = (
                    conv_bias.detach()
                    if conv_bias is not None
                    else tensorplay.zeros_like(running_mean)
                )
                folded_bias = (
                    (base_bias - running_mean) * scale
                    + (
                        bn_bias.detach()
                        if bn_bias is not None
                        else tensorplay.zeros_like(running_mean)
                    )
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Keep the ordinary native Conv+BN path if a backend dtype or
            # device cannot materialize the folded constants.
            continue

        folded[conv] = (batch_norm, folded_weight, folded_bias)
    return folded

_NATIVE_OP_SUPPORT: dict[str, bool] = {}

def _native_runs_linear() -> bool:
    """Whether the loaded runtime can execute a fused ``linear`` node.

    The lowering and the runtime library are built separately, and a tree can
    hold one newer than the other, so a node this module knows how to emit is
    not necessarily one the runtime knows how to run.  Probed once per
    process with a one-node graph; a runtime that cannot run it keeps the
    transpose-product-add form, which every runtime can.
    """

    known = _NATIVE_OP_SUPPORT.get("linear")
    if known is not None:
        return known
    supported = False
    try:
        import tensorplay

        native = tensorplay._C._stax
        graph = native.Graph()
        node = graph.create_node("linear", "probe")
        node.add_input(graph.add_input())
        node.add_input(graph.add_input())
        graph.register_output(node.add_output())
        graph.execute([tensorplay.zeros((1, 1)), tensorplay.zeros((1, 1))])
        supported = True
    except Exception:  # noqa: BLE001 - any failure means "emit the long form"
        supported = False
    _NATIVE_OP_SUPPORT["linear"] = supported
    return supported

def _autocast_narrowed_operands(node: Node, autocast_dtype: Any) -> tuple[Node, ...]:
    """Operands a matrix-product consumer reads converted to the autocast type.

    Under autocast the lowering feeds convolutions, linear layers and
    attention with operands converted to the reduced element type; these are
    the operands it converts for ``node``.  The rule matches the conversions
    the lowering emits below, operand for operand.
    """

    if autocast_dtype is None or node.op not in {"call_function", "call_method"}:
        return ()
    op_name = _target_name(node.target)
    args = node.args
    if op_name in {"conv2d", "conv2d_relu"} and len(args) == 7:
        return tuple(arg for arg in args[:2] if isinstance(arg, Node))
    produced = getattr(node.meta.get("val"), "dtype", None)
    if produced != autocast_dtype:
        return ()
    if op_name == "linear" and len(args) in {2, 3} and _native_runs_linear():
        return tuple(arg for arg in args[:2] if isinstance(arg, Node))
    if op_name == "scaled_dot_product_attention" and len(args) >= 3:
        return tuple(arg for arg in args[:3] if isinstance(arg, Node))
    return ()


def _autocast_only_values(
    graph_module: GraphModule,
    required_nodes: set[Node],
    autocast_dtype: Any,
) -> dict[Node, Any]:
    """Values every reader consumes converted to the autocast element type.

    Such a value is never needed at its own width: not by a reader, not as a
    graph output and not as a saved value.  Its producer may store it
    narrowed in the first place.
    """

    import tensorplay

    if autocast_dtype is None:
        return {}
    result: dict[Node, Any] = {}
    for node in graph_module.graph.nodes:
        if node.op not in {"call_function", "call_method"} or node in required_nodes:
            continue
        if not node.users:
            continue
        sample = _traced_value(graph_module, node)
        if getattr(sample, "dtype", None) != tensorplay.float32:
            continue
        if all(
            node in _autocast_narrowed_operands(user, autocast_dtype)
            and node not in user.kwargs.values()
            and sum(arg is node for arg in user.args)
            == sum(arg is node for arg in _autocast_narrowed_operands(user, autocast_dtype))
            for user in node.users
        ):
            result[node] = autocast_dtype
    return result


def _lower_native(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_fusion: bool = True,
    extra_output_nodes: list[Node] | None = None,
    save_autocast_inputs: bool = False,
) -> _NativeLowering | None:
    """Lower the canonical graph into the native Stax IR when possible."""

    try:
        import tensorplay

        native_module = getattr(tensorplay._C, "_stax", None)
    except (AttributeError, ImportError):
        native_module = None
    if native_module is None:
        return None
    if not hasattr(native_module.Graph, "execute"):
        return None

    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if len(example_inputs) != len(graph_module.graph.placeholders):
        return None
    if any(not isinstance(value, tensor_type) for value in example_inputs):
        # Stax's native ABI is Tensor-only.  The generated GraphModule remains
        # the correct compiled path for scalar/keyword placeholders.
        return None

    graph = native_module.Graph()
    values: dict[Node, Any] = {}
    literal_values: dict[int, Any] = {}
    attribute_targets: list[str] = []
    constant_values: list[Any] = []
    required_nodes = {
        value
        for output in graph_module.graph.outputs
        for value in _nodes(output.args)
    }
    required_nodes.update(extra_output_nodes or [])
    autocast_dtype = (
        tensorplay.get_autocast_dtype("cuda")
        if save_autocast_inputs and tensorplay.is_autocast_enabled("cuda")
        else None
    )
    fused_pointwise_plans = (
        _native_fused_pointwise_plans(
            graph_module,
            required_nodes,
            _autocast_only_values(graph_module, required_nodes, autocast_dtype),
        )
        if use_fusion and extra_output_nodes is not None
        else {}
    )
    # Plan outputs stored at the autocast width: the conversion a matrix
    # product would otherwise apply to them is already done.
    prenarrowed: dict[Node, Any] = {}
    emitted_fused_pointwise_nodes: set[Node] = set()
    folded_convs = _fold_eval_conv_batch_norm(graph_module, example_inputs)
    # convolution inputs, weights, and outputs.  Its generated wrapper uses
    # ``empty_strided`` tensors with the channels-last strides and the current
    # rather than assuming NCHW.  Materialize the same weight layout once for
    # folded constants; runtime activations are converted by a native graph
    # node immediately before the first convolution that consumes them.
    use_channels_last = bool(
        example_inputs
        and example_inputs[0].device.is_cuda()
        and not tensorplay.is_grad_enabled()
    )
    if use_channels_last and folded_convs:
        try:
            folded_with_layout: dict[Node, tuple[Node, Any, Any]] = {}
            for conv, (batch_norm, weight, bias) in folded_convs.items():
                # NCHW logical shape, NHWC physical storage, then reinterpret
                # as NCHW. This layout is represented by generated weight
                # strides, e.g. [K*C*R*S, 1, S*C, C].
                physical_weight = weight.permute((0, 2, 3, 1)).clone()
                channels_last_weight = physical_weight.permute((0, 3, 1, 2))
                folded_with_layout[conv] = (
                    batch_norm,
                    channels_last_weight,
                    bias,
                )
            folded_convs = folded_with_layout
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Keep the already validated folding path if this backend cannot
            # materialize the specialized layout for a particular dtype.
            use_channels_last = False
    folded_native_inputs: dict[Node, tuple[Any, Any]] = {}
    folded_batch_norm_to_conv = {
        batch_norm: conv for conv, (batch_norm, _, _) in folded_convs.items()
    }

    # belong in the native ABI.  Inference Conv+BN folding leaves the original
    # parameter nodes in the captured graph, but the native graph consumes only the folded
    # weight/bias constants.  Passing those dead tensors through Python and
    # C++ on every invocation is pure call-boundary overhead.
    live_attribute_nodes: set[Node] = set()
    visited_nodes: set[Node] = set()

    def visit_native_dependency(value: Any) -> None:
        if isinstance(value, Node):
            if value in visited_nodes:
                return
            visited_nodes.add(value)
            if value.op == "get_attr":
                live_attribute_nodes.add(value)
                return
            if value in folded_convs or value in folded_batch_norm_to_conv:
                visit_native_dependency(value.args[0])
                return
            for argument in value.args:
                visit_native_dependency(argument)
            for argument in value.kwargs.values():
                visit_native_dependency(argument)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit_native_dependency(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit_native_dependency(item)
        elif isinstance(value, slice):
            visit_native_dependency(value.start)
            visit_native_dependency(value.stop)
            visit_native_dependency(value.step)

    for output in graph_module.graph.outputs:
        visit_native_dependency(output.args)
    for extra_node in extra_output_nodes or []:
        visit_native_dependency(extra_node)
    # Buffer-update nodes are epilogue state: even when no consumer reads
    # their value, the update must run, so their module-state operands stay
    # live inputs of the native graph.
    for node in graph_module.graph.nodes:
        if node.op == "call_method" and node.target in {"add_", "sub_", "mul_"}:
            visit_native_dependency(node)

    fused_relu_convs: dict[Node, Node] = {}
    fused_add_relus: dict[Node, Node] = {}
    fused_relu_nodes: set[Node] = set()
    layout_values: dict[Node, bool] = {}
    channels_last_values: dict[Node, Any] = {}
    autocast_values: dict[tuple[Node, Any], Any] = {}
    mutations: list[tuple[int, Any]] = []
    # A training capture keeps the routes that report values beside their
    # result -- an attention normalizer, a normalization's row statistics --
    # so the reverse pass reads them instead of reducing the input again.
    # Only the training lowering asks for saved values at all, which makes
    # that request the switch.
    keep_saved_results = extra_output_nodes is not None
    saved_result_values: list[tuple[Node, tuple[Any, ...]]] = []
    # Keep bias in the convolution during training so the backend can place
    # it in the same execution plan as the convolution.  The inference path
    # retains the specialized post-convolution handling.
    peel_conv_bias = bool(
        example_inputs
        and example_inputs[0].device.is_cuda()
        and not tensorplay.is_grad_enabled()
    )
    # This is the same producer/sole-consumer legality check used by
    # training graph because add_relu has a generated autograd formula; the
    # Conv->ReLU and Conv+BN folding paths remain inference-only.
    if use_fusion:
        for relu in graph_module.graph.nodes:
            if (
                relu.op != "call_function"
                or _target_name(relu.target) != "relu"
                or len(relu.args) != 1
                or relu.kwargs not in ({}, {"inplace": False}, {"inplace": True})
            ):
                continue
            source = relu.args[0]
            if (
                isinstance(source, Node)
                and source.op == "call_function"
                and _target_name(source.target) == "add"
                and source.users == {relu}
            ):
                if len(source.args) == 2:
                    lhs, rhs = source.args
                    alpha = 1
                elif len(source.args) == 3:
                    lhs, rhs, alpha = source.args
                else:
                    continue
                if (
                    alpha == 1
                    and isinstance(lhs, Node)
                    and isinstance(rhs, Node)
                ):
                    fused_add_relus[source] = relu
                continue
            if tensorplay.is_grad_enabled():
                continue
            conv = folded_batch_norm_to_conv.get(source)
            if conv is None:
                conv = source
                if not (
                    isinstance(conv, Node)
                    and conv.op == "call_function"
                    and _target_name(conv.target) == "conv2d"
                ):
                    continue
            if source.users != {relu} or conv.users != ({source} if source is not conv else {relu}):
                continue
            fused_relu_convs[conv] = relu
    input_positions: dict[Node, int] = {}
    next_input_index = 0
    for node in graph_module.graph.placeholders:
        values[node] = graph.add_input()
        input_positions[node] = next_input_index
        next_input_index += 1
        layout_values[node] = False

    def channels_last_value(node: Any) -> Any | None:
        """Return the native value in the preferred 4-D layout."""

        if not isinstance(node, Node) or node not in values:
            return None
        if not use_channels_last:
            return values[node]
        if layout_values.get(node, False):
            return values[node]
        cached = channels_last_values.get(node)
        if cached is not None:
            return cached
        reorder = graph.create_node("channels_last", f"{node.name}_channels_last")
        reorder.add_input(values[node])
        converted = reorder.add_output()
        channels_last_values[node] = converted
        return converted

    def saved_autocast_value(node: Node, value: Any) -> Any:
        if autocast_dtype is None:
            return value
        sample = _traced_value(graph_module, node)
        if getattr(sample, "dtype", None) != tensorplay.float32:
            return value
        key = (node, autocast_dtype)
        cached = autocast_values.get(key)
        if cached is not None:
            return cached
        stored = prenarrowed.get(node)
        if stored is not None:
            autocast_values[key] = stored
            return stored
        cast = graph.create_node("cast", f"{node.name}_autocast")
        cast.add_input(value)
        cast.set_str_attr("dtype", str(autocast_dtype).rsplit(".", 1)[-1])
        converted = cast.add_output()
        autocast_values[key] = converted
        return converted

    # Register all live module attributes before synthetic folded weights so
    # Graph::execute's input order is independent of where get_attr nodes are
    # placed in the Python graph.
    for node in graph_module.graph.nodes:
        if node.op != "get_attr" or node not in live_attribute_nodes:
            continue
        attribute = graph_module._get_attr(node.target)
        if not isinstance(attribute, tensor_type):
            return None
        values[node] = graph.add_input()
        input_positions[node] = next_input_index
        next_input_index += 1
        attribute_targets.append(node.target)

    for node in graph_module.graph.nodes:
        folded = folded_convs.get(node)
        if folded is None:
            continue
        _, folded_weight, folded_bias = folded
        native_weight = graph.add_input()
        native_bias = graph.add_input()
        next_input_index += 2
        folded_native_inputs[node] = (native_weight, native_bias)
        constant_values.extend((folded_weight, folded_bias))

    def fuser_sample(node: Node) -> Any:
        sample = _traced_value(graph_module, node)
        if sample is not None:
            return sample
        if node.op == "placeholder":
            try:
                return example_inputs[graph_module.graph.placeholders.index(node)]
            except (ValueError, IndexError):
                return None
        if node.op == "get_attr":
            try:
                return graph_module._get_attr(node.target)
            except (AttributeError, KeyError, RuntimeError):
                return None
        return None

    fuser = _ForwardPointwiseFuser(
        graph,
        values,
        fuser_sample,
        set(fused_pointwise_plans)
        | fused_relu_nodes
        | set(fused_add_relus)
        | set(fused_add_relus.values()),
    )

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op not in {"call_function", "call_method"}:
            return None
        if use_fusion:
            try:
                if not fuser.ensure_for(node):
                    return None
                if node not in emitted_fused_pointwise_nodes and fuser.absorb(node):
                    continue
                if not fuser.flush_for(node):
                    return None
            except (KeyError, RuntimeError, TypeError, ValueError):
                return None
            if node in emitted_fused_pointwise_nodes:
                continue
        pointwise_plan = fused_pointwise_plans.get(node) if use_fusion else None
        if pointwise_plan is not None:
            (
                plan_nodes,
                externals,
                program,
                constants,
                output_refs,
                exports,
                op_name,
                narrowed_exports,
            ) = pointwise_plan
            if any(values.get(external) is None for external in externals):
                return None
            native_node = graph.create_node(
                "custom_op" if op_name is not None else "fused_pointwise",
                f"{node.name}_fused",
            )
            if op_name is not None:
                native_node.set_str_attr("op_name", op_name)
            else:
                native_node.set_int_attr("input_count", len(externals))
                native_node.set_ints_attr("program", program)
                native_node.set_floats_attr("constants", constants)
            for external in externals:
                native_node.add_input(values[external])
            native_outputs = {
                exported: native_node.add_output() for exported in exports
            }
            if op_name is None and len(exports) > 1:
                native_node.set_ints_attr("output_refs", output_refs)
            for exported, native_output in native_outputs.items():
                values[exported] = native_output
                layout_values[exported] = False
                if exported in narrowed_exports:
                    prenarrowed[exported] = native_output
            emitted_fused_pointwise_nodes.update(plan_nodes)
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) == 2
            and isinstance(node.args[0], Node)
            and isinstance(node.args[1], tuple)
            and len(node.args[1]) == 2
            and isinstance(node.args[1][0], slice)
            and node.args[1][0] == slice(None)
            and node.args[1][1] is None
            and node.args[0] in values
        ):
            native_node = graph.create_node("unsqueeze", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("dim", 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) == 2
            and isinstance(node.args[0], Node)
            and isinstance(node.args[1], int)
            and isinstance(values.get(node.args[0]), tuple)
        ):
            source_values = values[node.args[0]]
            index = node.args[1]
            if index < 0:
                index += len(source_values)
            if index < 0 or index >= len(source_values):
                return None
            values[node] = source_values[index]
            layout_values[node] = False
            continue
        op_name = _target_name(node.target)
        # User-defined operators enter the native dispatcher bridge.  Their
        # output arity is carried by the graph metadata because the native
        # value table reserves one value per result.
        if isinstance(node.target, _CustomOpDef):
            if node.kwargs or not node.args:
                return None
            native_node = graph.create_node("custom_op", node.name)
            native_node.set_str_attr("op_name", node.target.name)
            for argument in node.args:
                resolved = values.get(argument) if isinstance(argument, Node) else None
                if resolved is None or isinstance(resolved, tuple):
                    return None
                native_node.add_input(resolved)
            custom = node.meta.get("custom")
            output_count = 1
            if isinstance(custom, dict):
                output_count = int(custom.get("nested_output_count", 1))
            if output_count < 1:
                return None
            output_values = tuple(native_node.add_output() for _ in range(output_count))
            values[node] = output_values[0] if output_count == 1 else output_values
            continue
        # ``ReLU(inplace=True)`` is an aliasing detail of the captured graph.
        # The native kernel accepts that schema explicitly; other keyword
        # combinations are rejected by this lowering.
        if node.kwargs:
            if op_name == "relu":
                if node.kwargs not in ({"inplace": False}, {"inplace": True}):
                    return None
            elif op_name == "silu":
                # The native graph has no mutation output for SiLU. Preserve
                # the in-place contract by using the general executor there.
                if node.kwargs not in ({"inplace": False},):
                    return None
            elif op_name not in {
                "sum",
                "mean",
                "add",
                "sub",
                "silu",
                "gelu",
                "softmax",
                "log_softmax",
                "layer_norm",
                "where",
                "clamp",
                "clamp_min",
                "clamp_max",
                "unsqueeze",
                "squeeze",
                "zeros_like",
                "permute_backward",
                "expand",
                "cat",
                "chunk",
                "split",
                "split_with_sizes",
                "unbind",
                "slice",
                "stack",
                "repeat",
                "embedding",
                "scaled_dot_product_attention",
            }:
                return None
        if op_name in {"add_", "sub_", "mul_"}:
            # In-place updates of module state (e.g. a batch counter) have no
            # aliasing target inside the functional native graph.  Lower the
            # arithmetic functionally; the executable wrapper copies the
            # result back into the source input after every execution.
            if len(node.args) != 2:
                return None
            source_node, delta = node.args
            if not isinstance(source_node, Node) or source_node not in input_positions:
                return None
            if not _is_scalar(delta) and not (
                isinstance(delta, Node) and delta in values
            ):
                return None
            native_node = graph.create_node(op_name[:-1], node.name)
            native_node.add_input(values[source_node])
            if _is_scalar(delta):
                _set_scalar_attr(native_node, delta, 1)
            else:
                native_node.add_input(values[delta])
            mutation_value = native_node.add_output()
            values[node] = mutation_value
            layout_values[node] = False
            mutations.append((input_positions[source_node], mutation_value))
            continue
        if op_name not in _NATIVE_OPS:
            return None

        if op_name == "relu" and node in fused_relu_nodes:
            source = node.args[0]
            if source not in values:
                return None
            # The producer was lowered with a fused Conv+ReLU primitive; the
            # ReLU value is an alias of that already-activated output.
            values[node] = values[source]
            layout_values[node] = layout_values.get(source, False)
            continue

        if op_name == "relu" and node in fused_add_relus.values():
            source = node.args[0]
            if not isinstance(source, Node) or source not in values:
                return None
            # The residual add is lowered as add_relu below, so the ReLU
            # node observes the already-activated output without another
            # native launch.
            values[node] = values[source]
            layout_values[node] = layout_values.get(source, False)
            continue

        def node_value(value: Any) -> Any | None:
            if isinstance(value, Node):
                return values.get(value)
            if isinstance(value, tensor_type):
                key = id(value)
                resolved = literal_values.get(key)
                if resolved is None:
                    resolved = graph.add_input()
                    literal_values[key] = resolved
                    constant_values.append(value)
                return resolved
            return None

        def sample_value(value: Any) -> Any | None:
            if not isinstance(value, Node):
                return None
            sample = _traced_value(graph_module, value)
            if sample is not None:
                return sample
            if value.op == "placeholder":
                try:
                    return example_inputs[graph_module.graph.placeholders.index(value)]
                except (ValueError, IndexError):
                    return None
            if value.op == "get_attr":
                try:
                    return graph_module._get_attr(value.target)
                except (AttributeError, KeyError, RuntimeError):
                    return None
            return None

        def add_tensor_input(native_node: Any, value: Any) -> bool:
            resolved = node_value(value)
            if resolved is None or isinstance(resolved, tuple):
                return False
            native_node.add_input(resolved)
            return True

        if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if len(node.args) != 2 or node.kwargs:
                return None
            lhs, rhs = node.args
            lhs_is_tensor = isinstance(lhs, (Node, tensor_type))
            rhs_is_tensor = isinstance(rhs, (Node, tensor_type))
            if lhs_is_tensor == rhs_is_tensor:
                if not lhs_is_tensor:
                    return None
            elif not lhs_is_tensor and not _is_scalar(lhs):
                return None
            elif not rhs_is_tensor and not _is_scalar(rhs):
                return None
            native_node = graph.create_node(op_name, node.name)
            if lhs_is_tensor and not add_tensor_input(native_node, lhs):
                return None
            if rhs_is_tensor and not add_tensor_input(native_node, rhs):
                return None
            if not lhs_is_tensor:
                _set_scalar_attr(native_node, lhs, 0)
            elif not rhs_is_tensor:
                _set_scalar_attr(native_node, rhs, 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "where":
            if len(node.args) != 3 or node.kwargs:
                return None
            condition, self_value, other_value = node.args
            if not isinstance(condition, Node) or condition not in values:
                return None
            self_is_tensor = isinstance(self_value, (Node, tensor_type))
            other_is_tensor = isinstance(other_value, (Node, tensor_type))
            if not self_is_tensor and not _is_scalar(self_value):
                return None
            if not other_is_tensor and not _is_scalar(other_value):
                return None
            native_node = graph.create_node("where", node.name)
            native_node.add_input(values[condition])
            if self_is_tensor:
                if not add_tensor_input(native_node, self_value):
                    return None
            elif not _set_named_scalar_attr(native_node, "self_scalar", self_value):
                return None
            if other_is_tensor:
                if not add_tensor_input(native_node, other_value):
                    return None
            elif not _set_named_scalar_attr(native_node, "other_scalar", other_value):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"clamp", "clamp_min", "clamp_max"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs and set(node.kwargs) - {"min", "max"}:
                return None
            source = node.args[0]
            if source not in values:
                return None
            args = list(node.args[1:])
            kwargs = dict(node.kwargs or {})
            if op_name == "clamp":
                if len(args) > 2 or (args and "min" in kwargs) or (
                    len(args) > 1 and "max" in kwargs
                ):
                    return None
                lower = args[0] if args else kwargs.get("min")
                upper = args[1] if len(args) > 1 else kwargs.get("max")
                if lower is None and upper is None:
                    return None
            else:
                if len(args) != 1 or kwargs:
                    return None
                lower = args[0] if op_name == "clamp_min" else None
                upper = args[0] if op_name == "clamp_max" else None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            for key, bound in (("min", lower), ("max", upper)):
                tensor_bound = isinstance(bound, (Node, tensor_type))
                native_node.set_int_attr(
                    "min_tensor" if key == "min" else "max_tensor",
                    int(tensor_bound),
                )
                if bound is None:
                    continue
                if tensor_bound:
                    if not add_tensor_input(native_node, bound):
                        return None
                elif not _set_named_scalar_attr(
                    native_node, "min_scalar" if key == "min" else "max_scalar", bound
                ):
                    return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "gelu":
            if len(node.args) != 1 or not isinstance(node.args[0], Node):
                return None
            approximate = node.kwargs.get("approximate", "none")
            if not isinstance(approximate, str) or approximate not in {"none", "tanh"}:
                return None
            native_node = graph.create_node("gelu", node.name)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            native_node.set_str_attr("approximate", approximate)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"softmax", "log_softmax"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            source = node.args[0]
            if node.op == "call_method":
                if len(node.args) not in {2, 3} or kwargs:
                    return None
                dim = node.args[1]
                dtype = node.args[2] if len(node.args) == 3 else tensorplay.undefined
            else:
                if len(node.args) not in {2, 3}:
                    return None
                dim = node.args[1]
                dtype = node.args[2] if len(node.args) == 3 else tensorplay.undefined
            if source not in values or isinstance(dim, bool) or not isinstance(dim, int):
                return None
            if dtype is None:
                dtype = tensorplay.undefined
            dtype_name = getattr(dtype, "name", None)
            if dtype_name != "undefined":
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim", int(dim))
            native_node.set_int_attr("dtype", 22)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "layer_norm":
            if len(node.args) != 5 or node.kwargs:
                return None
            source, normalized_shape, weight, bias, eps = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            shape = _int_list(normalized_shape)
            if shape is None or not isinstance(eps, numbers.Real):
                return None
            native_node = graph.create_node("layer_norm", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("normalized_shape", shape)
            for attr_name, optional_node in (("has_weight", weight), ("has_bias", bias)):
                if optional_node is None:
                    native_node.set_int_attr(attr_name, 0)
                elif add_tensor_input(native_node, optional_node):
                    native_node.set_int_attr(attr_name, 1)
                else:
                    return None
            native_node.set_float_attr("eps", float(eps))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "transpose":
            if len(node.args) != 3 or node.kwargs:
                return None
            source, dim0, dim1 = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in (dim0, dim1)):
                return None
            native_node = graph.create_node("transpose", node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim0", int(dim0))
            native_node.set_int_attr("dim1", int(dim1))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "select":
            if len(node.args) != 3 or node.kwargs:
                return None
            source, dim, index = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            if any(isinstance(item, bool) or not isinstance(item, int) for item in (dim, index)):
                return None
            native_node = graph.create_node("select", node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim", int(dim))
            native_node.set_int_attr("index", int(index))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "slice":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs:
                return None
            args = list(node.args[1:])
            if len(args) > 4:
                return None
            args.extend([None] * (4 - len(args)))
            dim, start, end, step = args
            if not isinstance(dim, int) or isinstance(dim, bool):
                return None
            if step is None:
                step = 1
            if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
                return None
            native_node = graph.create_node("slice", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("dim", int(dim))
            native_node.set_int_attr("step", int(step))
            for key, bound in (("start", start), ("end", end)):
                if bound is None:
                    native_node.set_int_attr("has_" + key, 0)
                elif isinstance(bound, int) and not isinstance(bound, bool):
                    native_node.set_int_attr("has_" + key, 1)
                    native_node.set_int_attr(key, int(bound))
                else:
                    return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "stack":
            if not node.args or not isinstance(node.args[0], (tuple, list)):
                return None
            tensors = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            if node.kwargs or not tensors or not isinstance(dim, int):
                return None
            native_node = graph.create_node("stack", node.name)
            for item in tensors:
                if not add_tensor_input(native_node, item):
                    return None
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "repeat":
            if len(node.args) != 2 or node.kwargs or not isinstance(node.args[0], Node):
                return None
            repeats = _int_list(node.args[1])
            if repeats is None or node.args[0] not in values:
                return None
            native_node = graph.create_node("repeat", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_ints_attr("repeats", repeats)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "index_select" or op_name == "gather":
            kwargs = dict(node.kwargs or {})
            if node.op == "call_method":
                if set(kwargs) - {"dim", "index"} or not node.args:
                    return None
                source = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "index" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                index = positional[1] if len(positional) > 1 else kwargs.get("index")
                if dim is None or index is None:
                    return None
            else:
                if len(node.args) > 3 or set(kwargs) - {"dim", "index"}:
                    return None
                source = node.args[0]
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "index" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                index = node.args[2] if len(node.args) > 2 else kwargs.get("index")
                if dim is None or index is None:
                    return None
            if not isinstance(source, Node) or source not in values:
                return None
            if not isinstance(dim, int) or isinstance(dim, bool):
                return None
            if node_value(index) is None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            if not add_tensor_input(native_node, index):
                return None
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "embedding":
            if len(node.args) != 7 or node.kwargs:
                return None
            index_node, weight_node, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse = node.args
            if not isinstance(index_node, Node) or not isinstance(weight_node, Node):
                return None
            if index_node not in values or weight_node not in values:
                return None
            if max_norm is not None or norm_type != 2.0:
                return None
            if not isinstance(padding_idx, (int, type(None))) or isinstance(padding_idx, bool):
                return None
            if not isinstance(scale_grad_by_freq, bool) or not isinstance(sparse, bool):
                return None
            native_node = graph.create_node("embedding", node.name)
            native_node.add_input(values[weight_node])
            native_node.add_input(values[index_node])
            native_node.set_int_attr("padding_idx", -1 if padding_idx is None else int(padding_idx))
            native_node.set_int_attr("scale_grad_by_freq", int(scale_grad_by_freq))
            native_node.set_int_attr("sparse", int(sparse))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"constant_pad_nd", "pad"}:
            if (len(node.args) != 3 and len(node.args) != 4) or node.kwargs:
                return None
            if len(node.args) == 4:
                source, pad, mode, value = node.args
                if mode != "constant":
                    return None
            else:
                source, pad, value = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            padding = _int_list(pad)
            if padding is None or not _is_scalar(value):
                return None
            native_node = graph.create_node("constant_pad_nd", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("pad", padding)
            if not _set_named_scalar_attr(native_node, "value", value):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"view", "reshape"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if kwargs:
                return None
            shape_args = list(node.args[1:])
            if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
                shape_args = list(shape_args[0])
            shape = _int_list(shape_args)
            if shape is None or node.args[0] not in values:
                return None
            native_node = graph.create_node("reshape", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_ints_attr("shape", shape)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "permute":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs or node.args[0] not in values:
                return None
            dims_args = list(node.args[1:])
            if len(dims_args) == 1 and isinstance(dims_args[0], (tuple, list)):
                dims_args = list(dims_args[0])
            dims = _int_list(dims_args)
            if dims is None:
                return None
            native_node = graph.create_node("permute", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_ints_attr("dims", dims)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "contiguous":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.args[0] not in values:
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"memory_format"} or len(node.args) > 2:
                return None
            memory_format = node.args[1] if len(node.args) > 1 else kwargs.get(
                "memory_format", 0
            )
            if not isinstance(memory_format, int) or isinstance(memory_format, bool):
                return None
            native_node = graph.create_node("contiguous", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("memory_format", int(memory_format))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "unsqueeze":
            if len(node.args) != 2 or not isinstance(node.args[0], Node):
                return None
            if node.kwargs or node.args[0] not in values:
                return None
            dim = node.args[1]
            if isinstance(dim, bool) or not isinstance(dim, int):
                return None
            native_node = graph.create_node("unsqueeze", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "squeeze":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim"} or len(node.args) > 2:
                return None
            if len(node.args) > 1 and "dim" in kwargs:
                return None
            dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
            source = node.args[0]
            if source not in values:
                return None
            native_node = graph.create_node("squeeze", node.name)
            native_node.add_input(values[source])
            if dim is not None:
                if isinstance(dim, bool):
                    return None
                if isinstance(dim, int):
                    native_node.set_int_attr("dim", int(dim))
                elif not _set_int_list_attr(native_node, "dims", dim):
                    return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "zeros_like":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs or len(node.args) not in {1, 4}:
                # The native node preserves the template dtype and device.
                # Explicit metadata changes require a typed factory node and
                # are kept on the general lowering path.
                return None
            if len(node.args) == 4 and (
                node.args[1] is not tensorplay.undefined
                or node.args[2] is not None
                or node.args[3] is not False
            ):
                return None
            source = node.args[0]
            if source not in values:
                return None
            sample = sample_value(source)
            if sample is None:
                sample = sample_value(node)
            try:
                shape = [int(item) for item in sample.shape]
            except (AttributeError, TypeError, ValueError):
                return None
            native_node = graph.create_node("zeros_like", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("shape", shape)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "permute_backward":
            if len(node.args) != 3 or node.kwargs:
                return None
            grad_node, source_node, dims = node.args
            if (
                not isinstance(grad_node, Node)
                or not isinstance(source_node, Node)
                or grad_node not in values
                or source_node not in values
                or not _set_int_list_attr(
                    native_node := graph.create_node("permute_backward", node.name),
                    "dims",
                    dims,
                )
            ):
                return None
            native_node.add_input(values[grad_node])
            native_node.add_input(values[source_node])
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "float":
            if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], Node):
                return None
            if node.args[0] not in values:
                return None
            native_node = graph.create_node("float", node.name)
            native_node.add_input(values[node.args[0]])
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "group_norm":
            if len(node.args) != 5 or node.kwargs:
                return None
            input_node, num_groups, weight_node, bias_node, eps = node.args
            if (
                not isinstance(input_node, Node)
                or input_node not in values
                or isinstance(num_groups, bool)
                or not isinstance(num_groups, int)
                or not isinstance(eps, numbers.Real)
            ):
                return None
            native_node = graph.create_node("group_norm", node.name)
            native_node.add_input(values[input_node])
            native_node.set_int_attr("num_groups", int(num_groups))
            native_node.set_float_attr("eps", float(eps))
            for attr_name, optional_node in (
                ("has_weight", weight_node),
                ("has_bias", bias_node),
            ):
                if optional_node is None:
                    native_node.set_int_attr(attr_name, 0)
                    continue
                if not add_tensor_input(native_node, optional_node):
                    return None
                native_node.set_int_attr(attr_name, 1)
            if keep_saved_results:
                extents = _traced_value(graph_module, node)
                shape = (
                    tuple(int(item) for item in extents.shape)
                    if extents is not None and hasattr(extents, "shape")
                    else ()
                )
                if len(shape) >= 2 and shape[1] % int(num_groups) == 0:
                    spatial = 1
                    for extent in shape[2:]:
                        spatial *= extent
                    # The native spelling reports the per-group mean and
                    # reciprocal standard deviation next to its result, so a
                    # gradient reads those statistics instead of reducing the
                    # input a second time.
                    native_node.op_type = "native_group_norm"
                    native_node.set_int_attr("N", int(shape[0]))
                    native_node.set_int_attr("C", int(shape[1]))
                    native_node.set_int_attr("HxW", int(spatial))
                    native_node.set_int_attr("group", int(num_groups))
                    values[node] = native_node.add_output()
                    saved_result_values.append(
                        (node, (native_node.add_output(), native_node.add_output()))
                    )
                    layout_values[node] = False
                    continue
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"avg_pool1d", "avg_pool2d", "avg_pool3d"}:
            if len(node.args) not in {6, 7} or node.kwargs:
                return None
            input_node, kernel_size, stride, padding, ceil_mode, count_include_pad = node.args[:6]
            divisor = node.args[6] if len(node.args) == 7 else None
            if not isinstance(input_node, Node) or input_node not in values:
                return None
            if not isinstance(ceil_mode, bool) or not isinstance(count_include_pad, bool):
                return None
            spatial_rank = int(op_name[-2])
            kernel = _spatial_int_list(kernel_size, length=spatial_rank)
            stride_values = _spatial_int_list(
                stride, default=kernel, length=spatial_rank
            )
            padding_values = _spatial_int_list(
                padding, default=[0] * spatial_rank, length=spatial_rank
            )
            if kernel is None or stride_values is None or padding_values is None:
                return None
            if divisor is not None and (
                isinstance(divisor, bool) or not isinstance(divisor, int)
            ):
                return None
            if op_name == "avg_pool1d" and divisor is not None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[input_node])
            native_node.set_ints_attr("kernel_size", kernel)
            native_node.set_ints_attr("stride", stride_values)
            native_node.set_ints_attr("padding", padding_values)
            native_node.set_int_attr("ceil_mode", int(ceil_mode))
            native_node.set_int_attr("count_include_pad", int(count_include_pad))
            if divisor is not None:
                native_node.set_int_attr("divisor_override", int(divisor))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "interpolate":
            if len(node.args) != 7 or node.kwargs:
                return None
            input_node, size, scale_factor, mode, align_corners, recompute, antialias = node.args
            if (
                not isinstance(input_node, Node)
                or input_node not in values
                or scale_factor is not None
                or mode != "nearest"
                or align_corners is not None
                or recompute is not None
                or antialias is not False
            ):
                return None
            sample = sample_value(input_node)
            try:
                spatial_rank = int(len(sample.shape)) - 2
            except (AttributeError, TypeError):
                return None
            if spatial_rank not in (1, 2, 3):
                return None
            output_size = _spatial_int_list(size, length=spatial_rank)
            if output_size is None:
                return None
            native_node = graph.create_node("interpolate", node.name)
            native_node.add_input(values[input_node])
            native_node.set_ints_attr("output_size", output_size)
            native_node.set_int_attr("spatial_rank", spatial_rank)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "scaled_dot_product_attention":
            if len(node.args) < 3 or len(node.args) > 8:
                return None
            query, key, value_node = node.args[:3]
            if any(not isinstance(item, Node) or item not in values for item in (query, key, value_node)):
                return None
            names = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
            options = dict(node.kwargs or {})
            for index, item in enumerate(node.args[3:]):
                if names[index] in options:
                    return None
                options[names[index]] = item
            if set(options) - set(names):
                return None
            if options.get("attn_mask") is not None:
                return None
            dropout_p = options.get("dropout_p", 0.0)
            if not isinstance(dropout_p, numbers.Real) or float(dropout_p) != 0.0:
                return None
            if options.get("scale") is not None or options.get("enable_gqa", False):
                return None
            is_causal = options.get("is_causal", False)
            if not isinstance(is_causal, bool):
                return None
            attention_inputs = []
            for input_node in (query, key, value_node):
                input_value = values[input_node]
                if autocast_dtype is not None and getattr(node.meta.get("val"), "dtype", None) == autocast_dtype:
                    input_value = saved_autocast_value(input_node, input_value)
                attention_inputs.append(input_value)
            if keep_saved_results:
                # The fused forward computes the softmax normalizer anyway, so
                # the route that reports it costs nothing here and spares the
                # gradient a pass that rebuilds the score matrix.
                native_node = graph.create_node(
                    "_scaled_dot_product_attention_with_lse", node.name
                )
                for input_value in attention_inputs:
                    native_node.add_input(input_value)
                native_node.set_int_attr("is_causal", int(is_causal))
                native_node.set_int_attr("impl", 0)
                values[node] = native_node.add_output()
                saved_result_values.append((node, (native_node.add_output(),)))
                layout_values[node] = False
                continue
            native_node = graph.create_node("scaled_dot_product_attention", node.name)
            for input_value in attention_inputs:
                native_node.add_input(input_value)
            native_node.set_int_attr("is_causal", int(is_causal))
            native_node.set_int_attr("impl", 0)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "dropout":
            if len(node.args) != 4 or node.kwargs or not isinstance(node.args[0], Node):
                return None
            input_node, probability, training, inplace = node.args
            if input_node not in values or not isinstance(probability, numbers.Real):
                return None
            if not isinstance(training, bool) or not isinstance(inplace, bool):
                return None
            if float(probability) == 0.0 or not training:
                values[node] = values[input_node]
                layout_values[node] = layout_values.get(input_node, False)
                continue
            if inplace:
                return None
            native_node = graph.create_node("dropout", node.name)
            native_node.add_input(values[input_node])
            native_node.set_float_attr("p", float(probability))
            native_node.set_int_attr("training", int(training))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        alpha_form = op_name in {"add", "sub"} and (
            len(node.args) == 3
            or (len(node.args) == 2 and "alpha" in (node.kwargs or {}))
        )
        if op_name in {"add", "sub", "mul", "div", "pow", "matmul"} and not alpha_form:
            if len(node.args) != 2 or node.kwargs:
                return None
            lhs, rhs = node.args
            lhs_is_tensor = isinstance(lhs, (Node, tensor_type))
            rhs_is_tensor = isinstance(rhs, (Node, tensor_type))
            if not lhs_is_tensor and not _is_scalar(lhs):
                return None
            if not rhs_is_tensor and not _is_scalar(rhs):
                return None
            if not lhs_is_tensor and not rhs_is_tensor:
                return None
            native_node = graph.create_node(op_name, node.name)
            if lhs_is_tensor:
                if not add_tensor_input(native_node, lhs):
                    return None
            if rhs_is_tensor:
                if not add_tensor_input(native_node, rhs):
                    return None
            if not lhs_is_tensor:
                _set_scalar_attr(native_node, lhs, 0)
            elif not rhs_is_tensor:
                _set_scalar_attr(native_node, rhs, 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"conv2d", "conv2d_relu"}:
            if len(node.args) != 7:
                return None
            input_node, weight_node, bias_node, stride, padding, dilation, groups = node.args
            fused_relu = fused_relu_convs.get(node)
            use_conv_relu = op_name == "conv2d_relu" or (
                fused_relu is not None and not peel_conv_bias
            )
            # A consumer must be created after the layout-conversion node
            # feeding it: native execution walks nodes in creation order.
            conv_input = channels_last_value(input_node)
            if conv_input is None:
                return None
            if autocast_dtype is not None:
                conv_input = saved_autocast_value(input_node, conv_input)
            folded_inputs = folded_native_inputs.get(node)
            bias_input = None
            bias_tensor = None
            if folded_inputs is not None:
                weight_input = folded_inputs[0]
                bias_input = folded_inputs[1]
                folded_spec = folded_convs.get(node)
                bias_tensor = folded_spec[2] if folded_spec is not None else None
            else:
                weight_input = channels_last_value(weight_node)
                if weight_input is None:
                    return None
                if autocast_dtype is not None:
                    weight_input = saved_autocast_value(weight_node, weight_input)
                if bias_node is not None:
                    bias_input = node_value(bias_node)
                    if bias_input is None:
                        return None
                    if peel_conv_bias and not use_conv_relu:
                        try:
                            bias_tensor = graph_module._get_attr(bias_node.target)
                        except (AttributeError, TypeError):
                            return None
            native_node = graph.create_node(
                "conv2d_relu" if use_conv_relu else "conv2d",
                node.name,
            )
            native_node.add_input(conv_input)
            native_node.add_input(weight_input)
            if bias_node is None and folded_inputs is None:
                native_node.set_int_attr("has_bias", 0)
            if bias_input is None:
                native_node.set_int_attr("has_bias", 0)
            elif not peel_conv_bias or use_conv_relu:
                native_node.add_input(bias_input)
                native_node.set_int_attr("has_bias", 1)
            else:
                # the cuDNN call because cuDNN is slower with it.  Keep the
                # bias as a broadcast pointwise input after the convolution.
                native_node.set_int_attr("has_bias", 0)
            for key, value, default in (
                ("stride", stride, [1, 1]),
                ("padding", padding, [0, 0]),
                ("dilation", dilation, [1, 1]),
            ):
                normalized = _spatial_int_list(value, default=default)
                if normalized is None or not _set_int_list_attr(
                    native_node, key, normalized
                ):
                    return None
            if isinstance(groups, bool) or not isinstance(groups, int):
                return None
            native_node.set_int_attr("groups", int(groups))
            conv_value = native_node.add_output()
            values[node] = conv_value
            layout_values[node] = use_channels_last

            if peel_conv_bias and bias_input is not None and not use_conv_relu:
                if bias_tensor is None or not hasattr(bias_tensor, "shape"):
                    return None
                bias_shape = tuple(int(item) for item in bias_tensor.shape)
                if len(bias_shape) != 1:
                    return None
                bias_view = graph.create_node("reshape", f"{node.name}_bias_view")
                bias_view.add_input(bias_input)
                bias_view.set_ints_attr("shape", [1, bias_shape[0], 1, 1])
                bias_value = bias_view.add_output()
                add_node = graph.create_node(
                    "add_relu" if fused_relu is not None else "add",
                    f"{node.name}_bias_add",
                )
                add_node.add_input(conv_value)
                add_node.add_input(bias_value)
                values[node] = add_node.add_output()
                layout_values[node] = use_channels_last and fused_relu is not None
                if fused_relu is not None:
                    fused_relu_nodes.add(fused_relu)
            elif use_conv_relu and fused_relu is not None:
                fused_relu_nodes.add(fused_relu)
            continue

        if op_name == "batch_norm":
            if len(node.args) != 8:
                return None
            folded_batch_norm = next(
                (
                    batch_norm
                    for batch_norm, _, _ in folded_convs.values()
                    if batch_norm is node
                ),
                None,
            )
            if folded_batch_norm is not None:
                values[node] = values[node.args[0]]
                layout_values[node] = layout_values.get(node.args[0], False)
                continue
            input_node, running_mean, running_var, weight, bias, training, momentum, eps = node.args
            native_node = graph.create_node("batch_norm", node.name)
            if not add_tensor_input(native_node, input_node):
                return None
            optional_inputs = (
                ("has_running_mean", running_mean),
                ("has_running_var", running_var),
                ("has_weight", weight),
                ("has_bias", bias),
            )
            for attr_name, optional_node in optional_inputs:
                if optional_node is None:
                    native_node.set_int_attr(attr_name, 0)
                    continue
                if not add_tensor_input(native_node, optional_node):
                    return None
                native_node.set_int_attr(attr_name, 1)
            if not isinstance(training, bool) or not isinstance(momentum, numbers.Real) or not isinstance(eps, numbers.Real):
                return None
            native_node.set_int_attr("training", int(training))
            native_node.set_float_attr("momentum", float(momentum))
            native_node.set_float_attr("eps", float(eps))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"max_pool1d", "max_pool2d", "max_pool3d"}:
            if len(node.args) not in {6, 7}:
                return None
            input_node, kernel_size, stride, padding, dilation, ceil_mode = node.args[:6]
            return_indices = node.args[6] if len(node.args) == 7 else False
            if not isinstance(return_indices, bool) or not isinstance(ceil_mode, bool):
                return None
            spatial_rank = int(op_name[-2])
            kernel = _spatial_int_list(kernel_size, length=spatial_rank)
            stride_values = _spatial_int_list(stride, default=kernel, length=spatial_rank)
            padding_values = _spatial_int_list(
                padding, default=[0] * spatial_rank, length=spatial_rank
            )
            dilation_values = _spatial_int_list(
                dilation, default=[1] * spatial_rank, length=spatial_rank
            )
            if any(value is None for value in (kernel, stride_values, padding_values, dilation_values)):
                return None
            native_op = f"{op_name}_with_indices" if return_indices else op_name
            native_node = graph.create_node(native_op, node.name)
            if not add_tensor_input(native_node, input_node):
                return None
            native_node.set_ints_attr("kernel_size", kernel)
            native_node.set_ints_attr("stride", stride_values)
            native_node.set_ints_attr("padding", padding_values)
            native_node.set_ints_attr("dilation", dilation_values)
            native_node.set_int_attr("ceil_mode", int(ceil_mode))
            if return_indices:
                values[node] = (native_node.add_output(), native_node.add_output())
            else:
                values[node] = native_node.add_output()
            # The cuDNN tensor descriptor and output follow the input layout;
            # a later convolution can therefore consume the max-pool result
            # without an NCHW round-trip.
            layout_values[node] = use_channels_last and layout_values.get(
                input_node, False
            )
            continue

        if op_name in {"adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d"}:
            if len(node.args) != 2:
                return None
            spatial_rank = int(op_name[-2])
            output_size = _spatial_int_list(node.args[1], length=spatial_rank)
            if output_size is None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.set_ints_attr("output_size", output_size)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"}:
            if len(node.args) != 2:
                return None
            spatial_rank = int(op_name[-2])
            output_size = _spatial_int_list(node.args[1], length=spatial_rank)
            if output_size is None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.set_ints_attr("output_size", output_size)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "flatten":
            # The canonical signature is (input, start_dim=0, end_dim=-1);
            # capture stamps the end_dim default, so both the two- and
            # three-argument forms arrive here.  The native node flattens
            # (1, -1) only, which is what every accepted form must request.
            if (
                node.kwargs
                or len(node.args) not in (2, 3)
                or node.args[1] != 1
                or (len(node.args) == 3 and node.args[2] != -1)
            ):
                return None
            native_node = graph.create_node("flatten", node.name)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            native_node.set_int_attr("start_dim", 1)
            native_node.set_int_attr("end_dim", -1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"sum", "mean"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim", "keepdim"}:
                return None
            if node.op == "call_method":
                source = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "keepdim" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                keepdim = positional[1] if len(positional) > 1 else kwargs.get(
                    "keepdim", False
                )
            else:
                source = node.args[0]
                if len(node.args) > 3:
                    return None
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "keepdim" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                keepdim = (
                    node.args[2]
                    if len(node.args) > 2
                    else kwargs.get("keepdim", False)
                )
            if source not in values or not isinstance(keepdim, bool):
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            if dim is not None:
                dims = [int(dim)] if isinstance(dim, int) else _int_list(dim)
                if dims is None:
                    return None
                native_node.set_ints_attr("dim", dims)
                native_node.set_int_attr("keepdim", int(keepdim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "unsqueeze":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim"} or len(node.args) > 2:
                return None
            if len(node.args) > 1 and "dim" in kwargs:
                return None
            dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
            source = node.args[0]
            if source not in values or isinstance(dim, bool) or not isinstance(dim, int):
                return None
            native_node = graph.create_node("unsqueeze", node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "expand":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"size", "implicit"}:
                return None
            source = node.args[0]
            if len(node.args) > 2 or (len(node.args) > 1 and "size" in kwargs):
                return None
            shape = node.args[1] if len(node.args) > 1 else kwargs.get("size")
            if source not in values or _int_list(shape) is None:
                return None
            native_node = graph.create_node("expand", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("shape", _int_list(shape))
            if "implicit" in kwargs:
                if not isinstance(kwargs["implicit"], bool):
                    return None
                native_node.set_int_attr("implicit", int(kwargs["implicit"]))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "cat":
            if not node.args:
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim", "out"}:
                return None
            tensors = node.args[0]
            if not isinstance(tensors, (tuple, list)) or not tensors:
                return None
            if any(not isinstance(item, Node) or item not in values for item in tensors):
                return None
            if len(node.args) > 2 or (len(node.args) > 1 and "dim" in kwargs):
                return None
            dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim", 0)
            if not isinstance(dim, int) or kwargs.get("out") is not None:
                return None
            native_node = graph.create_node("cat", node.name)
            for item in tensors:
                native_node.add_input(values[item])
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"chunk", "split", "split_with_sizes", "unbind"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            source = node.args[0]
            if source not in values:
                return None
            sample = node.meta.get("val")
            if not isinstance(sample, (tuple, list)) or not sample:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            if op_name == "chunk":
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"chunks", "dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 1 and "chunks" in kwargs:
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                chunks = node.args[1] if len(node.args) > 1 else kwargs.get("chunks")
                dim = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
                if not isinstance(chunks, int) or not isinstance(dim, int):
                    return None
                chunks = int(chunks)
                dim = int(dim)
                if chunks <= 0:
                    return None
                native_node.set_int_attr("chunks", chunks)
                native_node.set_int_attr("dim", dim)
            elif op_name == "split":
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"split_size", "split_size_or_sizes", "dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 1 and ("split_size" in kwargs or "split_size_or_sizes" in kwargs):
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                split_size = node.args[1] if len(node.args) > 1 else kwargs.get(
                    "split_size", kwargs.get("split_size_or_sizes")
                )
                dim = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
                if not isinstance(dim, int):
                    return None
                if isinstance(split_size, int) and not isinstance(split_size, bool):
                    native_node.set_int_attr("split_size", int(split_size))
                elif _set_int_list_attr(native_node, "split_sizes", split_size):
                    pass
                else:
                    return None
                native_node.set_int_attr("dim", int(dim))
            elif op_name == "split_with_sizes":
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"split_sizes", "dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 1 and "split_sizes" in kwargs:
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                split_sizes = node.args[1] if len(node.args) > 1 else kwargs.get("split_sizes")
                if not _set_int_list_attr(native_node, "split_sizes", split_sizes):
                    return None
                dim = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
                if not isinstance(dim, int):
                    return None
                native_node.set_int_attr(
                    "dim", int(dim)
                )
            else:
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"dim"} or len(node.args) > 2:
                    return None
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim", 0)
                if not isinstance(dim, int):
                    return None
                native_node.set_int_attr(
                    "dim", int(dim)
                )
            values[node] = tuple(native_node.add_output() for _ in sample)
            layout_values[node] = False
            continue

        if op_name == "add" and node in fused_add_relus:
            if len(node.args) == 2:
                input_node, other_node = node.args
                alpha = 1
            elif len(node.args) == 3:
                input_node, other_node, alpha = node.args
            else:
                return None
            if (
                alpha != 1
                or not isinstance(input_node, Node)
                or not isinstance(other_node, Node)
                or input_node not in values
                or other_node not in values
            ):
                return None
            fused = graph.create_node("add_relu", node.name)
            fused.add_input(values[input_node])
            fused.add_input(values[other_node])
            values[node] = fused.add_output()
            layout_values[node] = use_channels_last and (
                layout_values.get(input_node, False)
                or layout_values.get(other_node, False)
            )
            continue

        # ``alpha`` argument, so their graph node has
        # ``(input, other, alpha)`` even when alpha is the default 1, and
        # keyword-only spellings record alpha in the node kwargs.  Lower
        # that contract to the native pointwise IR instead of falling back
        # to a Python method call.  Non-unit alpha becomes a scalar multiply
        # and can be consumed by Stax's mul-add fusion pass.
        if op_name in {"add", "sub"} and (
            len(node.args) == 3
            or (len(node.args) == 2 and "alpha" in (node.kwargs or {}))
        ):
            if set(node.kwargs or {}) - {"alpha"}:
                return None
            if len(node.args) == 3 and not node.kwargs:
                input_node, other_node, alpha = node.args
            else:
                input_node, other_node = node.args
                alpha = node.kwargs.get("alpha", 1)
            if not isinstance(input_node, Node) or not _is_scalar(alpha):
                return None
            if input_node not in values:
                return None
            if isinstance(other_node, Node) and other_node not in values:
                return None

            if alpha == 1:
                binary = graph.create_node(op_name, node.name)
                binary.add_input(values[input_node])
                if isinstance(other_node, Node):
                    binary.add_input(values[other_node])
                elif _is_scalar(other_node):
                    _set_scalar_attr(binary, other_node, 1)
                else:
                    return None
                values[node] = binary.add_output()
                layout_values[node] = False
                continue

            if isinstance(other_node, Node):
                scale = graph.create_node("mul", f"{node.name}_alpha")
                scale.add_input(values[other_node])
                _set_scalar_attr(scale, alpha, 1)
                scaled_other = scale.add_output()

                binary = graph.create_node(op_name, node.name)
                binary.add_input(values[input_node])
                binary.add_input(scaled_other)
                values[node] = binary.add_output()
                layout_values[node] = False
                continue

            if _is_scalar(other_node):
                binary = graph.create_node(op_name, node.name)
                binary.add_input(values[input_node])
                _set_scalar_attr(binary, other_node * alpha, 1)
                values[node] = binary.add_output()
                layout_values[node] = False
                continue
            return None

        if op_name == "linear":
            if len(node.args) not in {2, 3} or any(
                not isinstance(arg, Node) and arg is not None for arg in node.args
            ):
                return None
            input_node, weight_node = node.args[:2]
            bias_node = node.args[2] if len(node.args) == 3 else None
            if not isinstance(input_node, Node) or not isinstance(weight_node, Node):
                return None
            if bias_node is not None and not isinstance(bias_node, Node):
                return None
            if any(
                value_node not in values
                for value_node in (input_node, weight_node, bias_node)
                if value_node is not None
            ):
                return None

            # One node, not a transpose plus a product plus an addition: the
            # bias belongs in the product's epilogue, and adding it back
            # separately costs a whole pass over the output.
            if _native_runs_linear():
                linear_input = values[input_node]
                linear_weight = values[weight_node]
                if autocast_dtype is not None and getattr(node.meta.get("val"), "dtype", None) == autocast_dtype:
                    linear_input = saved_autocast_value(input_node, linear_input)
                    linear_weight = saved_autocast_value(weight_node, linear_weight)
                fused = graph.create_node("linear", node.name)
                fused.add_input(linear_input)
                fused.add_input(linear_weight)
                if bias_node is not None:
                    fused.add_input(values[bias_node])
                values[node] = fused.add_output()
                layout_values[node] = False
                continue

            transpose = graph.create_node("t", f"{node.name}_weight_t")
            transpose.add_input(values[weight_node])
            transposed_weight = transpose.add_output()

            matmul = graph.create_node("matmul", f"{node.name}_matmul")
            matmul.add_input(values[input_node])
            matmul.add_input(transposed_weight)
            result = matmul.add_output()
            if bias_node is not None:
                add = graph.create_node("add", f"{node.name}_bias")
                add.add_input(result)
                add.add_input(values[bias_node])
                result = add.add_output()
            values[node] = result
            layout_values[node] = False
            continue

        input_nodes: list[Node] = []
        scalar_args: list[tuple[int, Any]] = []
        for position, arg in enumerate(node.args):
            if isinstance(arg, Node):
                input_nodes.append(arg)
            elif _is_scalar(arg):
                scalar_args.append((position, arg))
            else:
                return None
        if len(scalar_args) > 1:
            return None
        if op_name in {
            "neg",
            "pos",
            "abs",
            "sin",
            "cos",
            "exp",
            "log",
            "sigmoid",
            "sqrt",
            "square",
            "tanh",
            "relu",
            "conj",
        }:
            if len(node.args) != 1 or len(input_nodes) != 1:
                return None
        elif len(input_nodes) not in {1, 2} or len(node.args) not in {1, 2}:
            return None
        if any(input_node not in values for input_node in input_nodes):
            return None
        if op_name == "mm":
            if len(node.args) != 2 or len(input_nodes) != 2:
                return None
            native_node = graph.create_node("mm", node.name)
            native_node.add_input(values[input_nodes[0]])
            native_node.add_input(values[input_nodes[1]])
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue
        native_node = graph.create_node(op_name, node.name)
        for input_node in input_nodes:
            native_node.add_input(values[input_node])
        if scalar_args:
            _set_scalar_attr(native_node, scalar_args[0][1], scalar_args[0][0])
        if op_name == "relu":
            # Preserve the functional schema.  The executor may call relu_
            # only when the captured call explicitly requested mutation.
            native_node.set_int_attr(
                "inplace", int(node.kwargs.get("inplace", False))
            )
        values[node] = native_node.add_output()
        layout_values[node] = False

    try:
        fuser.finish(required_nodes)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None

    try:
        output_arg = graph_module.graph.output_node.args[0]
        output_values = _native_value_leaves(output_arg, values)
        output_spec = _native_output_spec(output_arg, values)
    except (IndexError, KeyError, RuntimeError, TypeError):
        return None
    if not output_values or any(value is None for value in output_values):
        return None

    for output_value in output_values:
        graph.register_output(output_value)
    public_output_count = len(output_values)
    registered_extra_outputs = 0
    for extra_node in extra_output_nodes or []:
        if extra_node not in values:
            return None
        extra_values = values[extra_node]
        if isinstance(extra_values, tuple):
            if not extra_values or any(value is None for value in extra_values):
                return None
            for extra_value in extra_values:
                graph.register_output(extra_value)
            registered_extra_outputs += len(extra_values)
        else:
            if extra_values is None:
                return None
            graph.register_output(extra_values)
            registered_extra_outputs += 1

    for converted in autocast_values.values():
        graph.register_output(converted)
        registered_extra_outputs += 1

    for _result_node, extra_values in saved_result_values:
        for extra_value in extra_values:
            graph.register_output(extra_value)
            registered_extra_outputs += 1

    # Buffer updates run last: each registered mutation output pairs with an
    # input position, and the wrapper copies it back after execution.
    mutation_outputs: list[tuple[int, int]] = []
    for position, value in mutations:
        graph.register_output(value)
        mutation_outputs.append(
            (position, public_output_count + registered_extra_outputs + len(mutation_outputs))
        )

    if use_fusion and not mutation_outputs:
        graph.fuse()
    return _NativeLowering(
        graph_module,
        graph,
        attribute_targets,
        constant_values,
        output_count=public_output_count + registered_extra_outputs,
        native_values=values,
        output_spec=output_spec,
        public_output_count=public_output_count,
        mutations=mutation_outputs,
        autocast_outputs=list(autocast_values),
        saved_result_outputs=[(node, len(values)) for node, values in saved_result_values],
    )


#: Names this layer owns.  The driver re-exports them, so the
#: import surface of the package does not change: the captured-graph walk.
__all__ = [
    "_NATIVE_OPS",
    "_NATIVE_OP_SUPPORT",
    "_fold_eval_conv_batch_norm",
    "_lower_native",
    "_native_output_spec",
    "_native_runs_linear",
    "_native_value_leaves",
]
