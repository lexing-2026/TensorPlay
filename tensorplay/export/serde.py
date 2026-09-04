"""Stable serialization of exported programs.

The program artifact is a JSON document describing the graph, the signature,
and the call contract; tensor payloads travel separately in the archive so
that loading never executes pickled program code.  Values that cannot be
represented structurally fall back to a pickled, base64-encoded payload and
are marked explicitly in the document.
"""

from __future__ import annotations

import base64
import importlib
import json
import pickle
from collections.abc import Mapping
from typing import Any

__all__ = ["SerializedArtifact", "deserialize", "serialize"]

SCHEMA_VERSION = 1
_NODE_VALUE: tuple[type, ...] = ()


class SerializedArtifact:
    """Container for the serialized program and its example inputs."""

    def __init__(self, exported_program: bytes, example_inputs: bytes) -> None:
        self.exported_program = exported_program
        self.example_inputs = example_inputs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SerializedArtifact):
            return NotImplemented
        return (
            self.exported_program == other.exported_program
            and self.example_inputs == other.example_inputs
        )

    def __repr__(self) -> str:
        return (
            f"SerializedArtifact(exported_program={len(self.exported_program)} bytes, "
            f"example_inputs={len(self.example_inputs)} bytes)"
        )


# -- primitive encodings ----------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _pickle_b64(value: Any, protocol: int) -> str:
    return _b64(pickle.dumps(value, protocol=protocol))


def _unpickle_b64(data: str) -> Any:
    return pickle.loads(_unb64(data))


def _resolve_reference(module: str, qualname: str) -> Any:
    obj = importlib.import_module(module)
    for atom in qualname.split("."):
        obj = getattr(obj, atom)
    return obj


def _encode_reference(obj: Any, protocol: int) -> dict[str, Any]:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if module and qualname and "<" not in qualname:
        try:
            resolved = _resolve_reference(module, qualname)
            if resolved is obj:
                return {"kind": "ref", "module": module, "qualname": qualname}
        except (ImportError, AttributeError):
            pass
    return {"kind": "pickle", "data": _pickle_b64(obj, protocol)}


def _decode_reference(entry: dict[str, Any]) -> Any:
    if entry["kind"] == "ref":
        return _resolve_reference(entry["module"], entry["qualname"])
    return _unpickle_b64(entry["data"])


def _encode_value(value: Any, node_names: set[str], protocol: int) -> Any:
    if isinstance(value, _NODE_VALUE):
        return {"__node__": value.name}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_value(item, node_names, protocol) for item in value]}
    if isinstance(value, list):
        return {"__list__": [_encode_value(item, node_names, protocol) for item in value]}
    if isinstance(value, dict):
        return {
            "__dict__": [
                [key, _encode_value(item, node_names, protocol)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, slice):
        return {
            "__slice__": [
                _encode_value(value.start, node_names, protocol),
                _encode_value(value.stop, node_names, protocol),
                _encode_value(value.step, node_names, protocol),
            ]
        }
    return {"__pickle__": _pickle_b64(value, protocol)}


def _decode_value(value: Any, nodes: Mapping[str, Any]) -> Any:
    if isinstance(value, dict):
        if "__node__" in value:
            return nodes[value["__node__"]]
        if "__tuple__" in value:
            return tuple(_decode_value(item, nodes) for item in value["__tuple__"])
        if "__list__" in value:
            return [_decode_value(item, nodes) for item in value["__list__"]]
        if "__dict__" in value:
            return {key: _decode_value(item, nodes) for key, item in value["__dict__"]}
        if "__slice__" in value:
            start, stop, step = (_decode_value(item, nodes) for item in value["__slice__"])
            return slice(start, stop, step)
        if "__pickle__" in value:
            return _unpickle_b64(value["__pickle__"])
    return value


def _encode_target(node: Any, protocol: int) -> dict[str, Any]:
    op = node.op
    target = node.target
    if op in {"placeholder", "output", "get_attr"}:
        return {"kind": "name", "name": str(target)}
    if op == "call_method":
        return {"kind": "method", "name": str(target)}
    if op == "call_module":
        return {"kind": "name", "name": str(target)}
    return _encode_reference(target, protocol)


def _decode_target(op: str, entry: dict[str, Any]) -> Any:
    kind = entry.get("kind")
    if kind in {"name", "method"}:
        return entry["name"]
    return _decode_reference(entry)


# -- tree specs ---------------------------------------------------------------


def _treespec_encode(spec: Any, protocol: int) -> dict[str, Any]:
    if spec.type is None:
        return {"t": "leaf"}
    children = [_treespec_encode(child, protocol) for child in spec.children_specs]
    spec_type = spec.type
    context = spec.context
    if isinstance(spec_type, type):
        if spec_type in (dict, list, tuple):
            return {"t": spec_type.__name__, "ctx": _plain(context), "c": children}
        entry = _encode_reference(spec_type, protocol)
        entry.update({"ctx": _plain(context), "c": children})
        return entry
    return {"t": "pickle", "data": _pickle_b64(spec_type, protocol), "c": children}


def _treespec_decode(entry: dict[str, Any]) -> Any:
    from ..graph._pytree import TreeSpec

    kind = entry.get("t")
    children = tuple(_treespec_decode(child) for child in entry.get("c", ()))
    if kind == "leaf":
        return TreeSpec(None)
    context = entry.get("ctx")
    if kind in {"dict", "list", "tuple"}:
        if kind == "dict" and isinstance(context, (list, tuple)):
            context = tuple(context)
        container = {"dict": dict, "list": list, "tuple": tuple}[kind]
        return TreeSpec(container, context, children)
    if kind == "ref":
        spec_type = _resolve_reference(entry["module"], entry["qualname"])
        return TreeSpec(spec_type, _restore_plain(context), children)
    return TreeSpec(_unpickle_b64(entry["data"]), _restore_plain(context), children)


def _plain(value: Any) -> Any:
    """Reduce a tree context to JSON-able data, pickling what remains."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return {"__tuple__": [_plain(item) for item in value]}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {"__map__": [[key, _plain(item)] for key, item in value.items()]}
    if isinstance(value, slice):
        return {"__slice__": [_plain(value.start), _plain(value.stop), _plain(value.step)]}
    return {"__pickle__": _pickle_b64(value, 4)}


def _restore_plain(value: Any) -> Any:
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(_restore_plain(item) for item in value["__tuple__"])
        if "__map__" in value:
            return {key: _restore_plain(item) for key, item in value["__map__"]}
        if "__slice__" in value:
            start, stop, step = (_restore_plain(item) for item in value["__slice__"])
            return slice(start, stop, step)
        if "__pickle__" in value:
            return _unpickle_b64(value["__pickle__"])
    if isinstance(value, list):
        return [_restore_plain(item) for item in value]
    return value


# -- argument specs and signature ---------------------------------------------


_ARGUMENT_FIELDS = {
    "TensorArgument": ("name",),
    "TokenArgument": ("name",),
    "SymIntArgument": ("name",),
    "SymFloatArgument": ("name",),
    "SymBoolArgument": ("name",),
    "CustomObjArgument": ("name", "class_fqn", "fake_val"),
    "ConstantArgument": ("name", "value"),
}


def _encode_argument(arg: Any) -> dict[str, Any]:
    for type_name, fields in _ARGUMENT_FIELDS.items():
        if type(arg).__name__ == type_name:
            payload: dict[str, Any] = {"t": type_name}
            for field in fields:
                payload[field] = getattr(arg, field, None)
            return payload
    raise TypeError(f"cannot serialize argument spec {type(arg).__name__}")


def _decode_argument(entry: dict[str, Any]) -> Any:
    from . import graph_signature as gs

    type_name = entry["t"]
    cls = getattr(gs, type_name)
    fields = _ARGUMENT_FIELDS[type_name]
    kwargs = {field: entry.get(field) for field in fields}
    if type_name == "CustomObjArgument":
        kwargs["fake_val"] = None
    return cls(**kwargs)


def _encode_spec(spec: Any) -> dict[str, Any]:
    return {
        "kind": spec.kind.name,
        "arg": _encode_argument(spec.arg),
        "target": spec.target,
        "persistent": getattr(spec, "persistent", None),
    }


def _build_spec(entry: dict[str, Any]) -> Any:
    from .graph_signature import InputKind, InputSpec, OutputKind, OutputSpec

    if entry["kind"] in InputKind.__members__:
        return InputSpec(
            InputKind[entry["kind"]],
            _decode_argument(entry["arg"]),
            entry.get("target"),
            entry.get("persistent"),
        )
    return OutputSpec(
        OutputKind[entry["kind"]], _decode_argument(entry["arg"]), entry.get("target")
    )


def _encode_call_signature(signature: Any) -> dict[str, Any]:
    if signature is None:
        return None
    return {
        "inputs": [_encode_argument(arg) for arg in signature.inputs],
        "outputs": [_encode_argument(arg) for arg in signature.outputs],
        "in_spec": _treespec_encode(signature.in_spec, 4) if signature.in_spec else None,
        "out_spec": _treespec_encode(signature.out_spec, 4) if signature.out_spec else None,
        "forward_arg_names": signature.forward_arg_names,
    }


def _decode_call_signature(entry: dict[str, Any]) -> Any:
    from .exported_program import ModuleCallSignature
    from ..graph._pytree import TreeSpec

    if entry is None:
        return None
    leaf = TreeSpec(None)
    return ModuleCallSignature(
        inputs=[_decode_argument(arg) for arg in entry["inputs"]],
        outputs=[_decode_argument(arg) for arg in entry["outputs"]],
        in_spec=_treespec_decode(entry["in_spec"]) if entry.get("in_spec") else leaf,
        out_spec=_treespec_decode(entry["out_spec"]) if entry.get("out_spec") else leaf,
        forward_arg_names=entry.get("forward_arg_names"),
    )


# -- dynamic shapes -----------------------------------------------------------


def _encode_dynamic_shapes(spec: Any) -> Any:
    if isinstance(spec, dict):
        return {
            "__map__": [
                [key, _encode_dynamic_shapes(value)] for key, value in spec.items()
            ]
        }
    if isinstance(spec, (list, tuple)):
        wrapped = [_encode_dynamic_shapes(item) for item in spec]
        return {"__tuple__": wrapped} if isinstance(spec, tuple) else wrapped
    from .dynamic_shapes import Dim, _DerivedDim, _StaticDim

    if isinstance(spec, _DerivedDim):
        return {
            "__derived_dim__": {
                "name": spec.__name__,
                "root": spec.root.__name__,
                "scale": spec.scale,
                "offset": spec.offset,
                "min": spec.min,
                "max": spec.max,
            }
        }
    if isinstance(spec, _StaticDim):
        return {"__static_dim__": spec.value}
    if isinstance(spec, Dim):
        return {
            "__dim__": {
                "name": spec.__name__,
                "min": spec.min,
                "max": spec.max,
            }
        }
    return spec


def _decode_dynamic_shapes(value: Any) -> Any:
    from .dynamic_shapes import Dim, _DerivedDim, _StaticDim

    if isinstance(value, dict):
        if "__dim__" in value:
            entry = value["__dim__"]
            return Dim(entry["name"], min=entry["min"], max=entry["max"])
        if "__derived_dim__" in value:
            entry = value["__derived_dim__"]
            root = Dim(entry["root"], min=entry["min"] or 0, max=entry["max"])
            return _DerivedDim(entry["name"], root, entry["scale"], entry["offset"])
        if "__static_dim__" in value:
            return _StaticDim(value["__static_dim__"])
        if "__map__" in value:
            return {key: _decode_dynamic_shapes(item) for key, item in value["__map__"]}
        if "__tuple__" in value:
            return tuple(_decode_dynamic_shapes(item) for item in value["__tuple__"])
        return {key: _decode_dynamic_shapes(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_dynamic_shapes(item) for item in value]
    return value


# -- program document ----------------------------------------------------------


def serialize(
    program: Any,
    opset_version: Any = None,
    pickle_protocol: int = 4,
    serialize_state_dict: bool = True,
    serialize_constants: bool = True,
) -> SerializedArtifact:
    """Serialize a program into JSON + example-input artifacts."""

    del serialize_state_dict, serialize_constants
    graph_module = program.graph_module
    node_names = {node.name for node in graph_module.graph.nodes}

    nodes_payload = []
    for node in graph_module.graph.nodes:
        nodes_payload.append(
            {
                "op": node.op,
                "target": _encode_target(node, pickle_protocol),
                "args": _encode_value(node.args, node_names, pickle_protocol),
                "kwargs": _encode_value(node.kwargs, node_names, pickle_protocol),
                "name": node.name,
            }
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "dialect": getattr(program.verifier, "dialect", "STABLE"),
        "graph": {"nodes": nodes_payload},
        "signature": {
            "input_specs": [_encode_spec(spec) for spec in program.graph_signature.input_specs],
            "output_specs": [_encode_spec(spec) for spec in program.graph_signature.output_specs],
        },
        "module_call_graph": [
            {"fqn": entry.fqn, "signature": _encode_call_signature(entry.signature)}
            for entry in program.module_call_graph
        ],
        "range_constraints": {
            str(name): bounds
            for name, bounds in (program.range_constraints or {}).items()
        },
        "equality_constraints": [
            {
                "sites": [[name, int(dim)] for name, dim in constraint.sites],
                "name": constraint.name,
            }
            for constraint in (program.equality_constraints or [])
        ],
        "dynamic_shapes": _encode_dynamic_shapes(program.dynamic_shapes),
        "meta": {
            "num_mutations": graph_module.meta.get("num_mutations", 0),
            "in_spec": _treespec_encode(graph_module.meta["in_spec"], pickle_protocol)
            if graph_module.meta.get("in_spec") is not None
            else None,
            "out_spec": _treespec_encode(graph_module.meta["out_spec"], pickle_protocol)
            if graph_module.meta.get("out_spec") is not None
            else None,
        },
        "opset_version": opset_version or {},
    }
    artifact = SerializedArtifact(
        exported_program=json.dumps(document).encode("utf-8"),
        example_inputs=pickle.dumps(
            dict(program.example_inputs), protocol=pickle_protocol
        ),
    )
    return artifact


def deserialize(
    artifact: Any,
    state_dict: Mapping[str, Any] | None = None,
    constants: Mapping[str, Any] | None = None,
    example_inputs: Any = None,
) -> Any:
    """Rebuild a program from :func:`serialize` artifacts."""

    import tensorplay as tp
    from ..graph import Graph, GraphModule
    from .exported_program import EqualityConstraint, ExportedProgram, ModuleCallEntry
    from .graph_signature import ExportGraphSignature

    if isinstance(artifact, SerializedArtifact):
        document = json.loads(artifact.exported_program.decode("utf-8"))
        examples = pickle.loads(artifact.example_inputs)
    elif isinstance(artifact, (bytes, bytearray)):
        document = json.loads(bytes(artifact).decode("utf-8"))
        examples = example_inputs or {}
    elif isinstance(artifact, Mapping):
        document = artifact
        examples = example_inputs or {}
    else:
        raise TypeError(f"unsupported artifact type {type(artifact).__name__}")

    if int(document.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            f"archive schema version {document.get('schema_version')!r} is not "
            f"supported (expected {SCHEMA_VERSION})"
        )

    state = dict(state_dict or {})
    const_values = dict(constants or {})

    graph = Graph()
    nodes: dict[str, Any] = {}
    state_targets: dict[str, tuple[str, str]] = {}

    for spec_entry in document["signature"]["input_specs"]:
        target = spec_entry.get("target")
        if target and spec_entry["kind"] in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR", "CUSTOM_OBJ"}:
            state_targets[spec_entry["arg"]["name"]] = (
                target,
                spec_entry["kind"],
            )

    for entry in document["graph"]["nodes"]:
        op = entry["op"]
        target = _decode_target(op, entry["target"])
        args = _decode_value(entry["args"], nodes)
        kwargs = _decode_value(entry["kwargs"], nodes)
        name = entry["name"]
        if op == "output":
            node = graph.create_node("output", target, tuple(args), dict(kwargs), name=name)
        elif op == "placeholder" and name in state_targets:
            # flat-lifted capture: state enters as a placeholder
            node = graph.create_node("placeholder", name, tuple(args), dict(kwargs), name=name)
        else:
            node = graph.create_node(op, target, tuple(args), dict(kwargs), name=name)
        nodes[name] = node

    root = tp.nn.Module()
    for entry in document["signature"]["input_specs"]:
        kind = entry["kind"]
        target = entry.get("target")
        if kind not in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR", "CUSTOM_OBJ"}:
            continue
        if not target:
            continue
        if target in state:
            value = state[target]
        elif target in const_values:
            value = const_values[target]
        else:
            raise KeyError(
                f"no serialized value for {kind.lower()} {target!r}; supply it "
                f"via state_dict or constants"
            )
        _assign_state(root, target, value)

    for node in graph.nodes:
        if node.op != "get_attr":
            continue
        target = str(node.target)
        if target in state or target in const_values:
            continue
        raise KeyError(
            f"no serialized value for attribute {target!r}; supply it via "
            f"state_dict or constants"
        )

    graph_module = GraphModule(root, graph, None)
    meta = document.get("meta", {})
    graph_module.meta["num_mutations"] = int(meta.get("num_mutations", 0) or 0)
    if meta.get("in_spec"):
        graph_module.meta["in_spec"] = _treespec_decode(meta["in_spec"])
    if meta.get("out_spec"):
        graph_module.meta["out_spec"] = _treespec_decode(meta["out_spec"])

    signature = ExportGraphSignature(
        input_specs=[_build_spec(entry) for entry in document["signature"]["input_specs"]],
        output_specs=[_build_spec(entry) for entry in document["signature"]["output_specs"]],
    )
    program = ExportedProgram(
        graph_module=graph_module,
        graph_signature=signature,
        example_inputs=examples or {},
        dynamic_shapes=_decode_dynamic_shapes(document.get("dynamic_shapes")),
        module_call_graph=[
            ModuleCallEntry(entry["fqn"], _decode_call_signature(entry["signature"]))
            for entry in document.get("module_call_graph", [])
        ],
        range_constraints={
            key: value for key, value in document.get("range_constraints", {}).items()
        },
        equality_constraints=[
            EqualityConstraint(
                tuple((name, int(dim)) for name, dim in entry["sites"]),
                name=entry.get("name"),
            )
            for entry in document.get("equality_constraints", [])
        ],
    )
    return program


def _assign_state(root: Any, target: str, value: Any) -> None:
    import tensorplay as tp

    parent_name, _, leaf = target.rpartition(".")
    parent: Any = root
    if parent_name:
        parent = _ensure_parent(root, parent_name)
    if isinstance(value, tp.nn.Parameter):
        parent.register_parameter(leaf, value)
    elif hasattr(value, "shape") and hasattr(value, "requires_grad"):
        parent.register_buffer(leaf, value)
    else:
        setattr(parent, leaf, value)


def _ensure_parent(root: Any, path: str) -> Any:
    import tensorplay as tp

    parent: Any = root
    for atom in path.split("."):
        child = getattr(parent, atom, None)
        if child is None:
            child = tp.nn.Module()
            setattr(parent, atom, child)
        parent = child
    return parent


try:
    from ..graph.node import Node as _NodeType
    from ..graph.proxy import Proxy as _ProxyType

    _NODE_VALUE = (_NodeType, _ProxyType)
except Exception:  # pragma: no cover - import ordering safety
    _NODE_VALUE = ()
