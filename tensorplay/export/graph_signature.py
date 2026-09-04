"""Structured input and output descriptions for exported graphs."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.node import Node

__all__ = [
    "ArgumentSpec",
    "ConstantArgument",
    "CustomObjArgument",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "GraphSignature",
    "InputKind",
    "InputSpec",
    "OutputKind",
    "OutputSpec",
    "SymBoolArgument",
    "SymFloatArgument",
    "SymIntArgument",
    "TensorArgument",
    "TokenArgument",
]


@dataclasses.dataclass
class TensorArgument:
    name: str


@dataclasses.dataclass
class TokenArgument:
    name: str


@dataclasses.dataclass
class SymIntArgument:
    name: str


@dataclasses.dataclass
class SymFloatArgument:
    name: str


@dataclasses.dataclass
class SymBoolArgument:
    name: str


@dataclasses.dataclass
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: Any = None


@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: int | float | bool | str | None


ArgumentSpec = (
    TensorArgument
    | SymIntArgument
    | SymFloatArgument
    | SymBoolArgument
    | ConstantArgument
    | CustomObjArgument
    | TokenArgument
)


class InputKind(Enum):
    USER_INPUT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    CONSTANT_TENSOR = auto()
    CUSTOM_OBJ = auto()
    TOKEN = auto()


class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    PARAMETER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()
    TOKEN = auto()


_ARGUMENT_TYPES = (
    TensorArgument,
    SymIntArgument,
    SymFloatArgument,
    SymBoolArgument,
    ConstantArgument,
    CustomObjArgument,
    TokenArgument,
)


@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: str | None = None
    persistent: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, InputKind):
            raise TypeError(f"input kind must be InputKind, got {type(self.kind).__name__}")
        if not isinstance(self.arg, _ARGUMENT_TYPES):
            raise TypeError(f"invalid input argument type: {type(self.arg).__name__}")
        if self.kind is InputKind.BUFFER and self.persistent is None:
            raise ValueError("buffer input requires a persistence flag")

    def __str__(self) -> str:
        target = "" if self.target is None else f" target={self.target!r}"
        persistent = "" if self.persistent is None else f" persistent={self.persistent}"
        return f"{self.arg.name}: {self.kind.name}{target}{persistent}"


@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OutputKind):
            raise TypeError(f"output kind must be OutputKind, got {type(self.kind).__name__}")
        if not isinstance(self.arg, _ARGUMENT_TYPES):
            raise TypeError(f"invalid output argument type: {type(self.arg).__name__}")

    def __str__(self) -> str:
        target = "" if self.target is None else f" target={self.target!r}"
        return f"{self.arg.name}: {self.kind.name}{target}"


@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    """Describe lifted state, user values, mutations, and graph outputs."""

    input_specs: list[InputSpec]
    output_specs: list[OutputSpec]

    def __post_init__(self) -> None:
        self.input_specs = list(self.input_specs)
        self.output_specs = list(self.output_specs)

    @property
    def parameters(self) -> Collection[str]:
        return tuple(
            spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.PARAMETER and isinstance(spec.target, str)
        )

    @property
    def buffers(self) -> Collection[str]:
        return tuple(
            spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.BUFFER and isinstance(spec.target, str)
        )

    @property
    def non_persistent_buffers(self) -> Collection[str]:
        return tuple(
            spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.BUFFER
            and spec.persistent is False
            and isinstance(spec.target, str)
        )

    @property
    def lifted_tensor_constants(self) -> Collection[str]:
        return tuple(
            spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.CONSTANT_TENSOR and isinstance(spec.target, str)
        )

    @property
    def lifted_custom_objs(self) -> Collection[str]:
        return tuple(
            spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.CUSTOM_OBJ and isinstance(spec.target, str)
        )

    # neutral aliases: tensor constants and custom objects live in the same
    # input-spec table; callers may prefer the shorter names
    @property
    def constants(self) -> Collection[str]:
        return self.lifted_tensor_constants

    @property
    def tensor_constants(self) -> Collection[str]:
        return self.lifted_tensor_constants

    @property
    def custom_objs(self) -> Collection[str]:
        return self.lifted_custom_objs

    def is_param(self, name: str) -> bool:
        """Whether ``name`` is a placeholder carrying a lifted parameter."""
        return any(
            spec.kind is InputKind.PARAMETER
            and isinstance(spec.arg, TensorArgument)
            and spec.arg.name == name
            for spec in self.input_specs
        )

    def is_buffer(self, name: str) -> bool:
        """Whether ``name`` is a placeholder carrying a lifted buffer."""
        return any(
            spec.kind is InputKind.BUFFER
            and isinstance(spec.arg, TensorArgument)
            and spec.arg.name == name
            for spec in self.input_specs
        )

    def get_param_to_buffer(self) -> Mapping[str, str]:
        """Map parameter targets to the buffer targets holding their gradients.

        Gradients are declared as ``GRADIENT_TO_PARAMETER`` outputs; a gradient
        for a parameter whose optimizer state lives in a buffer binds the two
        targets under the parameter's FQN.
        """
        result: dict[str, str] = {}
        gradient_targets = {
            spec.target
            for spec in self.output_specs
            if spec.kind is OutputKind.GRADIENT_TO_PARAMETER and isinstance(spec.target, str)
        }
        if not gradient_targets:
            return result
        buffer_targets = {
            spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.BUFFER and isinstance(spec.target, str)
        }
        for target in gradient_targets:
            if target in buffer_targets:
                result[target] = target
        return result

    @property
    def user_inputs(self) -> Collection[Any]:
        values: list[Any] = []
        for spec in self.input_specs:
            if spec.kind is not InputKind.USER_INPUT:
                continue
            if isinstance(spec.arg, ConstantArgument):
                values.append(spec.arg.value)
            else:
                values.append(spec.arg.name)
        return tuple(values)

    @property
    def user_outputs(self) -> Collection[Any]:
        values: list[Any] = []
        for spec in self.output_specs:
            if spec.kind not in (OutputKind.USER_OUTPUT, OutputKind.LOSS_OUTPUT):
                continue
            if isinstance(spec.arg, ConstantArgument):
                values.append(spec.arg.value)
            else:
                values.append(spec.arg.name)
        return tuple(values)

    @property
    def inputs_to_parameters(self) -> Mapping[str, str]:
        return {
            spec.arg.name: spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.PARAMETER
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }

    @property
    def inputs_to_buffers(self) -> Mapping[str, str]:
        return {
            spec.arg.name: spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.BUFFER
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }

    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]:
        return {
            spec.arg.name: spec.target
            for spec in self.input_specs
            if spec.kind is InputKind.CONSTANT_TENSOR
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }

    @property
    def buffers_to_mutate(self) -> Mapping[str, str]:
        return {
            spec.arg.name: spec.target
            for spec in self.output_specs
            if spec.kind is OutputKind.BUFFER_MUTATION
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }

    @property
    def parameters_to_mutate(self) -> Mapping[str, str]:
        return {
            spec.arg.name: spec.target
            for spec in self.output_specs
            if spec.kind is OutputKind.PARAMETER_MUTATION
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }

    @property
    def user_inputs_to_mutate(self) -> Mapping[str, str]:
        return {
            spec.arg.name: spec.target
            for spec in self.output_specs
            if spec.kind is OutputKind.USER_INPUT_MUTATION
            and isinstance(spec.arg, TensorArgument)
            and isinstance(spec.target, str)
        }

    @property
    def backward_signature(self) -> ExportBackwardSignature | None:
        loss_output: str | None = None
        gradients_to_parameters: dict[str, str] = {}
        gradients_to_user_inputs: dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind is OutputKind.LOSS_OUTPUT:
                if loss_output is not None or not isinstance(spec.arg, TensorArgument):
                    raise ValueError("loss output must be one tensor argument")
                loss_output = spec.arg.name
            elif spec.kind is OutputKind.GRADIENT_TO_PARAMETER:
                if isinstance(spec.arg, TensorArgument) and isinstance(spec.target, str):
                    gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind is OutputKind.GRADIENT_TO_USER_INPUT:
                if isinstance(spec.arg, TensorArgument) and isinstance(spec.target, str):
                    gradients_to_user_inputs[spec.arg.name] = spec.target
        if loss_output is None:
            return None
        return ExportBackwardSignature(
            gradients_to_parameters=gradients_to_parameters,
            gradients_to_user_inputs=gradients_to_user_inputs,
            loss_output=loss_output,
        )

    @property
    def input_tokens(self) -> Collection[str]:
        return tuple(
            spec.arg.name
            for spec in self.input_specs
            if spec.kind is InputKind.TOKEN and isinstance(spec.arg, TokenArgument)
        )

    @property
    def output_tokens(self) -> Collection[str]:
        return tuple(
            spec.arg.name
            for spec in self.output_specs
            if spec.kind is OutputKind.TOKEN and isinstance(spec.arg, TokenArgument)
        )

    @property
    def assertion_dep_token(self) -> Mapping[int, str] | None:
        """Position of the assertion dependency token output, if present."""

        tokens = self.output_tokens
        if not tokens:
            return None
        index = len(self.user_outputs) + len(self.buffers_to_mutate)
        return {index: tokens[0]}

    def replace_all_uses(self, old: str, new: str) -> None:
        """Rename a graph value across every input and output spec."""

        if not isinstance(old, str) or not isinstance(new, str):
            raise TypeError("replace_all_uses expects string names")
        for spec in (*self.output_specs, *self.input_specs):
            if spec.arg.name == old:
                spec.arg.name = new

    def get_replace_hook(self, replace_inputs: bool = False):
        """Build a rename hook suitable for graph rewriting passes."""

        def hook(old: Any, new: Any, user: Any) -> None:
            if getattr(user, "op", None) == "output":
                self.replace_all_uses(old.name, new)
            if replace_inputs and getattr(old, "op", None) == "placeholder":
                self.replace_all_uses(old.name, new)

        return hook

    def clone(self) -> "ExportGraphSignature":
        """Deep copy: specs and argument records are duplicated, not shared."""
        return dataclasses.replace(
            self,
            input_specs=[
                dataclasses.replace(spec, arg=dataclasses.replace(spec.arg))
                for spec in self.input_specs
            ],
            output_specs=[
                dataclasses.replace(spec, arg=dataclasses.replace(spec.arg))
                for spec in self.output_specs
            ],
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "ExportGraphSignature":
        return self.clone()

    def __str__(self) -> str:
        inputs = "\n".join(str(spec) for spec in self.input_specs)
        outputs = "\n".join(str(spec) for spec in self.output_specs)
        return f"\n# inputs\n{inputs}\n\n# outputs\n{outputs}\n"


@dataclasses.dataclass(frozen=True)
class GraphSignature:
    """Compact signature retained for callers that build signatures directly."""

    parameters: tuple[str, ...]
    buffers: tuple[str, ...]
    non_persistent_buffers: tuple[str, ...]
    user_inputs: tuple[str, ...]

    def to_export_signature(self) -> ExportGraphSignature:
        inputs: list[InputSpec] = [
            InputSpec(InputKind.PARAMETER, TensorArgument(name), name)
            for name in self.parameters
        ]
        inputs.extend(
            InputSpec(
                InputKind.BUFFER,
                TensorArgument(name),
                name,
                persistent=name not in self.non_persistent_buffers,
            )
            for name in self.buffers
        )
        inputs.extend(
            InputSpec(InputKind.USER_INPUT, TensorArgument(name), None)
            for name in self.user_inputs
        )
        return ExportGraphSignature(inputs, [])


def _immutable_dict(items) -> Mapping[str, str]:
    """A mapping that rejects addition, deletion, and item assignment."""

    from types import MappingProxyType

    return MappingProxyType(dict(items))


def _make_argument_spec(value: Any, token_names: Any = ()) -> ArgumentSpec:
    """Classify one flattened graph value for signature bookkeeping."""

    token_names = set(token_names)
    if isinstance(value, Node):
        if value.name in token_names:
            return TokenArgument(value.name)
        meta = getattr(value, "meta", {})
        val = meta.get("val")
        if val is not None and not hasattr(val, "shape"):
            if isinstance(val, (int, float, bool, str)) or val is None:
                return ConstantArgument(value.name, val)
            fqn = getattr(val, "constant_name", None) or f"{type(val).__module__}.{type(val).__qualname__}"
            return CustomObjArgument(value.name, fqn)
        return TensorArgument(value.name)
    if isinstance(value, (int, float, bool, str)) or value is None:
        return ConstantArgument("", value)
    raise TypeError(
        f"expected a graph node or a scalar constant, got {type(value).__name__}"
    )


def _convert_to_export_graph_signature(
    graph: Any,
    *,
    user_inputs: Any,
    inputs_to_parameters: Mapping[str, str],
    inputs_to_buffers: Mapping[str, str],
    user_outputs: Any,
    buffer_mutations: Mapping[str, str] | None = None,
    parameter_mutations: Mapping[str, str] | None = None,
    user_input_mutations: Mapping[str, str] | None = None,
    input_tokens: Any = (),
    output_tokens: Any = (),
    non_persistent_buffers: Any = (),
) -> ExportGraphSignature:
    """Build an :class:`ExportGraphSignature` from a flat graph and state maps.

    ``graph`` supplies the ordered placeholders and the output leaves;
    the mapping arguments classify each one.
    """

    buffer_mutations = buffer_mutations or {}
    parameter_mutations = parameter_mutations or {}
    user_input_mutations = user_input_mutations or {}
    input_tokens = list(input_tokens)
    output_tokens = list(output_tokens)

    input_specs: list[InputSpec] = []
    for node in graph.nodes:
        if node.op != "placeholder":
            continue
        if node.name in input_tokens:
            input_specs.append(InputSpec(InputKind.TOKEN, TokenArgument(node.name), None))
            continue
        if node.name in inputs_to_parameters:
            input_specs.append(
                InputSpec(InputKind.PARAMETER, TensorArgument(node.name), inputs_to_parameters[node.name])
            )
        elif node.name in inputs_to_buffers:
            target = inputs_to_buffers[node.name]
            input_specs.append(
                InputSpec(
                    InputKind.BUFFER,
                    TensorArgument(node.name),
                    target,
                    persistent=target not in set(non_persistent_buffers),
                )
            )
        else:
            input_specs.append(
                InputSpec(InputKind.USER_INPUT, _make_argument_spec(node, input_tokens), None)
            )

    output_node = graph.output_node
    leaves = []
    stack = [output_node.args[0]]
    while stack:
        item = stack.pop(0)
        if isinstance(item, Node):
            leaves.append(item)
        elif isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        else:
            leaves.append(item)

    output_specs: list[OutputSpec] = []
    mutation_slots = len(buffer_mutations) + len(parameter_mutations) + len(user_input_mutations) + len(output_tokens)
    for index, value in enumerate(leaves):
        spec = _make_argument_spec(value, output_tokens)
        if isinstance(spec, TokenArgument):
            output_specs.append(OutputSpec(OutputKind.TOKEN, spec, None))
            continue
        if index < mutation_slots and isinstance(spec, TensorArgument):
            if spec.name in buffer_mutations:
                output_specs.append(OutputSpec(OutputKind.BUFFER_MUTATION, spec, buffer_mutations[spec.name]))
                continue
            if spec.name in parameter_mutations:
                output_specs.append(OutputSpec(OutputKind.PARAMETER_MUTATION, spec, parameter_mutations[spec.name]))
                continue
            if spec.name in user_input_mutations:
                output_specs.append(OutputSpec(OutputKind.USER_INPUT_MUTATION, spec, user_input_mutations[spec.name]))
                continue
        output_specs.append(OutputSpec(OutputKind.USER_OUTPUT, spec, None))

    if user_outputs is not None:
        named = set(user_outputs)
        for spec in output_specs:
            if spec.kind is OutputKind.USER_OUTPUT and spec.arg.name in named:
                spec.kind = OutputKind.USER_OUTPUT
    return ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)
