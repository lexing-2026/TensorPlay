"""Native recurrent-cell rewrites used while capturing recurrent modules."""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Sequence
from typing import Any, Callable

__all__ = [
    "_clone_tensors_that_alias",
    "gru_while_loop_impl",
    "lstm_while_loop_impl",
    "one_layer_while_loop_gru",
    "one_layer_while_loop_lstm",
    "register_gru_while_loop_decomposition",
    "register_lstm_while_loop_decomposition",
]

_RNN_DECOMPOSITIONS: dict[str, Callable[..., Any]] = {}


def _clone_tensors_that_alias(values: Sequence[Any]) -> tuple[Any, ...]:
    result: list[Any] = []
    previous: list[Any] = []
    for value in values:
        current = value
        for old in previous:
            if current is old:
                current = current.clone() if hasattr(current, "clone") else current
                break
        result.append(current)
        if current is not None:
            previous.append(current)
    return tuple(result)


def _linear(value: Any, weight: Any, bias: Any = None) -> Any:
    result = value @ weight.transpose(-1, -2)
    return result + bias if bias is not None else result


def _split(value: Any, count: int) -> tuple[Any, ...]:
    if hasattr(value, "chunk"):
        return tuple(value.chunk(count, dim=-1))
    raise TypeError("recurrent cell requires a tensor with chunk support")


def _sigmoid(value: Any) -> Any:
    import tensorplay

    return tensorplay.sigmoid(value)


def _tanh(value: Any) -> Any:
    import tensorplay

    return tensorplay.tanh(value)


def one_layer_while_loop_lstm(
    inp: Any,
    hidden: tuple[Any, Any],
    params: Sequence[Any],
    has_biases: bool,
    reverse: bool = False,
) -> tuple[Any, tuple[Any, Any]]:
    if len(hidden) != 2:
        raise ValueError("lstm hidden state must contain cell and hidden values")
    weight_ih, weight_hh = params[:2]
    bias_ih = params[2] if has_biases else None
    bias_hh = params[3] if has_biases else None
    sequence = list(inp.unbind(0))
    if reverse:
        sequence.reverse()
    hidden_state, cell_state = hidden
    outputs: list[Any] = []
    for item in sequence:
        gates = _linear(item, weight_ih, bias_ih) + _linear(hidden_state, weight_hh, bias_hh)
        input_gate, forget_gate, cell_gate, output_gate = _split(gates, 4)
        input_gate = _sigmoid(input_gate)
        forget_gate = _sigmoid(forget_gate)
        cell_gate = _tanh(cell_gate)
        output_gate = _sigmoid(output_gate)
        cell_state = forget_gate * cell_state + input_gate * cell_gate
        hidden_state = output_gate * _tanh(cell_state)
        outputs.append(hidden_state)
    if reverse:
        outputs.reverse()
    return tensor_stack(outputs, 0), (hidden_state, cell_state)


def one_layer_while_loop_gru(
    inp: Any,
    hidden: Any,
    params: Sequence[Any],
    has_biases: bool,
    reverse: bool = False,
) -> tuple[Any, Any]:
    weight_ih, weight_hh = params[:2]
    bias_ih = params[2] if has_biases else None
    bias_hh = params[3] if has_biases else None
    sequence = list(inp.unbind(0))
    if reverse:
        sequence.reverse()
    state = hidden
    outputs: list[Any] = []
    for item in sequence:
        input_gates = _linear(item, weight_ih, bias_ih)
        hidden_gates = _linear(state, weight_hh, bias_hh)
        reset_i, update_i, new_i = _split(input_gates, 3)
        reset_h, update_h, new_h = _split(hidden_gates, 3)
        reset = _sigmoid(reset_i + reset_h)
        update = _sigmoid(update_i + update_h)
        new = _tanh(new_i + reset * new_h)
        state = new + update * (state - new)
        outputs.append(state)
    if reverse:
        outputs.reverse()
    return tensor_stack(outputs, 0), state


def tensor_stack(values: Sequence[Any], dim: int) -> Any:
    if not values:
        raise ValueError("cannot stack an empty recurrent sequence")
    import tensorplay

    return tensorplay.stack(tuple(values), dim=dim)


def _layer_parameters(params: Sequence[Any], layer: int, direction: int, has_biases: bool) -> Sequence[Any]:
    per_direction = 4 if has_biases else 2
    index = (layer * 2 + direction) * per_direction
    return params[index : index + per_direction]


def lstm_while_loop_impl(
    input: Any,
    hx: tuple[Any, Any],
    params: Sequence[Any],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple[Any, Any, Any]:
    if dropout and train:
        raise NotImplementedError("dropout inside a captured recurrent rewrite is unavailable")
    if batch_first:
        input = input.transpose(0, 1)
    directions = 2 if bidirectional else 1
    sequence = input
    final_h: list[Any] = []
    final_c: list[Any] = []
    hidden_values = list(hx[0].unbind(0))
    cell_values = list(hx[1].unbind(0))
    for layer in range(num_layers):
        direction_outputs: list[Any] = []
        next_h: list[Any] = []
        next_c: list[Any] = []
        for direction in range(directions):
            output, (state_h, state_c) = one_layer_while_loop_lstm(
                sequence,
                (hidden_values[layer * directions + direction], cell_values[layer * directions + direction]),
                _layer_parameters(params, layer, direction, has_biases),
                has_biases,
                reverse=direction == 1,
            )
            direction_outputs.append(output)
            next_h.append(state_h)
            next_c.append(state_c)
        sequence = tensor_stack(direction_outputs, -1).flatten(-2, -1) if directions == 2 else direction_outputs[0]
        final_h.extend(next_h)
        final_c.extend(next_c)
        hidden_values = next_h
        cell_values = next_c
    output = sequence.transpose(0, 1) if batch_first else sequence
    return output, tensor_stack(final_h, 0), tensor_stack(final_c, 0)


def gru_while_loop_impl(
    input: Any,
    hx: Any,
    params: Sequence[Any],
    has_biases: bool,
    num_layers: int,
    dropout: float,
    train: bool,
    bidirectional: bool,
    batch_first: bool,
) -> tuple[Any, Any]:
    if dropout and train:
        raise NotImplementedError("dropout inside a captured recurrent rewrite is unavailable")
    if batch_first:
        input = input.transpose(0, 1)
    directions = 2 if bidirectional else 1
    sequence = input
    hidden_values = list(hx.unbind(0))
    final_h: list[Any] = []
    for layer in range(num_layers):
        outputs: list[Any] = []
        next_hidden: list[Any] = []
        for direction in range(directions):
            output, state = one_layer_while_loop_gru(
                sequence,
                hidden_values[layer * directions + direction],
                _layer_parameters(params, layer, direction, has_biases),
                has_biases,
                reverse=direction == 1,
            )
            outputs.append(output)
            next_hidden.append(state)
        sequence = tensor_stack(outputs, -1).flatten(-2, -1) if directions == 2 else outputs[0]
        final_h.extend(next_hidden)
        hidden_values = next_hidden
    output = sequence.transpose(0, 1) if batch_first else sequence
    return output, tensor_stack(final_h, 0)


@contextlib.contextmanager
def _register_rnn_while_loop_decomposition(
    name: str, implementation: Callable[..., Any]
) -> Generator[None, None, None]:
    previous = _RNN_DECOMPOSITIONS.get(name)
    _RNN_DECOMPOSITIONS[name] = implementation
    try:
        yield
    finally:
        if previous is None:
            _RNN_DECOMPOSITIONS.pop(name, None)
        else:
            _RNN_DECOMPOSITIONS[name] = previous


@contextlib.contextmanager
def register_lstm_while_loop_decomposition() -> Generator[None, None, None]:
    with _register_rnn_while_loop_decomposition("lstm", lstm_while_loop_impl):
        yield


@contextlib.contextmanager
def register_gru_while_loop_decomposition() -> Generator[None, None, None]:
    with _register_rnn_while_loop_decomposition("gru", gru_while_loop_impl):
        yield
