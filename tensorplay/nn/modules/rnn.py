r"""Recurrent modules: RNN, LSTM, GRU and their single-step cells.

The recurrence kernels are Python ports of the CPU cell functors and the
layer/stack combinators of ``aten/src/ATen/native/RNN.cpp`` (``SimpleCell``,
``LSTMCell``, ``GRUCell``, ``FullLayer``, ``FullBidirectionalLayer``,
``PackedLayer``, ``ReversedPackedLayer``, ``apply_layer_stack``), including
the one-GEMM input pre-computation the CPU path performs per layer.  The
module classes mirror ``torch/nn/modules/rnn.py``.
"""

import math
import numbers
import warnings
import weakref

import tensorplay as tp
from tensorplay import Tensor
from .module import Module
from ..parameter import Parameter
from .. import init
from ..utils.rnn import PackedSequence, invert_permutation
from .. import functional as F

__all__ = [
    "RNNBase",
    "RNN",
    "LSTM",
    "GRU",
    "RNNCellBase",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
]


def _apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    return tensor.index_select(dim, permutation)


def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
    warnings.warn(
        "`apply_permutation` is deprecated, please use `tensor.index_select(dim, permutation)` instead",
        category=FutureWarning,
        stacklevel=2,
    )
    return _apply_permutation(tensor, permutation, dim)


# ---------------------------------------------------------------------------
# Cell kernels (aten/src/ATen/native/RNN.cpp CPU paths)
#
# Each cell has the aten signature
#   cell(input, hidden, params, pre_compute_input=False) -> hidden
# where params is the (w_ih, w_hh, b_ih, b_hh, w_hr) tuple of CellParams and
# pre_compute_input indicates that `input` already holds linear_ih(input).
# ---------------------------------------------------------------------------


def _rnn_tanh_cell(input, hidden, params, pre_compute_input=False):
    # SimpleCell<tanh_f> (aten RNN.cpp:736-746)
    w_ih, w_hh, b_ih, b_hh, _ = params
    out = F.linear(hidden, w_hh, b_hh)
    out.add_(input if pre_compute_input else F.linear(input, w_ih, b_ih))
    return out.tanh()


def _rnn_relu_cell(input, hidden, params, pre_compute_input=False):
    # SimpleCell<relu_f> (aten RNN.cpp:736-746)
    w_ih, w_hh, b_ih, b_hh, _ = params
    out = F.linear(hidden, w_hh, b_hh)
    out.add_(input if pre_compute_input else F.linear(input, w_ih, b_ih))
    return out.relu()


def _lstm_cell(input, hidden, params, pre_compute_input=False):
    # LSTMCell CPU path (aten RNN.cpp:772-782)
    w_ih, w_hh, b_ih, b_hh, w_hr = params
    hx, cx = hidden

    gates = F.linear(hx, w_hh, b_hh)
    gates.add_(input if pre_compute_input else F.linear(input, w_ih, b_ih))
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()
    cy = (forgetgate * cx).add_(ingate * cellgate)
    hy = outgate * cy.tanh()
    if w_hr is not None:
        hy = hy.matmul(w_hr.t())
    return hy, cy


def _gru_cell(input, hidden, params, pre_compute_input=False):
    # GRUCell CPU path (aten RNN.cpp:805-816)
    w_ih, w_hh, b_ih, b_hh, _ = params

    igates = input if pre_compute_input else F.linear(input, w_ih, b_ih)
    hgates = F.linear(hidden, w_hh, b_hh)
    ri, zi, ni = igates.chunk(3, 1)
    rh, zh, nh = hgates.chunk(3, 1)
    reset_gate = (rh + ri).sigmoid()
    input_gate = (zh + zi).sigmoid()
    new_gate = (ni + nh * reset_gate).tanh()
    return (hidden - new_gate) * input_gate + new_gate


# ---------------------------------------------------------------------------
# Generic helpers over "hidden" values (Tensor for RNN/GRU, (h, c) for LSTM);
# ports of hidden_slice / hidden_concat / hidden_as_output (aten RNN.cpp:594+)
# ---------------------------------------------------------------------------


def _hidden_slice(hidden, start, end):
    length = end - start
    if isinstance(hidden, tuple):
        return (hidden[0].narrow(0, start, length), hidden[1].narrow(0, start, length))
    return hidden.narrow(0, start, length)


def _hidden_cat(hiddens):
    if isinstance(hiddens[0], tuple):
        return (
            tp.cat([h[0] for h in hiddens], 0),
            tp.cat([h[1] for h in hiddens], 0),
        )
    return tp.cat(hiddens, 0)


def _hidden_as_output(hidden):
    if isinstance(hidden, tuple):
        return hidden[0]
    return hidden


# ---------------------------------------------------------------------------
# Layers: scan a cell over a sequence (aten RNN.cpp:848-1093)
# ---------------------------------------------------------------------------


def _layer_scan(step_inputs, hidden, params, cell, pre_compute_input):
    # The step-loop body of FullLayer (aten RNN.cpp:857-869)
    step_outputs = []
    for inp in step_inputs:
        hidden = cell(inp, hidden, params, pre_compute_input)
        step_outputs.append(_hidden_as_output(hidden))
    return step_outputs, hidden


def _full_layer(input, input_hidden, params, cell, is_cpu):
    # FullLayer::operator()(Tensor) (aten RNN.cpp:871-887).  On CPU the whole
    # input projection is computed as one GEMM before the scan
    # (pre_compute_input).
    if is_cpu:
        input_w = F.linear(input, params[0], params[2])
        step_inputs = input_w.unbind(0)
        step_outputs, final_hidden = _layer_scan(
            step_inputs, input_hidden, params, cell, True
        )
    else:
        step_inputs = input.unbind(0)
        step_outputs, final_hidden = _layer_scan(
            step_inputs, input_hidden, params, cell, False
        )
    if not step_outputs:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")
    return tp.stack(step_outputs, 0), final_hidden


def _full_bidirectional_layer(input, input_hidden, params_pair, cell, is_cpu):
    # FullBidirectionalLayer (aten RNN.cpp:902-948)
    fw_params, rev_params = params_pair
    fw_hidden, rev_hidden = input_hidden
    if is_cpu:
        input_w = F.linear(input, fw_params[0], fw_params[2])
        fw_outputs, fw_hidden_out = _layer_scan(
            input_w.unbind(0), fw_hidden, fw_params, cell, True
        )
        if not fw_outputs:
            raise RuntimeError("Expected sequence length to be larger than 0 in RNN")
        fw_output = tp.stack(fw_outputs, 0)
        input_w = F.linear(input, rev_params[0], rev_params[2])
        rev_step_inputs = list(input_w.unbind(0))[::-1]
        rev_outputs, rev_hidden_out = _layer_scan(
            rev_step_inputs, rev_hidden, rev_params, cell, True
        )
        rev_outputs.reverse()
        rev_output = tp.stack(rev_outputs, 0)
    else:
        step_inputs = input.unbind(0)
        fw_outputs, fw_hidden_out = _layer_scan(
            step_inputs, fw_hidden, fw_params, cell, False
        )
        if not fw_outputs:
            raise RuntimeError("Expected sequence length to be larger than 0 in RNN")
        fw_output = tp.stack(fw_outputs, 0)
        rev_step_inputs = list(step_inputs)[::-1]
        rev_outputs, rev_hidden_out = _layer_scan(
            rev_step_inputs, rev_hidden, rev_params, cell, False
        )
        rev_outputs.reverse()
        rev_output = tp.stack(rev_outputs, 0)
    output = tp.cat([fw_output, rev_output], fw_output.dim() - 1)
    return output, (fw_hidden_out, rev_hidden_out)


def _packed_layer(packed_input, input_hidden, params, cell, is_cpu):
    # PackedLayer (aten RNN.cpp:951-1009)
    data = packed_input.data
    batch_sizes = packed_input.batch_sizes.tolist()
    step_outputs = []
    hiddens = []
    input_offset = 0
    num_steps = len(batch_sizes)
    last_batch_size = batch_sizes[0]

    pre_compute_input = False
    if is_cpu:
        data = F.linear(data, params[0], params[2])
        pre_compute_input = True

    # Batch sizes is a sequence of decreasing lengths, which are offsets
    # into a 1D list of inputs. At every step we slice out batch_size elements,
    # and possibly account for the decrease in the batch size since the last
    # step, which requires us to slice the hidden state (since some sequences
    # are completed now). The sliced parts are also saved, because we will
    # need to return a tensor of final hidden state.
    hidden = input_hidden
    for i in range(num_steps):
        batch_size = batch_sizes[i]
        step_input = data.narrow(0, input_offset, batch_size)
        input_offset += batch_size
        dec = last_batch_size - batch_size
        if dec > 0:
            hiddens.append(
                _hidden_slice(hidden, last_batch_size - dec, last_batch_size)
            )
            hidden = _hidden_slice(hidden, 0, last_batch_size - dec)

        last_batch_size = batch_size
        hidden = cell(step_input, hidden, params, pre_compute_input)
        step_outputs.append(_hidden_as_output(hidden))
    hiddens.append(hidden)
    hiddens.reverse()

    output = PackedSequence(tp.cat(step_outputs, 0), packed_input.batch_sizes)
    return output, _hidden_cat(hiddens)


def _reversed_packed_layer(packed_input, input_hidden, params, cell, is_cpu):
    # ReversedPackedLayer (aten RNN.cpp:1012-1063)
    data = packed_input.data
    batch_sizes = packed_input.batch_sizes.tolist()
    step_outputs = []
    input_offset = data.size(0)
    num_steps = len(batch_sizes)
    last_batch_size = batch_sizes[num_steps - 1]

    pre_compute_input = False
    if is_cpu:
        data = F.linear(data, params[0], params[2])
        pre_compute_input = True

    # Here the situation is similar to above, except we start out with the
    # smallest batch size (and a small set of hidden states we actually use),
    # and progressively expand the hidden states, as we move backwards over
    # the 1D list of inputs.
    hidden = _hidden_slice(input_hidden, 0, batch_sizes[num_steps - 1])
    for i in range(num_steps - 1, -1, -1):
        batch_size = batch_sizes[i]
        inc = batch_size - last_batch_size
        if inc > 0:
            hidden = _hidden_cat(
                [hidden, _hidden_slice(input_hidden, last_batch_size, batch_size)]
            )
        step_input = data.narrow(0, input_offset - batch_size, batch_size)
        input_offset -= batch_size
        last_batch_size = batch_size
        hidden = cell(step_input, hidden, params, pre_compute_input)
        step_outputs.append(_hidden_as_output(hidden))
    step_outputs.reverse()
    output = PackedSequence(tp.cat(step_outputs, 0), packed_input.batch_sizes)
    return output, hidden


def _packed_bidirectional_layer(packed_input, input_hidden, params_pair, cell, is_cpu):
    # PackedBidirectionalLayer (aten RNN.cpp:1066-1093)
    fw_params, rev_params = params_pair
    fw_hidden, rev_hidden = input_hidden
    fw_result = _packed_layer(packed_input, fw_hidden, fw_params, cell, is_cpu)
    rev_result = _reversed_packed_layer(packed_input, rev_hidden, rev_params, cell, is_cpu)
    output = PackedSequence(
        tp.cat([fw_result[0].data, rev_result[0].data], -1), packed_input.batch_sizes
    )
    return output, (fw_result[1], rev_result[1])


def _rnn_dropout(input, p):
    # aten RNN.cpp:1103-1109
    if isinstance(input, PackedSequence):
        return PackedSequence(F.dropout(input.data, p, True), input.batch_sizes)
    return F.dropout(input, p, True)


def _apply_layer_stack(layer_fn, input, hiddens, params, num_layers, dropout_p, train):
    # apply_layer_stack (aten RNN.cpp:1111-1135)
    if num_layers != len(hiddens):
        raise RuntimeError("Expected more hidden states in stacked_rnn")
    if num_layers != len(params):
        raise RuntimeError("Expected more hidden states in stacked_rnn")

    layer_input = input
    final_hiddens = []
    for l in range(num_layers):
        layer_output, final_hidden = layer_fn(layer_input, hiddens[l], params[l])
        final_hiddens.append(final_hidden)
        layer_input = layer_output

        if dropout_p != 0 and train and l < num_layers - 1:
            layer_input = _rnn_dropout(layer_input, dropout_p)

    return layer_input, final_hiddens


def _gather_params(flat_weights, has_biases, has_projections=False):
    # Parses a flat list of parameter tensors into per-(layer, direction)
    # (w_ih, w_hh, b_ih, b_hh, w_hr) tuples; port of gather_params
    # (aten RNN.cpp:613-645).
    result = []
    if has_biases:
        if has_projections:
            if len(flat_weights) % 5 != 0:
                raise RuntimeError("got an incorrect number of RNN parameters")
            for i in range(0, len(flat_weights), 5):
                result.append(tuple(flat_weights[i : i + 5]))
        else:
            if len(flat_weights) % 4 != 0:
                raise RuntimeError("got an incorrect number of RNN parameters")
            for i in range(0, len(flat_weights), 4):
                result.append(
                    (flat_weights[i], flat_weights[i + 1], flat_weights[i + 2], flat_weights[i + 3], None)
                )
    else:
        if has_projections:
            if len(flat_weights) % 3 != 0:
                raise RuntimeError("got an incorrect number of RNN parameters")
            for i in range(0, len(flat_weights), 3):
                result.append((flat_weights[i], flat_weights[i + 1], None, None, flat_weights[i + 2]))
        else:
            if len(flat_weights) % 2 != 0:
                raise RuntimeError("got an incorrect number of RNN parameters")
            for i in range(0, len(flat_weights), 2):
                result.append((flat_weights[i], flat_weights[i + 1], None, None, None))
    return result


def _check_attributes(input, flat_weights, hiddens):
    # Port of check_attributes (aten RNN.cpp): all parameters, the input and
    # the hidden states must share one dtype and live on one device.
    for tensor in flat_weights:
        if tensor is None:
            continue
        if tensor.dtype != input.dtype or tensor.device != input.device:
            raise RuntimeError(
                "Input and parameter tensors have different dtype or device: "
                f"input {input.dtype} {input.device}, parameter {tensor.dtype} {tensor.device}"
            )
    for hidden in hiddens:
        if hidden.dtype != input.dtype or hidden.device != input.device:
            raise RuntimeError(
                "Input and hidden tensors have different dtype or device: "
                f"input {input.dtype} {input.device}, hidden {hidden.dtype} {hidden.device}"
            )


def _one_hidden_rnn(cell, input, hx, flat_weights, has_biases, num_layers,
                    dropout_p, train, bidirectional, batch_first):
    # Port of the ONE_HIDDEN_RNN macro body for RNN/GRU (aten RNN.cpp:1228-1289)
    _check_attributes(input, flat_weights, [hx])
    input = input.transpose(0, 1) if batch_first else input
    params = _gather_params(flat_weights, has_biases)
    is_cpu = not input.is_cuda
    if bidirectional:
        params = [(params[2 * i], params[2 * i + 1]) for i in range(num_layers)]
        hiddens = [(hx[2 * i], hx[2 * i + 1]) for i in range(num_layers)]

        def layer_fn(layer_input, hidden, param):
            return _full_bidirectional_layer(layer_input, hidden, param, cell, is_cpu)

        layer_input, final_hidden = _apply_layer_stack(
            layer_fn, input, hiddens, params, num_layers, dropout_p, train
        )
        # unpair_vec: [fw_l0, rev_l0, fw_l1, rev_l1, ...] (aten RNN.cpp:602-610)
        flat_hiddens = []
        for pair in final_hidden:
            flat_hiddens.extend(pair)
        output, hy = layer_input, tp.stack(flat_hiddens, 0)
    else:

        def layer_fn(layer_input, hidden, param):
            return _full_layer(layer_input, hidden, param, cell, is_cpu)

        output, final_hidden = _apply_layer_stack(
            layer_fn, input, list(hx.unbind(0)), params, num_layers, dropout_p, train
        )
        hy = tp.stack(final_hidden, 0)
    if batch_first:
        output = output.transpose(0, 1)
    return output, hy


def _one_hidden_rnn_packed(cell, data, batch_sizes, hx, flat_weights, has_biases,
                           num_layers, dropout_p, train, bidirectional):
    # Port of the packed ONE_HIDDEN_RNN overload (aten RNN.cpp:1291-1349)
    packed_input = PackedSequence(data, batch_sizes)
    _check_attributes(data, flat_weights, [hx])
    params = _gather_params(flat_weights, has_biases)
    is_cpu = not data.is_cuda
    if bidirectional:
        params = [(params[2 * i], params[2 * i + 1]) for i in range(num_layers)]
        hiddens = [(hx[2 * i], hx[2 * i + 1]) for i in range(num_layers)]

        def layer_fn(layer_input, hidden, param):
            return _packed_bidirectional_layer(layer_input, hidden, param, cell, is_cpu)

        layer_input, final_hidden = _apply_layer_stack(
            layer_fn, packed_input, hiddens, params, num_layers, dropout_p, train
        )
        flat_hiddens = []
        for pair in final_hidden:
            flat_hiddens.extend(pair)
        hy = tp.stack(flat_hiddens, 0)
    else:

        def layer_fn(layer_input, hidden, param):
            return _packed_layer(layer_input, hidden, param, cell, is_cpu)

        layer_input, final_hidden = _apply_layer_stack(
            layer_fn, packed_input, list(hx.unbind(0)), params, num_layers, dropout_p, train
        )
        hy = tp.stack(final_hidden, 0)
    return layer_input.data, hy


def _lstm_impl(cell, packed_input, hx, cx, params, num_layers, dropout_p, train,
               bidirectional, is_cpu):
    # Port of _lstm_impl (aten RNN.cpp:1168-1195): transpose the (hx, cx) pair
    # into per-layer pairs, run the stack, and stack hy/cy back.
    layer_hx = list(hx.unbind(0))
    layer_cx = list(cx.unbind(0))
    total_layers = len(layer_hx)
    hiddens = [(layer_hx[i], layer_cx[i]) for i in range(total_layers)]

    if bidirectional:
        param_pairs = [(params[2 * i], params[2 * i + 1]) for i in range(num_layers)]
        hidden_pairs = [(hiddens[2 * i], hiddens[2 * i + 1]) for i in range(num_layers)]
        if isinstance(packed_input, PackedSequence):

            def layer_fn(layer_input, hidden, param):
                return _packed_bidirectional_layer(layer_input, hidden, param, cell, is_cpu)

        else:

            def layer_fn(layer_input, hidden, param):
                return _full_bidirectional_layer(layer_input, hidden, param, cell, is_cpu)

        layer_input, final_hidden = _apply_layer_stack(
            layer_fn, packed_input, hidden_pairs, param_pairs, num_layers, dropout_p, train
        )
        # unpair the per-layer (fw, rev) pairs into the layer-major order the
        # module API uses
        hiddens = []
        for pair in final_hidden:
            hiddens.extend(pair)
    else:
        if isinstance(packed_input, PackedSequence):

            def layer_fn(layer_input, hidden, param):
                return _packed_layer(layer_input, hidden, param, cell, is_cpu)

        else:

            def layer_fn(layer_input, hidden, param):
                return _full_layer(layer_input, hidden, param, cell, is_cpu)

        layer_input, hiddens = _apply_layer_stack(
            layer_fn, packed_input, hiddens, params, num_layers, dropout_p, train
        )

    hy = [hidden[0] for hidden in hiddens]
    cy = [hidden[1] for hidden in hiddens]
    return layer_input, tp.stack(hy, 0), tp.stack(cy, 0)


def _lstm(input, hx, flat_weights, has_biases, num_layers, dropout_p, train,
          bidirectional, batch_first):
    # Port of at::lstm for the non-packed input (aten RNN.cpp:1464-1526).
    if len(hx) != 2:
        raise RuntimeError("lstm expects two hidden states")
    _check_attributes(input, flat_weights, hx)
    input = input.transpose(0, 1) if batch_first else input
    # if cells are of different size, that means projections are used
    has_projections = hx[0].size(2) != hx[1].size(2)
    params = _gather_params(flat_weights, has_biases, has_projections)
    is_cpu = not input.is_cuda
    output, hy, cy = _lstm_impl(
        _lstm_cell, input, hx[0], hx[1], params, num_layers, dropout_p, train,
        bidirectional, is_cpu,
    )
    if batch_first:
        output = output.transpose(0, 1)
    return output, hy, cy


def _lstm_packed(data, batch_sizes, hx, flat_weights, has_biases, num_layers,
                 dropout_p, train, bidirectional):
    # Port of at::lstm for packed input (aten RNN.cpp:1528+)
    if len(hx) != 2:
        raise RuntimeError("lstm expects two hidden states")
    packed_input = PackedSequence(data, batch_sizes)
    _check_attributes(data, flat_weights, hx)
    # if cells are of different size, that means projections are used
    has_projections = hx[0].size(2) != hx[1].size(2)
    params = _gather_params(flat_weights, has_biases, has_projections)
    is_cpu = not data.is_cuda
    output, hy, cy = _lstm_impl(
        _lstm_cell, packed_input, hx[0], hx[1], params, num_layers, dropout_p,
        train, bidirectional, is_cpu,
    )
    return output.data, hy, cy


def _any_autocast_enabled() -> bool:
    # Stand-in for torch._C._is_any_autocast_enabled(): autocast can be
    # enabled per device type.
    return tp.is_autocast_enabled("cpu") or tp.is_autocast_enabled("cuda")


class RNNBase(Module):
    r"""Base class for RNN modules (RNN, LSTM, GRU).

    Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization
    and utility methods for parameter storage management.

    .. note::
        The forward method is not implemented by the RNNBase class.

    .. note::
        LSTM and GRU classes override some methods implemented by RNNBase.
    """

    __constants__ = [
        "mode",
        "input_size",
        "hidden_size",
        "num_layers",
        "bias",
        "batch_first",
        "dropout",
        "bidirectional",
        "proj_size",
    ]
    __jit_unused_properties__ = ["all_weights"]

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self._flat_weight_refs: list = []
        num_directions = 2 if bidirectional else 1

        if (
            not isinstance(dropout, numbers.Number)
            or not 0 <= dropout <= 1
            or isinstance(dropout, bool)
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] "
                "representing the probability of an element being "
                "zeroed"
            )
        if dropout > 0 and num_layers == 1:
            warnings.warn(
                "dropout option adds dropout after all but last "
                "recurrent layer, so non-zero dropout expects "
                f"num_layers greater than 1, but got dropout={dropout} and "
                f"num_layers={num_layers}",
                stacklevel=2,
            )

        if not isinstance(bias, bool):
            raise TypeError(f"bias should be of type bool, got: {type(bias).__name__}")
        if not isinstance(batch_first, bool):
            raise TypeError(
                f"batch_first should be of type bool, got: {type(batch_first).__name__}"
            )
        if not isinstance(input_size, int):
            raise TypeError(
                f"input_size should be of type int, got: {type(input_size).__name__}"
            )
        if input_size <= 0:
            raise ValueError("input_size must be greater than zero")
        if not isinstance(hidden_size, int):
            raise TypeError(
                f"hidden_size should be of type int, got: {type(hidden_size).__name__}"
            )
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        if proj_size < 0:
            raise ValueError(
                "proj_size should be a positive integer or zero to disable projections"
            )
        if proj_size >= hidden_size:
            raise ValueError("proj_size has to be smaller than hidden_size")

        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        elif mode == "RNN_TANH":
            gate_size = hidden_size
        elif mode == "RNN_RELU":
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self._flat_weights_names = []
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = (
                    input_size if layer == 0 else real_hidden_size * num_directions
                )

                w_ih = Parameter(
                    tp.empty((gate_size, layer_input_size), **factory_kwargs)
                )
                w_hh = Parameter(
                    tp.empty((gate_size, real_hidden_size), **factory_kwargs)
                )
                b_ih = Parameter(tp.empty(gate_size, **factory_kwargs))
                # Second bias vector included for CuDNN compatibility. Only one
                # bias vector is needed in standard definition.
                b_hh = Parameter(tp.empty(gate_size, **factory_kwargs))
                layer_params = ()
                if self.proj_size == 0:
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh)
                    else:
                        layer_params = (w_ih, w_hh)
                else:
                    w_hr = Parameter(
                        tp.empty((proj_size, hidden_size), **factory_kwargs)
                    )
                    if bias:
                        layer_params = (w_ih, w_hh, b_ih, b_hh, w_hr)
                    else:
                        layer_params = (w_ih, w_hh, w_hr)

                suffix = "_reverse" if direction == 1 else ""
                param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                if self.proj_size > 0:
                    param_names += ["weight_hr_l{}{}"]
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._flat_weights_names.extend(param_names)
                self._all_weights.append(param_names)

        self._init_flat_weights()

        self.reset_parameters()

    def _init_flat_weights(self) -> None:
        self._flat_weights = [
            getattr(self, wn) if hasattr(self, wn) else None
            for wn in self._flat_weights_names
        ]
        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]
        self.flatten_parameters()

    def __setattr__(self, attr, value) -> None:
        if hasattr(self, "_flat_weights_names") and attr in self._flat_weights_names:
            # keep self._flat_weights up to date if you do self.weight = ...
            idx = self._flat_weights_names.index(attr)
            self._flat_weights[idx] = value
        super().__setattr__(attr, value)

    def flatten_parameters(self) -> None:
        r"""Reset parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.

        TensorPlay has no cuDNN-backed fused RNN, so this is always a no-op
        here; the method is kept for torch API compatibility.
        """
        # Short-circuits if _flat_weights is only partially instantiated
        if len(self._flat_weights) != len(self._flat_weights_names):
            return

        for w in self._flat_weights:
            if not isinstance(w, Tensor):
                return
        return

    def _apply(self, fn, recurse=True):
        self._flat_weight_refs = []
        ret = super()._apply(fn, recurse)

        # Resets _flat_weights
        # Note: be v. careful before removing this, as 3rd party device types
        # likely rely on this behavior to properly .to() modules like LSTM.
        self._init_flat_weights()

        return ret

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def check_input(self, input: Tensor, batch_sizes) -> None:
        if (
            input.dtype != self._flat_weights[0].dtype
            and not _any_autocast_enabled()
        ):
            raise ValueError(
                f"RNN input dtype ({input.dtype}) does not match weight dtype ({self._flat_weights[0].dtype}). "
                f"Convert input: input.to({self._flat_weights[0].dtype}), or convert model: model.to({input.dtype})"
            )
        expected_input_dim = 2 if batch_sizes is not None else 3
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                f"input must have {expected_input_dim} dimensions, got {input.dim()}"
            )
        if self.input_size != input.size(-1):
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {input.size(-1)}"
            )

    def get_expected_hidden_size(
        self, input: Tensor, batch_sizes
    ) -> tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0].item())
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        if self.proj_size > 0:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.proj_size,
            )
        else:
            expected_hidden_size = (
                self.num_layers * num_directions,
                mini_batch,
                self.hidden_size,
            )
        return expected_hidden_size

    def get_expected_cell_size(
        self, input: Tensor, batch_sizes
    ) -> tuple[int, int, int]:
        if batch_sizes is not None:
            mini_batch = int(batch_sizes[0].item())
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)
        num_directions = 2 if self.bidirectional else 1
        expected_cell_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.hidden_size,
        )
        return expected_cell_size

    def check_hidden_size(
        self,
        hx: Tensor,
        expected_hidden_size: tuple[int, int, int],
        msg: str = "Expected hidden size {}, got {}",
    ) -> None:
        if tuple(hx.shape) != tuple(expected_hidden_size):
            raise RuntimeError(msg.format(expected_hidden_size, list(hx.shape)))

    def _weights_have_changed(self):
        # Returns True if the weight tensors have changed since the last forward pass.
        # This is the case when used with functional_call(), for example.
        weights_changed = False
        for ref, name in zip(self._flat_weight_refs, self._flat_weights_names):
            weight = getattr(self, name) if hasattr(self, name) else None
            if weight is not None and ref is not None and ref() is not weight:
                weights_changed = True
                break
        return weights_changed

    def check_forward_args(
        self, input: Tensor, hidden: Tensor, batch_sizes
    ) -> None:
        self.check_input(input, batch_sizes)
        expected_hidden_size = self.get_expected_hidden_size(input, batch_sizes)

        self.check_hidden_size(hidden, expected_hidden_size)

    def permute_hidden(self, hx: Tensor, permutation):
        if permutation is None:
            return hx
        return _apply_permutation(hx, permutation)

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if self.proj_size != 0:
            s += ", proj_size={proj_size}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def _update_flat_weights(self) -> None:
        if self._weights_have_changed():
            self._init_flat_weights()

    def __getstate__(self):
        # If weights have been changed, update the _flat_weights in __getstate__ here.
        self._update_flat_weights()
        # Don't serialize the weight references.
        state = self.__dict__.copy()
        del state["_flat_weight_refs"]
        return state

    def __setstate__(self, d):
        super().__setstate__(d)
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        # In PyTorch 1.8 we added a proj_size member variable to LSTM.
        # LSTMs that were serialized via save(module) before PyTorch 1.8
        # don't have it, so to preserve compatibility we set proj_size here.
        if "proj_size" not in d:
            self.proj_size = 0

        if not isinstance(self._all_weights[0][0], str):
            num_layers = self.num_layers
            num_directions = 2 if self.bidirectional else 1
            self._flat_weights_names = []
            self._all_weights = []
            for layer in range(num_layers):
                for direction in range(num_directions):
                    suffix = "_reverse" if direction == 1 else ""
                    weights = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "bias_ih_l{}{}",
                        "bias_hh_l{}{}",
                        "weight_hr_l{}{}",
                    ]
                    weights = [x.format(layer, suffix) for x in weights]
                    if self.bias:
                        if self.proj_size > 0:
                            self._all_weights += [weights]
                            self._flat_weights_names.extend(weights)
                        else:
                            self._all_weights += [weights[:4]]
                            self._flat_weights_names.extend(weights[:4])
                    else:
                        if self.proj_size > 0:
                            self._all_weights += [weights[:2]] + [weights[-1:]]
                            self._flat_weights_names.extend(
                                weights[:2] + [weights[-1:]]
                            )
                        else:
                            self._all_weights += [weights[:2]]
                            self._flat_weights_names.extend(weights[:2])
            self._flat_weights = [
                getattr(self, wn) if hasattr(self, wn) else None
                for wn in self._flat_weights_names
            ]

        self._flat_weight_refs = [
            weakref.ref(w) if w is not None else None for w in self._flat_weights
        ]

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]


class RNN(RNNBase):
    r"""__init__(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True,
    batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)

    Apply a multi-layer Elman RNN with :math:`\tanh` or :math:`\text{ReLU}`
    non-linearity to an input sequence. For each element in the input sequence,
    each layer computes the following function:

    .. math::
        h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, hx
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`tensorplay.nn.utils.rnn.pack_padded_sequence` or
          :func:`tensorplay.nn.utils.rnn.pack_sequence` for details.
        * **hx**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for the input sequence batch. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`~tensorplay.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    Examples::

        >>> rnn = tp.nn.RNN(10, 20, 2)
        >>> input = tp.randn(5, 3, 10)
        >>> h0 = tp.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        if len(args) > 3:
            self.nonlinearity = args[3]
            args = args[:3] + args[4:]
        else:
            self.nonlinearity = kwargs.pop("nonlinearity", "tanh")
        if self.nonlinearity == "tanh":
            mode = "RNN_TANH"
        elif self.nonlinearity == "relu":
            mode = "RNN_RELU"
        else:
            raise ValueError(
                f"Unknown nonlinearity '{self.nonlinearity}'. Select from 'tanh' or 'relu'."
            )
        super().__init__(mode, *args, **kwargs)

    def forward(self, input, hx=None):
        """
        Runs the forward pass.
        """
        self._update_flat_weights()

        num_directions = 2 if self.bidirectional else 1
        orig_input = input

        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0].item())
            if hx is None:
                hx = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            batch_sizes = None
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"RNN: Expected input to be 2D or 3D, got {input.dim()}D tensor instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                hx = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        if hx is None:
            raise AssertionError("hx must not be None")
        self.check_forward_args(input, hx, batch_sizes)
        if self.mode != "RNN_TANH" and self.mode != "RNN_RELU":
            raise AssertionError(f"mode must be RNN_TANH or RNN_RELU, got {self.mode}")
        cell = _rnn_tanh_cell if self.mode == "RNN_TANH" else _rnn_relu_cell
        if batch_sizes is None:
            result = _one_hidden_rnn(
                cell,
                input,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _one_hidden_rnn_packed(
                cell,
                input,
                batch_sizes,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )

        output = result[0]
        hidden = result[1]

        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)

        if not is_batched:
            output = output.squeeze(batch_dim)
            hidden = hidden.squeeze(1)

        return output, self.permute_hidden(hidden, unsorted_indices)


class LSTM(RNNBase):
    r"""__init__(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
    dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)

    Apply a multi-layer long short-term memory (LSTM) RNN to an input
    sequence. For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
            c_t = f_t \odot c_{(t-1)} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{(t-1)}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell,
    and output gates, respectively. :math:`\sigma` is the sigmoid function,
    and :math:`\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l`-th layer
    (:math`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\varphi^{(l-1)}_t` where each :math:`\varphi^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of such LSTM will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`tensorplay.nn.utils.rnn.pack_padded_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{cell} ={} & \text{hidden\_size} \\
                H_{out} ={} & \text{proj\_size if } \text{proj\_size}>0 \text{ otherwise hidden\_size} \\
            \end{aligned}

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`~tensorplay.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence.
        * **c_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{cell})` containing the
          final cell state for each element in the sequence.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer
            :math:`(W_{ii}|W_{if}|W_{ig}|W_{io})`, of shape
            `(4*hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(4*hidden_size, num_directions * hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, num_directions * proj_size)`.
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer
            :math:`(W_{hi}|W_{hf}|W_{hg}|W_{ho})`, of shape
            `(4*hidden_size, hidden_size)`. If ``proj_size > 0`` was specified,
            the shape will be `(4*hidden_size, proj_size)`.
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer
            :math:`(b_{ii}|b_{if}|b_{ig}|b_{io})`, of shape `(4*hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer
            :math:`(b_{hi}|b_{hf}|b_{hg}|b_{ho})`, of shape `(4*hidden_size)`
        weight_hr_l[k]: the learnable projection weights of the k-th layer
            :math:`W_{hr}`, of shape `(proj_size, hidden_size)`. Only present
            when ``proj_size > 0`` was specified.

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. note::
        For unbatched input, the ``h_n`` and ``c_n`` are tensors of shape `(num_layers * num_directions, H_out)`
        and `(num_layers * num_directions, H_cell)` respectively.

    Examples::

        >>> rnn = tp.nn.LSTM(10, 20, 2)
        >>> input = tp.randn(5, 3, 10)
        >>> h0 = tp.randn(2, 3, 20)
        >>> c0 = tp.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """

    def __init__(self, *args, **kwargs):
        super().__init__("LSTM", *args, **kwargs)

    def forward(self, input, hx=None):
        """
        Runs the forward pass.
        """
        self._update_flat_weights()

        orig_input = input
        batch_sizes = None
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0].item())
            if hx is None:
                h_zeros = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    real_hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                c_zeros = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                hx = (h_zeros, c_zeros)
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"LSTM: Expected input to be 2D or 3D, got {input.dim()}D instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                h_zeros = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    real_hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                c_zeros = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
                hx = (h_zeros, c_zeros)
                self.check_forward_args(input, hx, batch_sizes)
            else:
                if is_batched:
                    if hx[0].dim() != 3 or hx[1].dim() != 3:
                        msg = (
                            "For batched 3-D input, hx and cx should "
                            f"also be 3-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                else:
                    if hx[0].dim() != 2 or hx[1].dim() != 2:
                        msg = (
                            "For unbatched 2-D input, hx and cx should "
                            f"also be 2-D but got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                        )
                        raise RuntimeError(msg)
                    hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                self.check_forward_args(input, hx, batch_sizes)
                hx = self.permute_hidden(hx, sorted_indices)

        if batch_sizes is None:
            result = _lstm(
                input,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _lstm_packed(
                input,
                batch_sizes,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1:]
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
            return output, self.permute_hidden(hidden, unsorted_indices)

    def check_forward_args(
        self,
        input: Tensor,
        hidden: tuple,
        batch_sizes,
    ) -> None:
        self.check_input(input, batch_sizes)
        self.check_hidden_size(
            hidden[0],
            self.get_expected_hidden_size(input, batch_sizes),
            "Expected hidden[0] size {}, got {}",
        )
        self.check_hidden_size(
            hidden[1],
            self.get_expected_cell_size(input, batch_sizes),
            "Expected hidden[1] size {}, got {}",
        )

    def permute_hidden(
        self, hx, permutation
    ):
        if permutation is None:
            return hx
        return _apply_permutation(hx[0], permutation), _apply_permutation(
            hx[1], permutation
        )


class GRU(RNNBase):
    r"""__init__(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
    dropout=0.0, bidirectional=False, device=None, dtype=None)

    Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \begin{array}{ll} \\
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)} + b_{hn})) \\
            h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)} \\
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer at time `t-1`
    or the initial hidden state at time 0, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`tensorplay.nn.utils.rnn.pack_padded_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the initial hidden
          state for each element in the input sequence. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`. If a
          :class:`~tensorplay.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \text{num\_layers}, H_{out})` for unbatched input or
          :math:`(D * \text{num\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer
            :math:`(W_{ir}|W_{iz}|W_{in})`, of shape
            `(3*hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer
            :math:`(W_{hr}|W_{hz}|W_{hn})`, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer
            :math:`(b_{ir}|b_{iz}|b_{in})`, of shape `(3*hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer
            :math:`(b_{hr}|b_{hz}|b_{hn})`, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    .. note::
        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    Examples::

        >>> rnn = tp.nn.GRU(10, 20, 2)
        >>> input = tp.randn(5, 3, 10)
        >>> h0 = tp.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(self, *args, **kwargs):
        if "proj_size" in kwargs:
            raise ValueError(
                "proj_size argument is only supported for LSTM, not RNN or GRU"
            )
        super().__init__("GRU", *args, **kwargs)

    def forward(self, input, hx=None):
        """
        Runs the forward pass.
        """
        self._update_flat_weights()

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        batch_sizes = None
        num_directions = 2 if self.bidirectional else 1
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = int(batch_sizes[0].item())
            if hx is None:
                hx = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)
        else:
            if input.dim() not in (2, 3):
                raise ValueError(
                    f"GRU: Expected input to be 2D or 3D, got {input.dim()}D instead"
                )
            is_batched = input.dim() == 3
            batch_dim = 0 if self.batch_first else 1
            if not is_batched:
                input = input.unsqueeze(batch_dim)
                if hx is not None:
                    if hx.dim() != 2:
                        raise RuntimeError(
                            f"For unbatched 2-D input, hx should also be 2-D but got {hx.dim()}-D tensor"
                        )
                    hx = hx.unsqueeze(1)
            else:
                if hx is not None and hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None
            if hx is None:
                hx = tp.zeros(
                    self.num_layers * num_directions,
                    max_batch_size,
                    self.hidden_size,
                    dtype=input.dtype,
                    device=input.device,
                )
            else:
                # Each batch of the hidden state should match the input sequence that
                # the user believes he/she is passing in.
                hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if self.mode != "GRU":
            raise AssertionError(f"mode must be GRU, got {self.mode}")

        if batch_sizes is None:
            result = _one_hidden_rnn(
                _gru_cell,
                input,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        else:
            result = _one_hidden_rnn_packed(
                _gru_cell,
                input,
                batch_sizes,
                hx,
                self._flat_weights,
                self.bias,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
            )
        output = result[0]
        hidden = result[1]
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(
                output,
                batch_sizes,
                sorted_indices,
                unsorted_indices,
            )
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            if not is_batched:
                output = output.squeeze(batch_dim)
                hidden = hidden.squeeze(1)
            return output, self.permute_hidden(hidden, unsorted_indices)


class RNNCellBase(Module):
    __constants__ = ["input_size", "hidden_size", "bias"]

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(
            tp.empty((num_chunks * hidden_size, input_size), **factory_kwargs)
        )
        self.weight_hh = Parameter(
            tp.empty((num_chunks * hidden_size, hidden_size), **factory_kwargs)
        )
        if bias:
            self.bias_ih = Parameter(
                tp.empty(num_chunks * hidden_size, **factory_kwargs)
            )
            self.bias_hh = Parameter(
                tp.empty(num_chunks * hidden_size, **factory_kwargs)
            )
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        s = "{input_size}, {hidden_size}"
        if "bias" in self.__dict__ and self.bias is not True:
            s += ", bias={bias}"
        if "nonlinearity" in self.__dict__ and self.nonlinearity != "tanh":
            s += ", nonlinearity={nonlinearity}"
        return s.format(**self.__dict__)

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)


class RNNCell(RNNCellBase):
    r"""An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs: input, hidden
        - **input**: tensor containing input features
        - **hidden**: tensor containing the initial hidden state
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = tp.nn.RNNCell(10, 20)
        >>> input = tp.randn(6, 3, 10)
        >>> hx = tp.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "nonlinearity"]
    nonlinearity: str

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=1, **factory_kwargs)
        self.nonlinearity = nonlinearity

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"RNNCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = tp.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        params = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, None)
        if self.nonlinearity == "tanh":
            ret = _rnn_tanh_cell(input, hx, params)
        elif self.nonlinearity == "relu":
            ret = _rnn_relu_cell(input, hx, params)
        else:
            raise RuntimeError(f"Unknown nonlinearity: {self.nonlinearity}")

        if not is_batched:
            ret = ret.squeeze(0)

        return ret


class LSTMCell(RNNCellBase):
    r"""A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\
        o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f \odot c + i \odot g \\
        h' = o \odot \tanh(c') \\
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = tp.nn.LSTMCell(10, 20)  # (input_size, hidden_size)
        >>> input = tp.randn(2, 3, 10)  # (time_steps, batch, input_size)
        >>> hx = tp.randn(3, 20)  # (batch, hidden_size)
        >>> cx = tp.randn(3, 20)
        >>> output = []
        >>> for i in range(input.size()[0]):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
        >>> output = tp.stack(output, dim=0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=4, **factory_kwargs)

    def forward(self, input: Tensor, hx=None):
        if input.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None:
            for idx, value in enumerate(hx):
                if value.dim() not in (1, 2):
                    raise ValueError(
                        f"LSTMCell: Expected hx[{idx}] to be 1D or 2D, got {value.dim()}D instead"
                    )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = tp.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        params = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, None)
        ret = _lstm_cell(input, hx, params)

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret


class GRUCell(RNNCellBase):
    r"""A gated recurrent unit (GRU) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r \odot (W_{hn} h + b_{hn})) \\
        h' = (1 - z) \odot n + z \odot h
        \end{array}

    where :math:`\sigma` is the sigmoid function, and :math:`\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, hidden
        - **input** : tensor containing input features
        - **hidden** : tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** : tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        where :math:`k = \frac{1}{\text{hidden\_size}}`

    Examples::

        >>> rnn = tp.nn.GRUCell(10, 20)
        >>> input = tp.randn(6, 3, 10)
        >>> hx = tp.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(input_size, hidden_size, bias, num_chunks=3, **factory_kwargs)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        if input.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected input to be 1D or 2D, got {input.dim()}D instead"
            )
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(
                f"GRUCell: Expected hidden to be 1D or 2D, got {hx.dim()}D instead"
            )
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = tp.zeros(
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        params = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh, None)
        ret = _gru_cell(input, hx, params)

        if not is_batched:
            ret = ret.squeeze(0)

        return ret
