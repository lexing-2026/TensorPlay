// Composite kernels: lstm_cell / rnn_relu_cell / rnn_tanh_cell.
// LSTM decomposes into the standard gate formulas (gate order i, f, g, o).

#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <optional>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace composite {

namespace ops = tensorplay::tpx::ops;

std::tuple<Tensor, Tensor> lstm_cell_native(
        const Tensor& input, const Tensor& hidden, const Tensor& cx,
        const Tensor& w_ih, const Tensor& w_hh,
        const std::optional<Tensor>& b_ih,
        const std::optional<Tensor>& b_hh) {
    const Tensor gates = ops::add(ops::linear(input, w_ih, b_ih),
                                  ops::linear(hidden, w_hh, b_hh));
    const std::vector<Tensor> chunked = ops::chunk(gates, 4, 1);
    const Tensor ingate = ops::sigmoid(chunked[0]);
    const Tensor forgetgate = ops::sigmoid(chunked[1]);
    const Tensor cellgate = ops::tanh(chunked[2]);
    const Tensor outgate = ops::sigmoid(chunked[3]);
    const Tensor cy = ops::add(ops::mul(forgetgate, cx),
                               ops::mul(ingate, cellgate));
    const Tensor hy = ops::mul(outgate, ops::tanh(cy));
    return {hy, cy};
}

namespace {

Tensor simple_rnn_cell(const Tensor& input, const Tensor& hx,
                       const Tensor& w_ih, const Tensor& w_hh,
                       const std::optional<Tensor>& b_ih,
                       const std::optional<Tensor>& b_hh, bool use_tanh) {
    const Tensor gates = ops::add(ops::linear(input, w_ih, b_ih),
                                  ops::linear(hx, w_hh, b_hh));
    return use_tanh ? ops::tanh(gates) : ops::relu(gates);
}

} // anonymous namespace

Tensor rnn_relu_cell_native(const Tensor& input, const Tensor& hx,
                            const Tensor& w_ih, const Tensor& w_hh,
                            const std::optional<Tensor>& b_ih,
                            const std::optional<Tensor>& b_hh) {
    return simple_rnn_cell(input, hx, w_ih, w_hh, b_ih, b_hh, false);
}

Tensor rnn_tanh_cell_native(const Tensor& input, const Tensor& hx,
                            const Tensor& w_ih, const Tensor& w_hh,
                            const std::optional<Tensor>& b_ih,
                            const std::optional<Tensor>& b_hh) {
    return simple_rnn_cell(input, hx, w_ih, w_hh, b_ih, b_hh, true);
}

Tensor gru_cell_native(const Tensor& input, const Tensor& hx,
                       const Tensor& w_ih, const Tensor& w_hh,
                       const std::optional<Tensor>& b_ih,
                       const std::optional<Tensor>& b_hh) {
    const std::vector<Tensor> gi = ops::chunk(ops::linear(input, w_ih, b_ih), 3, 1);
    const std::vector<Tensor> gh = ops::chunk(ops::linear(hx, w_hh, b_hh), 3, 1);
    const Tensor r = ops::sigmoid(ops::add(gi[0], gh[0]));
    const Tensor z = ops::sigmoid(ops::add(gi[1], gh[1]));
    const Tensor n = ops::tanh(ops::add(gi[2], ops::mul(r, gh[2])));
    // hy = (1 - z) * n + z * h, with (1 - z) * n expanded to n - z * n.
    return ops::add(ops::sub(n, ops::mul(z, n)), ops::mul(z, hx));
}

TENSORPLAY_LIBRARY_IMPL(Composite, RNNComposite) {
    m.impl("lstm_cell", lstm_cell_native);
    m.impl("rnn_relu_cell", rnn_relu_cell_native);
    m.impl("rnn_tanh_cell", rnn_tanh_cell_native);
    m.impl("gru_cell", gru_cell_native);
}

} // namespace composite
} // namespace tensorplay
