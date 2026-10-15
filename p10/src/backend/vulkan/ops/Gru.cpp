#ifdef USE_VULKAN

#include "Blocks.h"
#include "Common.h"
#include "Convert.h"
#include "Utils.h"

#include "../api/Context.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <tuple>
#include <vector>

namespace tensorplay {
namespace vulkan {
namespace ops {

namespace {

// RNN weights live on the host in the public contract; materialize them on
// the device so the linear-algebra kernels below see dense Vulkan tensors.
Tensor ensure_vulkan(const Tensor& t) {
  if (t.device().is_vulkan()) return t;
  return t.to(Device(DeviceType::Vulkan));
}

void validate_rnn_common(
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool training,
    bool bidirectional,
    const Tensor& input,
    size_t param_count) {
  TP_CHECK(
      has_biases, "Vulkan RNN requires has_biases=true");
  TP_CHECK(
      !training, "Vulkan RNN requires training=false");
  TP_CHECK(
      !bidirectional, "Vulkan RNN requires bidirectional=false");
  TP_CHECK(
      dropout_p == 0.0, "Vulkan RNN requires dropout=0.0");
  TP_CHECK(
      input.dim() == 3, "Vulkan RNN expects a 3d input");
  TP_CHECK(
      param_count == static_cast<size_t>(4 * num_layers),
      "Vulkan RNN parameter count mismatch");
}

// Standard GRU cell with the tanh candidate.  The three gates live in the
// same row band of the packed input/hidden weights (r, z, n order), so one
// matrix product per side yields all gates and the slices are plain column
// bands of the result.
Tensor gru_cell_vulkan(
    const Tensor& x_t,
    const Tensor& h,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  using tpx::ops::add;
  using tpx::ops::addmm;
  using tpx::ops::mul;
  using tpx::ops::rsub;
  using tpx::ops::sigmoid;
  using tpx::ops::slice;
  using tpx::ops::tanh;
  using tpx::ops::transpose;

  const int64_t hidden = w_hh.size(0) / 3;

  const Tensor gate_x = addmm(b_ih, x_t, transpose(w_ih, 0, 1));
  const Tensor gate_h = addmm(b_hh, h, transpose(w_hh, 0, 1));

  const Tensor gx_r = slice(gate_x, 1, 0, hidden);
  const Tensor gx_z = slice(gate_x, 1, hidden, 2 * hidden);
  const Tensor gx_n = slice(gate_x, 1, 2 * hidden, 3 * hidden);
  const Tensor gh_r = slice(gate_h, 1, 0, hidden);
  const Tensor gh_z = slice(gate_h, 1, hidden, 2 * hidden);
  const Tensor gh_n = slice(gate_h, 1, 2 * hidden, 3 * hidden);

  const Tensor r = sigmoid(add(gx_r, gh_r));
  const Tensor z = sigmoid(add(gx_z, gh_z));
  const Tensor n = tanh(add(gx_n, mul(r, gh_n)));

  return add(mul(rsub(z, Scalar(1.0)), n), mul(z, h));
}

// Standard LSTM cell: input, forget, candidate and output gates share the
// packed rows of the weights (i, f, g, o order).
std::tuple<Tensor, Tensor> lstm_cell_vulkan(
    const Tensor& x_t,
    const Tensor& h,
    const Tensor& c,
    const Tensor& w_ih,
    const Tensor& w_hh,
    const Tensor& b_ih,
    const Tensor& b_hh) {
  using tpx::ops::add;
  using tpx::ops::addmm;
  using tpx::ops::mul;
  using tpx::ops::sigmoid;
  using tpx::ops::slice;
  using tpx::ops::tanh;
  using tpx::ops::transpose;

  const int64_t hidden = w_hh.size(0) / 4;

  const Tensor gate_x = addmm(b_ih, x_t, transpose(w_ih, 0, 1));
  const Tensor gate_h = addmm(b_hh, h, transpose(w_hh, 0, 1));

  const Tensor gx_i = slice(gate_x, 1, 0, hidden);
  const Tensor gx_f = slice(gate_x, 1, hidden, 2 * hidden);
  const Tensor gx_g = slice(gate_x, 1, 2 * hidden, 3 * hidden);
  const Tensor gx_o = slice(gate_x, 1, 3 * hidden, 4 * hidden);
  const Tensor gh_i = slice(gate_h, 1, 0, hidden);
  const Tensor gh_f = slice(gate_h, 1, hidden, 2 * hidden);
  const Tensor gh_g = slice(gate_h, 1, 2 * hidden, 3 * hidden);
  const Tensor gh_o = slice(gate_h, 1, 3 * hidden, 4 * hidden);

  const Tensor i = sigmoid(add(gx_i, gh_i));
  const Tensor f = sigmoid(add(gx_f, gh_f));
  const Tensor g = tanh(add(gx_g, gh_g));
  const Tensor o = sigmoid(add(gx_o, gh_o));

  const Tensor c_new = add(mul(f, c), mul(i, g));
  const Tensor h_new = mul(o, tanh(c_new));
  return {h_new, c_new};
}

} // namespace

std::tuple<Tensor, Tensor> gru_kernel(
    const Tensor& input,
    const std::vector<Tensor>& hx,
    const std::vector<Tensor>& params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool training,
    bool bidirectional,
    bool batch_first) {
  validate_rnn_common(
      has_biases, num_layers, dropout_p, training, bidirectional, input,
      params.size());
  TP_CHECK(
      hx.size() == 1, "Vulkan gru expects a single hx tensor");

  Tensor x = ensure_vulkan(input);
  const bool swap = !batch_first;
  if (swap) {
    x = tpx::ops::transpose(x, 0, 1);
  }
  const int64_t seq = x.size(1);

  Tensor hx0 = ensure_vulkan(hx[0]);

  std::vector<Tensor> h_n_list;
  h_n_list.reserve(num_layers);

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    Tensor h = tpx::ops::select(hx0, 0, layer);
    const Tensor w_ih = ensure_vulkan(params[layer * 4]);
    const Tensor w_hh = ensure_vulkan(params[layer * 4 + 1]);
    const Tensor b_ih = ensure_vulkan(params[layer * 4 + 2]);
    const Tensor b_hh = ensure_vulkan(params[layer * 4 + 3]);

    std::vector<Tensor> outputs;
    outputs.reserve(seq);
    for (int64_t t = 0; t < seq; ++t) {
      const Tensor x_t = tpx::ops::select(x, 1, t);
      h = gru_cell_vulkan(x_t, h, w_ih, w_hh, b_ih, b_hh);
      outputs.push_back(h);
    }
    x = tpx::ops::stack(outputs, 1);
    h_n_list.push_back(h);
  }

  Tensor output = x;
  if (swap) {
    output = tpx::ops::transpose(output, 0, 1);
  }
  const Tensor h_n = tpx::ops::stack(h_n_list, 0);
  return {output, h_n};
}

std::tuple<Tensor, Tensor, Tensor> lstm_kernel(
    const Tensor& input,
    const std::vector<Tensor>& hx,
    const std::vector<Tensor>& params,
    bool has_biases,
    int64_t num_layers,
    double dropout_p,
    bool training,
    bool bidirectional,
    bool batch_first) {
  validate_rnn_common(
      has_biases, num_layers, dropout_p, training, bidirectional, input,
      params.size());
  TP_CHECK(
      hx.size() == 2, "Vulkan lstm expects hx and cx tensors");

  Tensor x = ensure_vulkan(input);
  const bool swap = !batch_first;
  if (swap) {
    x = tpx::ops::transpose(x, 0, 1);
  }
  const int64_t seq = x.size(1);

  Tensor hx0 = ensure_vulkan(hx[0]);
  Tensor cx0 = ensure_vulkan(hx[1]);

  std::vector<Tensor> h_n_list;
  std::vector<Tensor> c_n_list;
  h_n_list.reserve(num_layers);
  c_n_list.reserve(num_layers);

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    Tensor h = tpx::ops::select(hx0, 0, layer);
    Tensor c = tpx::ops::select(cx0, 0, layer);
    const Tensor w_ih = ensure_vulkan(params[layer * 4]);
    const Tensor w_hh = ensure_vulkan(params[layer * 4 + 1]);
    const Tensor b_ih = ensure_vulkan(params[layer * 4 + 2]);
    const Tensor b_hh = ensure_vulkan(params[layer * 4 + 3]);

    std::vector<Tensor> outputs;
    outputs.reserve(seq);
    for (int64_t t = 0; t < seq; ++t) {
      const Tensor x_t = tpx::ops::select(x, 1, t);
      std::tie(h, c) = lstm_cell_vulkan(x_t, h, c, w_ih, w_hh, b_ih, b_hh);
      outputs.push_back(h);
    }
    x = tpx::ops::stack(outputs, 1);
    h_n_list.push_back(h);
    c_n_list.push_back(c);
  }

  Tensor output = x;
  if (swap) {
    output = tpx::ops::transpose(output, 0, 1);
  }
  const Tensor h_n = tpx::ops::stack(h_n_list, 0);
  const Tensor c_n = tpx::ops::stack(c_n_list, 0);
  return {output, h_n, c_n};
}

} // namespace ops
} // namespace vulkan
} // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(Vulkan, GruKernels) {
  m.impl("gru", &tensorplay::vulkan::ops::gru_kernel);
  m.impl("lstm", &tensorplay::vulkan::ops::lstm_kernel);
}

#endif /* USE_VULKAN */