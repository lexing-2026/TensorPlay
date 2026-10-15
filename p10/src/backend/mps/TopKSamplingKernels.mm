#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"

#include "backend/mps/MPSDevice.h"

#include <tuple>
#include <vector>

#include "tensorplay/ops/TPXOpsGenerated.h"

namespace tensorplay {
namespace mps {

namespace ops = tpx::ops;

namespace {

int64_t wrap_dim(int64_t dim, int64_t ndim) {
  int64_t d = dim < 0 ? dim + ndim : dim;
  if (d < 0 || d >= ndim) {
    TP_THROW(IndexError, "Dimension out of range (expected to be in range of [",
             -ndim, ", ", ndim - 1, "], but got ", dim, ")");
  }
  return d;
}

// Slices `[0, k)` along `dim` of a sorted tensor.
Tensor take_front(const Tensor& t, int64_t dim, int64_t k) {
  if (k <= 0) {
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(t.shape());
    shape[static_cast<size_t>(dim)] = 0;
    return Tensor::empty(shape, t.dtype(), t.device());
  }
  return ops::slice(t, dim, 0, k, 1);
}

} // namespace

// topk via sort + slice: sorting along the requested dim is GPU work
// (ScanSort shader), slicing is a view.
std::tuple<Tensor, Tensor> topk_kernel(const Tensor& self, int64_t k,
                                       int64_t dim, bool largest, bool sorted,
                                       int64_t impl) {
  (void)impl;
  if (self.dim() == 0) {
    TP_THROW(RuntimeError, "topk(): input must be at least 1-dimensional");
  }
  const int64_t ndim = self.dim();
  dim = wrap_dim(dim, ndim);
  const int64_t dim_size = self.size(dim);
  if (k < 0 || k > dim_size) {
    TP_THROW(RuntimeError,
             "topk(): k (", k, ") must be in the range [0, ", dim_size, "]");
  }
  if (!sorted && k == dim_size) {
    // Full slice is already sorted by construction; the flag only affects
    // the ordering guarantee of the returned values.
    sorted = true;
  }
  auto [values, indices] = ops::sort(self, dim, /*descending=*/largest);
  return {take_front(values, dim, k), take_front(indices, dim, k)};
}

// GPU-composed nucleus/top-k sampling:
//   1. logits /= temperature, softmax
//   2. top_k threshold masking (keep values >= k-th largest)
//   3. top_p nucleus filtering on the sorted distribution + renormalize
//   4. cumulative distribution, then binary-search each row's uniform draw
//      with the searchsorted shader and gather the original token ids.
Tensor sample_kernel(const Tensor& logits, double temperature, int64_t top_k,
                     double top_p, int64_t impl) {
  (void)impl;
  if (temperature <= 0) {
    TP_THROW(RuntimeError, "sample: temperature must be > 0");
  }
  if (top_p <= 0 || top_p > 1.0) {
    TP_THROW(RuntimeError, "sample: top_p must be in (0, 1]");
  }
  if (logits.dim() != 2) {
    TP_THROW(RuntimeError, "sample: logits must be 2D [batch, vocab]");
  }
  const int64_t rows = logits.size(0);
  const int64_t cols = logits.size(1);
  if (cols == 0) {
    TP_THROW(RuntimeError, "sample: vocab size must be > 0");
  }

  Tensor input = logits.is_contiguous() ? logits : logits.contiguous();
  if (input.dtype() != DType::Float32) {
    input = input.to(DType::Float32);
  }
  if (rows == 0) {
    return Tensor::empty({0}, DType::Int64, input.device());
  }

  // 1. Temperature-scaled probabilities.
  Tensor scaled = ops::div(input, Scalar(temperature));
  Tensor probs = ops::softmax(scaled, -1);

  // 2. top-k threshold masking on the raw logits (before softmax is already
  //    applied, so threshold on probabilities is equivalent for ranking).
  if (top_k > 0 && top_k < cols) {
    auto [sorted_vals, sorted_idx] = ops::sort(probs, -1, /*descending=*/true);
    Tensor kth = take_front(sorted_vals, -1, top_k);
    kth = ops::slice(kth, -1, top_k - 1, top_k, 1);  // [rows, 1]
    probs = ops::where(ops::ge(probs, kth), probs, Tensor::full(
        static_cast<std::vector<int64_t>>(probs.shape()), 0.0, DType::Float32,
        probs.device()));
    (void)sorted_idx;
  }

  // 3. Nucleus filtering on the sorted probabilities.
  if (top_p < 1.0) {
    auto [sorted_probs, sorted_idx] = ops::sort(probs, -1, /*descending=*/true);
    Tensor csum = ops::cumsum(sorted_probs, -1);
    // Keep the smallest set whose cumulative mass reaches top_p.
    Tensor keep = ops::le(ops::sub(csum, sorted_probs), Scalar(top_p));
    Tensor filtered = ops::where(keep, sorted_probs, Tensor::full(
        static_cast<std::vector<int64_t>>(sorted_probs.shape()), 0.0,
        DType::Float32, sorted_probs.device()));
    Tensor total = ops::sum(filtered, std::vector<int64_t>{-1},
                            /*keepdim=*/true);
    Tensor renorm = ops::div(filtered, total);
    Tensor cdf = ops::cumsum(renorm, -1);

    // 4. Sample each row with a uniform draw.
    Tensor u = ops::rand({rows, 1}, DType::Float32, input.device());
    Tensor pos = ops::searchsorted(cdf, u, /*out_int32=*/false,
                                 /*right=*/true);
    pos = ops::squeeze(pos, -1);  // [rows]
    Tensor pos_2d = ops::unsqueeze(pos, -1);  // [rows, 1]
    Tensor token_sorted = ops::gather(sorted_idx, -1, pos_2d);  // [rows, 1]
    return ops::squeeze(token_sorted, -1);
  }

  // No nucleus filtering: sample directly from the full distribution.
  Tensor cdf = ops::cumsum(probs, -1);
  Tensor u = ops::rand({rows, 1}, DType::Float32, input.device());
  Tensor pos = ops::searchsorted(cdf, u, /*out_int32=*/false, /*right=*/true);
  pos = ops::squeeze(pos, -1);
  Tensor pos_2d = ops::unsqueeze(pos, -1);
  return ops::squeeze(pos_2d, -1);  // token id == position in unsorted cdf
}

// multinomial: replacement=True samples via inverse-CDF (cumsum + random +
// searchsorted); replacement=False uses the Gumbel-top-k trick (add Gumbel
// noise to log probabilities, take top-k).  Both paths stay on the GPU.
Tensor multinomial_kernel(const Tensor& self, int64_t num_samples,
                          bool replacement,
                          std::optional<Generator> generator) {
  (void)generator;
  if (num_samples < 0) {
    TP_THROW(RuntimeError, "multinomial: num_samples must be >= 0");
  }
  if (!self.device().is_mps()) {
    TP_THROW(RuntimeError, "multinomial: tensor must be on the MPS device");
  }
  if (self.dim() != 1 && self.dim() != 2) {
    TP_THROW(RuntimeError, "multinomial: input must be 1D or 2D");
  }
  const bool is_1d = self.dim() == 1;
  const int64_t rows = is_1d ? 1 : self.size(0);
  const int64_t cols = is_1d ? self.numel() : self.size(1);
  if (cols == 0 && num_samples > 0) {
    TP_THROW(RuntimeError, "multinomial: input must have at least one category");
  }
  const std::vector<int64_t> out_shape =
      is_1d ? std::vector<int64_t>{num_samples}
            : std::vector<int64_t>{rows, num_samples};
  if (num_samples == 0) {
    return Tensor::empty(out_shape, DType::Int64, self.device());
  }

  Tensor probs =
      self.dtype() == DType::Float32 ? self : self.to(DType::Float32);

  if (replacement) {
    // Unnormalized CDF, sampled with u in [0, total); searchsorted gives the
    // category index, clamped to [0, cols-1].
    Tensor cdf = ops::cumsum(probs, -1, DType::Float32);
    Tensor totals = ops::unsqueeze(ops::select(cdf, -1, cols - 1), -1);
    Tensor uniform =
        ops::mul(ops::rand(out_shape, DType::Float32, self.device()), totals);
    Tensor indices = ops::searchsorted(cdf, uniform, false, false,
                                       std::nullopt, std::nullopt);
    return ops::clamp(indices, Scalar(0), Scalar(cols - 1));
  }

  // Without replacement: Gumbel-top-k on log probabilities.  Zero-probability
  // categories get -inf scores and are naturally avoided.
  Tensor u = ops::clamp_min(ops::rand({rows, cols}, DType::Float32,
                                       self.device()),
                            Scalar(1e-9f));
  Tensor gumbel = ops::neg(ops::log(ops::neg(ops::log(u))));
  Tensor scores = ops::add(ops::log(probs), gumbel);
  auto top = ops::topk(scores, num_samples, -1, true, true);
  // topk already returns 1D [num_samples] for 1D input and [rows, samples]
  // for 2D input, matching the expected output shape in both cases.
  return std::get<1>(top);
}

TENSORPLAY_LIBRARY_IMPL(MPS, TopKSamplingKernels) {
  m.impl("topk", topk_kernel);
  m.impl("sample", sample_kernel);
  m.impl("multinomial", multinomial_kernel);
}

} // namespace mps
} // namespace tensorplay