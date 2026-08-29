#include "Tensor.h"
#include "Dispatcher.h"
#include "Generator.h"
#include "DistributionsHelper.h"
#include "Exception.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <tuple>

#if defined(USE_MKL)
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

// CPU sampling/selection operators: multinomial, topk, and the LLM
// token sampler (temperature / top-k / top-p).  Canonical serial
// implementations; the CUDA kernels use the same semantics.
// layout).

namespace tensorplay {
namespace cpu {

namespace {

// Discrete sampling via inverse-CDF with a single double uniform per draw,
template <typename It>
int64_t sample_discrete(Generator& gen, It begin, It end) {
  using T = typename std::iterator_traits<It>::value_type;
  T total = 0;
  for (It it = begin; it != end; ++it) total += *it;
  if (!(total > 0)) return 0;
  uniform_real_distribution<double> uniform(0.0, 1.0);
  double r = uniform(&gen) * static_cast<double>(total);
  double cum = 0.0;
  int64_t idx = 0;
  for (It it = begin; it != end; ++it, ++idx) {
    cum += static_cast<double>(*it);
    if (r < cum) return idx;
  }
  return static_cast<int64_t>(end - begin) - 1;
}

// Discrete sampling from a row of (unnormalized) weights in [0, cols).
int64_t sample_from_weights(const float* row, int64_t cols) {
  return sample_discrete(default_generator(), row, row + cols);
}

} // namespace

Tensor multinomial_kernel_cpu(const Tensor& self, int64_t num_samples, bool replacement, int64_t impl) {
  if (num_samples < 0) {
    TP_THROW(RuntimeError, "multinomial: num_samples must be >= 0");
  }
  Tensor prob = self.contiguous();
  if (prob.dim() > 2) {
    TP_THROW(RuntimeError, "multinomial: input must be 1D or 2D");
  }
  bool is_1d = prob.dim() == 1;
  int64_t rows = is_1d ? 1 : prob.size(0);
  int64_t cols = is_1d ? prob.numel() : prob.size(1);
  if (cols == 0 && num_samples > 0) {
    TP_THROW(RuntimeError, "multinomial: input must have at least one category");
  }
  std::vector<double> work;
  std::vector<double> wrow;
  Tensor f32;
  const float* pdata = nullptr;
  if (prob.dtype() == DType::Float32) {
    pdata = prob.data_ptr<float>();
  } else {
    f32 = prob.to(DType::Float32);
    pdata = f32.data_ptr<float>();
  }

  std::vector<int64_t> out_shape = is_1d ? std::vector<int64_t>{num_samples}
                                         : std::vector<int64_t>{rows, num_samples};
  Tensor result = Tensor::empty(out_shape, DType::Int64, prob.device());
  int64_t* rdata = result.data_ptr<int64_t>();
  if (num_samples == 0) return result;

  auto& gen = default_generator();
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = pdata + r * cols;
    if (replacement) {
      for (int64_t s = 0; s < num_samples; ++s) {
        rdata[is_1d ? s : r * num_samples + s] = sample_discrete(gen, row, row + cols);
      }
    } else {
      // Without replacement: repeatedly sample from the remaining mass,
      std::vector<double> remaining(cols);
      for (int64_t c = 0; c < cols; ++c) remaining[c] = row[c];
      for (int64_t s = 0; s < num_samples; ++s) {
        int64_t pick = sample_discrete(gen, remaining.begin(), remaining.end());
        rdata[is_1d ? s : r * num_samples + s] = pick;
        remaining[pick] = 0.0;
      }
    }
  }
  return result;
}

std::tuple<Tensor, Tensor> topk_kernel_cpu(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted, int64_t impl) {
  Tensor input = self.contiguous();
  int64_t ndim = input.dim();
  if (dim < 0) dim += ndim;
  if (dim != ndim - 1) {
    TP_THROW(NotImplementedError, "topk: only the last dim is supported for now");
  }
  int64_t rows = input.numel() / input.size(ndim - 1);
  int64_t cols = input.size(ndim - 1);
  if (k < 0 || k > cols) {
    TP_THROW(RuntimeError, "topk: k must be in [0, cols]");
  }
  if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) {
    TP_THROW(NotImplementedError, "topk: only Float32/Float64 are supported for now");
  }

  std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(input.shape());
  shape[ndim - 1] = k;
  Tensor values = Tensor::empty(shape, input.dtype(), input.device());
  Tensor indices = Tensor::empty(shape, DType::Int64, input.device());
  if (k == 0) return {values, indices};

  auto run = [&](auto st) {
    using scalar_t = decltype(st);
    const scalar_t* idata = input.data_ptr<scalar_t>();
    scalar_t* vdata = values.data_ptr<scalar_t>();
    int64_t* idxdata = indices.data_ptr<int64_t>();

    std::vector<int64_t> perm(cols);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t r = 0; r < rows; ++r) {
      const scalar_t* row = idata + r * cols;
      std::partial_sort(perm.begin(), perm.begin() + k, perm.end(),
                        [row, largest](int64_t a, int64_t b) {
                          return largest ? (row[a] > row[b]) : (row[a] < row[b]);
                        });
      for (int64_t i = 0; i < k; ++i) {
        vdata[r * k + i] = row[perm[i]];
        idxdata[r * k + i] = perm[i];
      }
    }
  };
  if (input.dtype() == DType::Float64) run(double{}); else run(float{});
  return {values, indices};
}

Tensor sample_kernel_cpu(const Tensor& logits, double temperature, int64_t top_k, double top_p, int64_t impl) {
  if (temperature <= 0) {
    TP_THROW(RuntimeError, "sample: temperature must be > 0");
  }
  if (top_p <= 0 || top_p > 1.0) {
    TP_THROW(RuntimeError, "sample: top_p must be in (0, 1]");
  }
  Tensor input = logits.contiguous();
  if (input.dim() != 2) {
    TP_THROW(RuntimeError, "sample: logits must be 2D [batch, vocab]");
  }
  if (input.dtype() != DType::Float32) {
    input = input.to(DType::Float32);
  }
  int64_t rows = input.size(0);
  int64_t cols = input.size(1);
  if (cols == 0) {
    TP_THROW(RuntimeError, "sample: vocab size must be > 0");
  }

  Tensor result = Tensor::empty({rows}, DType::Int64, input.device());
  const float* idata = input.data_ptr<float>();
  int64_t* rdata = result.data_ptr<int64_t>();
  auto& gen = default_generator();

  std::vector<float> probs(cols);
  std::vector<int64_t> order(cols);
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = idata + r * cols;
    double mx = -INFINITY;
    for (int64_t c = 0; c < cols; ++c) mx = std::max(mx, (double)row[c]);
    double total = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      double e = std::exp(((double)row[c] - mx) / temperature);
      probs[c] = (float)e;
      total += e;
    }
    for (int64_t c = 0; c < cols; ++c) probs[c] = (float)(probs[c] / total);

    // top-k filter (exact, ties by lower index)
    if (top_k > 0 && top_k < cols) {
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
                        [&probs](int64_t a, int64_t b) { return probs[a] > probs[b]; });
      std::vector<char> keep(cols, 0);
      for (int64_t i = 0; i < top_k; ++i) keep[order[i]] = 1;
      for (int64_t c = 0; c < cols; ++c)
        if (!keep[c]) probs[c] = 0.f;
    }
    // top-p filter (nucleus, ties kept together)
    if (top_p < 1.0) {
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&probs](int64_t a, int64_t b) {
                  return probs[a] > probs[b] || (probs[a] == probs[b] && a < b);
                });
      double cum = 0.0;
      int64_t keep_n = 0;
      for (int64_t i = 0; i < cols; ++i) {
        cum += probs[order[i]];
        ++keep_n;
        if (cum >= top_p) break;
      }
      std::vector<char> keep(cols, 0);
      for (int64_t i = 0; i < keep_n; ++i) keep[order[i]] = 1;
      for (int64_t c = 0; c < cols; ++c)
        if (!keep[c]) probs[c] = 0.f;
    }

    rdata[r] = sample_discrete(gen, probs.begin(), probs.end());
  }
  return result;
}

TENSORPLAY_LIBRARY_IMPL(CPU, SamplingKernels) {
  m.impl("multinomial", multinomial_kernel_cpu);
  m.impl("topk", topk_kernel_cpu);
  m.impl("sample", sample_kernel_cpu);
}

} // namespace cpu
} // namespace tensorplay
