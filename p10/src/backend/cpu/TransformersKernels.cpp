// CPU scaled-dot-product attention kernels.
// lives under transformers/, not an "llm" grab-bag.
//
// The fused flash path tiles the query and key axes, runs BLAS gemm for
// Q K^T and P V per tile with a row softmax in between (runtime-dispatched
// vector exp: AVX-512 -> AVX2 -> scalar), and shares the tiles over the
// intra-op pool.  f16/bf16 inputs accumulate in f32; f64 accumulates in f64.

#include <algorithm>
#include <type_traits>
#include <vector>

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "DTypeNames.h"
#include "Parallel.h"

#include "../composite/AttentionComposite.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(USE_MKL)
#include <mkl.h>
#elif defined(USE_BLAS)
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif
#include "cpu/vec/SleefShims.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define TP_SDPA_SLEEF 1
#endif

namespace tensorplay {
namespace cpu {

namespace ops = tensorplay::tpx::ops;

namespace {

// Vectorized exp over a row.  Each lane width lives in its own function so
// the wider instruction set is confined to code reached only after the
// running CPU reported it; the TU itself compiles with base x86-64 flags.
// Every helper returns how many leading elements it handled and the caller
// finishes the tail through libm.
#if defined(TP_SDPA_SLEEF)
__attribute__((target("avx512f")))
int64_t vexp_f32_avx512(const float* x, float* y, int64_t n) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16)
    _mm512_storeu_ps(y + i, tensorplay::tpsleef::exp(_mm512_loadu_ps(x + i)));
  return i;
}

__attribute__((target("avx2")))
int64_t vexp_f32_avx2(const float* x, float* y, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(y + i, tensorplay::tpsleef::exp(_mm256_loadu_ps(x + i)));
  return i;
}

__attribute__((target("avx512f")))
int64_t vexp_f64_avx512(const double* x, double* y, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    _mm512_storeu_pd(y + i, tensorplay::tpsleef::exp(_mm512_loadu_pd(x + i)));
  return i;
}

__attribute__((target("avx2")))
int64_t vexp_f64_avx2(const double* x, double* y, int64_t n) {
  int64_t i = 0;
  for (; i + 4 <= n; i += 4)
    _mm256_storeu_pd(y + i, tensorplay::tpsleef::exp(_mm256_loadu_pd(x + i)));
  return i;
}

// 2: AVX-512F, 1: AVX2, 0: neither.
int vexp_isa_level() {
  static const int level = __builtin_cpu_supports("avx512f")
                               ? 2
                               : (__builtin_cpu_supports("avx2") ? 1 : 0);
  return level;
}
#endif

void vexp_f32(const float* x, float* y, int64_t n) {
  int64_t i = 0;
#if defined(TP_SDPA_SLEEF)
  const int level = vexp_isa_level();
  if (level == 2) {
    i = vexp_f32_avx512(x, y, n);
  } else if (level == 1) {
    i = vexp_f32_avx2(x, y, n);
  }
#endif
  for (; i < n; ++i) y[i] = std::exp(x[i]);
}

void vexp_f64(const double* x, double* y, int64_t n) {
  int64_t i = 0;
#if defined(TP_SDPA_SLEEF)
  const int level = vexp_isa_level();
  if (level == 2) {
    i = vexp_f64_avx512(x, y, n);
  } else if (level == 1) {
    i = vexp_f64_avx2(x, y, n);
  }
#endif
  for (; i < n; ++i) y[i] = std::exp(x[i]);
}

} // namespace

// The public entry point names what the caller wants -- a mask to add, a
// proportion to drop, a scale to fold in, grouped query heads -- and the
// attention those describe is computed once, for any device, by the composite
// that expresses it.  Going through it rather than keeping a second copy of
// the same arithmetic here is what keeps the two from disagreeing about what
// the mask and the scale mean.
Tensor sdpa_kernel_cpu(const Tensor& query, const Tensor& key,
                       const Tensor& value,
                       const std::optional<Tensor>& attn_mask, double dropout_p,
                       bool is_causal, std::optional<double> scale,
                       bool enable_gqa) {
  auto [output, logsumexp] = composite::sdpa_math_composite(
      query, key, value, attn_mask, dropout_p, is_causal,
      /*dropout_mask=*/std::nullopt, scale, enable_gqa);
  (void)logsumexp;
  return output;
}

// The gradient of the attention above, asked for in the same terms it was
// computed in.  Delegating to the composite that expresses it means a masked,
// dropped, scaled or grouped call is differentiated by the same arithmetic
// that produced it, rather than by a second path that knows only the plain
// case.
std::tuple<Tensor, Tensor, Tensor> sdpa_backward_kernel_cpu(
    const Tensor& grad_output, const Tensor& query, const Tensor& key,
    const Tensor& value, const std::optional<Tensor>& attn_mask, double dropout_p,
    bool is_causal, std::optional<double> scale, bool enable_gqa) {
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4 ||
      grad_output.dim() != 4) {
    TP_THROW(RuntimeError, "sdpa backward: q/k/v/grad_output must be 4D");
  }
  return composite::sdpa_math_backward_composite(
      grad_output, query, key, value, attn_mask, dropout_p, is_causal, scale,
      enable_gqa);
}

// ---------------------------------------------------------------------------
// semantics): self [M_total, K] @ mat2 [G, K, N] -> [M_total, N]; offs [G]
// holds cumulative END offsets, group g spans [prev_end, offs[g]).
// No-grad path is one dispatcher op wrapping per-group cblas_sgemm calls --
// the composite narrow/mm/cat loop it replaces pays ~4us dispatch overhead
// per expert per token-batch.  GradMode falls back to the differentiable
// composite so CIA records inner nodes automatically.
// ---------------------------------------------------------------------------
Tensor grouped_mm_cpu(const Tensor& self, const Tensor& mat2,
                      const Tensor& offs) {
  if (self.dim() != 2 || mat2.dim() != 3) {
    TP_THROW(RuntimeError,
             "grouped_mm(): expected 2D self and 3D mat2, got ", self.dim(),
             "D and ", mat2.dim(), "D");
  }
  const int64_t M = self.size(0), K = self.size(1);
  const int64_t G = mat2.size(0);
  if (mat2.size(1) != K) {
    TP_THROW(RuntimeError, "grouped_mm(): self.size(1) must match mat2.size(1): ",
             K, " vs ", mat2.size(1));
  }
  if (offs.dim() != 1 || offs.numel() != G) {
    TP_THROW(RuntimeError, "grouped_mm(): offs must be 1D of length mat2.size(0)=",
             G, ", got ", offs.dim(), "D/", offs.numel(), " elements");
  }
  if (self.dtype() != mat2.dtype()) {
    TP_THROW(RuntimeError, "grouped_mm(): expected self and mat2 to have the same dtype, but got: ",
             elementTypeName(self.dtype()), " != ",
             elementTypeName(mat2.dtype()));
  }
  if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
    TP_THROW(NotImplementedError,
             "grouped_mm cpu: expected float32/float64");
  }

  // Validate offsets: non-decreasing int32/int64 within [0, M].
  auto read_off = [&](int64_t i) -> int64_t {
    if (offs.dtype() == DType::Int32) return offs.data_ptr<int32_t>()[i];
    if (offs.dtype() == DType::Int64) return offs.data_ptr<int64_t>()[i];
    TP_THROW(TypeError, "grouped_mm(): offs must be int32 or int64");
  };
  int64_t prev = 0;
  for (int64_t g = 0; g < G; ++g) {
    const int64_t end = read_off(g);
    if (end < prev || end > M) {
      TP_THROW(RuntimeError, "grouped_mm(): offs must be non-decreasing in [0, M_total=",
               M, "], got offs[", g, "]=", end);
    }
    prev = end;
  }

  const bool needs_grad =
      GradMode::is_enabled() && (self.requires_grad() || mat2.requires_grad());
  if (needs_grad) {
    // Differentiable composite: mm over row slices, cat back together.
    std::vector<Tensor> parts;
    parts.reserve(G);
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int64_t len = end - start;
      if (len > 0) {
        Tensor wg = tpx::ops::narrow(mat2, 0, g, 1);
        wg = tpx::ops::reshape(wg, {K, mat2.size(2)});
        parts.push_back(tpx::ops::mm(tpx::ops::narrow(self, 0, start, len), wg));
      }
      start = end;
    }
    if (parts.empty()) {
      return Tensor::zeros({M, mat2.size(2)}, self.dtype(), self.device());
    }
    return tpx::ops::cat(parts, 0);
  }

#if !defined(USE_MKL) && !defined(USE_BLAS)
  TP_THROW(NotImplementedError, "grouped_mm cpu requires BLAS");
#else
  Tensor out = Tensor::empty({M, mat2.size(2)}, self.dtype(), self.device());
  if (M == 0 || mat2.size(2) == 0) return out;
  const int N = static_cast<int>(mat2.size(2));
  if (static_cast<int64_t>(N) * K > INT32_MAX ||
      static_cast<int64_t>(N) * M > INT32_MAX) {
    TP_THROW(RuntimeError, "grouped_mm cpu: shape too large for BLAS ints");
  }

  if (self.dtype() == DType::Float32) {
    float* od = out.data_ptr<float>();
    std::memset(od, 0, static_cast<size_t>(M) * N * sizeof(float));
    const float* ad = self.data_ptr<float>();
    const float* bd = mat2.data_ptr<float>();
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int len = static_cast<int>(end - start);
      if (len > 0) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, len, N,
                    static_cast<int>(K), 1.0f, ad + start * K,
                    static_cast<int>(K), bd + g * K * N, N, 1.0f,
                    od + start * N, N);
      }
      start = end;
    }
  } else {
    double* od = out.data_ptr<double>();
    std::memset(od, 0, static_cast<size_t>(M) * N * sizeof(double));
    const double* ad = self.data_ptr<double>();
    const double* bd = mat2.data_ptr<double>();
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int len = static_cast<int>(end - start);
      if (len > 0) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, len, N,
                    static_cast<int>(K), 1.0, ad + start * K,
                    static_cast<int>(K), bd + g * K * N, N, 1.0,
                    od + start * N, N);
      }
      start = end;
    }
  }
  return out;
#endif
}

// RoPE primitives follow the transformer attention path.  The layout matches
// [x0, x1, x2, x3, ...], while cos/sin are [positions, head_dim / 2].
namespace {

inline bool rope_float_dtype(DType dtype) {
  return dtype == DType::Float32 || dtype == DType::Float64 ||
         dtype == DType::Float16 || dtype == DType::BFloat16;
}

inline int64_t rope_tokens(const Tensor& input, const char* op) {
  if (input.dim() < 2) {
    TP_THROW(RuntimeError, op, ": input must have at least 2 dimensions");
  }
  if ((input.size(-1) & 1) != 0) {
    TP_THROW(RuntimeError, op, ": the last dimension must be even");
  }
  if (!rope_float_dtype(input.dtype())) {
    TP_THROW(NotImplementedError, op,
             ": only float32/float64/float16/bfloat16 are supported");
  }
  return input.size(-2);
}

struct RopeTable {
  int64_t rows;
  int64_t half_dim;
};

RopeTable check_rope_table(const Tensor& cos, const Tensor& sin,
                           int64_t half_dim, int64_t tokens,
                           int64_t position_offset, const Device& device,
                           const char* op) {
  if (position_offset < 0) {
    TP_THROW(RuntimeError, op, ": position_offset must be non-negative");
  }
  if (cos.device() != device || sin.device() != device ||
      cos.device() != sin.device()) {
    TP_THROW(DeviceMismatchError, op,
             ": input and cos/sin must be on the same device");
  }
  if (cos.dtype() != sin.dtype() || !rope_float_dtype(cos.dtype())) {
    TP_THROW(RuntimeError, op,
             ": cos and sin must have the same floating dtype");
  }

  int64_t rows = 0;
  if (cos.dim() == 1 && sin.dim() == 1) {
    if (cos.size(0) != half_dim || sin.size(0) != half_dim) {
      TP_THROW(RuntimeError, op,
               ": 1D cos/sin tables must have head_dim/2 entries");
    }
    rows = 1;
  } else if (cos.dim() == 2 && sin.dim() == 2) {
    if (cos.size(1) != half_dim || sin.size(1) != half_dim ||
        cos.size(0) != sin.size(0)) {
      TP_THROW(RuntimeError, op,
               ": cos/sin tables must be [positions, head_dim/2]");
    }
    rows = cos.size(0);
  } else {
    TP_THROW(RuntimeError, op,
             ": cos and sin must both be 1D or both be 2D");
  }

  if (rows != 1 &&
      (position_offset > rows || tokens > rows - position_offset)) {
    TP_THROW(RuntimeError, op,
             ": cos/sin table is shorter than the requested positions");
  }
  if (rows == 1 && position_offset != 0) {
    TP_THROW(RuntimeError, op,
             ": position_offset must be zero for a one-row table");
  }
  return {rows, half_dim};
}

template <typename T, typename C>
void rope_loop(const T* input, T* output, const C* cos, const C* sin,
               int64_t pairs, int64_t tokens, int64_t half_dim,
               int64_t table_rows, int64_t position_offset) {
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  parallel::parallel_for(0, pairs, parallel::GRAIN_SIZE,
                         [&](int64_t begin, int64_t end) {
    for (int64_t pair = begin; pair < end; ++pair) {
      const int64_t row = pair / half_dim;
      const int64_t pair_in_row = pair - row * half_dim;
      const int64_t token = row % tokens;
      const int64_t table_row = table_rows == 1
                                    ? 0
                                    : position_offset + token;
      const int64_t input_offset = row * (2 * half_dim) + 2 * pair_in_row;
      const int64_t table_offset = table_row * half_dim + pair_in_row;
      const Acc x0 = static_cast<Acc>(input[input_offset]);
      const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
      const Acc c = static_cast<Acc>(cos[table_offset]);
      const Acc s = static_cast<Acc>(sin[table_offset]);
      output[input_offset] = static_cast<T>(x0 * c - x1 * s);
      output[input_offset + 1] = static_cast<T>(x0 * s + x1 * c);
    }
  });
}

template <typename T, typename C>
Tensor rope_single_typed(const Tensor& input, const Tensor& cos,
                         const Tensor& sin, const RopeTable& table,
                         int64_t tokens, int64_t position_offset) {
  Tensor input_c = input.is_contiguous() ? input : input.contiguous();
  Tensor cos_c = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sin_c = sin.is_contiguous() ? sin : sin.contiguous();
  Tensor output = Tensor::empty(
      static_cast<std::vector<int64_t>>(input_c.shape()), input_c.dtype(),
      input_c.device());
  rope_loop<T, C>(input_c.data_ptr<T>(), output.data_ptr<T>(),
                  cos_c.data_ptr<C>(), sin_c.data_ptr<C>(),
                  input_c.numel() / 2, tokens, table.half_dim, table.rows,
                  position_offset);
  return output;
}

template <typename T, typename C>
std::tuple<Tensor, Tensor> rope_pair_typed(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  Tensor query_c = query.is_contiguous() ? query : query.contiguous();
  Tensor key_c = key.is_contiguous() ? key : key.contiguous();
  Tensor cos_c = cos.is_contiguous() ? cos : cos.contiguous();
  Tensor sin_c = sin.is_contiguous() ? sin : sin.contiguous();
  Tensor query_out = Tensor::empty(
      static_cast<std::vector<int64_t>>(query_c.shape()), query_c.dtype(),
      query_c.device());
  Tensor key_out = Tensor::empty(
      static_cast<std::vector<int64_t>>(key_c.shape()), key_c.dtype(),
      key_c.device());

  const int64_t query_pairs = query_c.numel() / 2;
  const int64_t key_pairs = key_c.numel() / 2;
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  parallel::parallel_for(
      0, query_pairs + key_pairs, parallel::GRAIN_SIZE,
      [&](int64_t begin, int64_t end) {
        for (int64_t pair = begin; pair < end; ++pair) {
          const T* input = query_c.data_ptr<T>();
          T* output = query_out.data_ptr<T>();
          int64_t local_pair = pair;
          int64_t tokens = query_tokens;
          if (pair >= query_pairs) {
            local_pair -= query_pairs;
            input = key_c.data_ptr<T>();
            output = key_out.data_ptr<T>();
            tokens = key_tokens;
          }
          const int64_t row = local_pair / table.half_dim;
          const int64_t pair_in_row = local_pair - row * table.half_dim;
          const int64_t token = row % tokens;
          const int64_t table_row = table.rows == 1
                                        ? 0
                                        : position_offset + token;
          const int64_t input_offset =
              row * (2 * table.half_dim) + 2 * pair_in_row;
          const int64_t table_offset = table_row * table.half_dim + pair_in_row;
          const Acc x0 = static_cast<Acc>(input[input_offset]);
          const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
          const Acc c = static_cast<Acc>(cos_c.data_ptr<C>()[table_offset]);
          const Acc s = static_cast<Acc>(sin_c.data_ptr<C>()[table_offset]);
          output[input_offset] = static_cast<T>(x0 * c - x1 * s);
          output[input_offset + 1] = static_cast<T>(x0 * s + x1 * c);
        }
      });
  return std::make_tuple(query_out, key_out);
}

template <typename T>
Tensor rope_single_dispatch(const Tensor& input, const Tensor& cos,
                            const Tensor& sin, const RopeTable& table,
                            int64_t tokens, int64_t position_offset) {
  switch (cos.dtype()) {
    case DType::Float32:
      return rope_single_typed<T, float>(input, cos, sin, table, tokens,
                                         position_offset);
    case DType::Float64:
      return rope_single_typed<T, double>(input, cos, sin, table, tokens,
                                          position_offset);
    case DType::Float16:
      return rope_single_typed<T, Half>(input, cos, sin, table, tokens,
                                        position_offset);
    case DType::BFloat16:
      return rope_single_typed<T, BFloat16>(input, cos, sin, table, tokens,
                                            position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported table dtype");
  }
}

template <typename T>
std::tuple<Tensor, Tensor> rope_pair_dispatch(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  switch (cos.dtype()) {
    case DType::Float32:
      return rope_pair_typed<T, float>(query, key, cos, sin, table,
                                       query_tokens, key_tokens,
                                       position_offset);
    case DType::Float64:
      return rope_pair_typed<T, double>(query, key, cos, sin, table,
                                        query_tokens, key_tokens,
                                        position_offset);
    case DType::Float16:
      return rope_pair_typed<T, Half>(query, key, cos, sin, table,
                                      query_tokens, key_tokens,
                                      position_offset);
    case DType::BFloat16:
      return rope_pair_typed<T, BFloat16>(query, key, cos, sin, table,
                                          query_tokens, key_tokens,
                                          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported table dtype");
  }
}

} // namespace

Tensor rotary_embedding_cpu(const Tensor& input, const Tensor& cos,
                            const Tensor& sin, int64_t position_offset) {
  const int64_t tokens = rope_tokens(input, "rotary_embedding");
  const RopeTable table = check_rope_table(
      cos, sin, input.size(-1) / 2, tokens, position_offset, input.device(),
      "rotary_embedding");
  switch (input.dtype()) {
    case DType::Float32:
      return rope_single_dispatch<float>(input, cos, sin, table, tokens,
                                         position_offset);
    case DType::Float64:
      return rope_single_dispatch<double>(input, cos, sin, table, tokens,
                                          position_offset);
    case DType::Float16:
      return rope_single_dispatch<Half>(input, cos, sin, table, tokens,
                                        position_offset);
    case DType::BFloat16:
      return rope_single_dispatch<BFloat16>(input, cos, sin, table, tokens,
                                            position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported input dtype");
  }
}

std::tuple<Tensor, Tensor> fused_rope_cpu(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, int64_t position_offset) {
  const int64_t query_tokens = rope_tokens(query, "fused_rope");
  const int64_t key_tokens = rope_tokens(key, "fused_rope");
  if (query.device() != key.device()) {
    TP_THROW(DeviceMismatchError,
             "fused_rope: query and key must be on the same device");
  }
  if (query.dtype() != key.dtype()) {
    TP_THROW(RuntimeError, "fused_rope: query and key must have the same dtype");
  }
  if (query.dim() != key.dim() || query.size(-1) != key.size(-1) ||
      query_tokens != key_tokens) {
    TP_THROW(RuntimeError,
             "fused_rope: query/key must have the same rank, token length, and head dimension");
  }
  const RopeTable table = check_rope_table(
      cos, sin, query.size(-1) / 2, query_tokens, position_offset,
      query.device(), "fused_rope");
  switch (query.dtype()) {
    case DType::Float32:
      return rope_pair_dispatch<float>(query, key, cos, sin, table,
                                       query_tokens, key_tokens,
                                       position_offset);
    case DType::Float64:
      return rope_pair_dispatch<double>(query, key, cos, sin, table,
                                        query_tokens, key_tokens,
                                        position_offset);
    case DType::Float16:
      return rope_pair_dispatch<Half>(query, key, cos, sin, table,
                                      query_tokens, key_tokens,
                                      position_offset);
    case DType::BFloat16:
      return rope_pair_dispatch<BFloat16>(query, key, cos, sin, table,
                                          query_tokens, key_tokens,
                                          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported input dtype");
  }
}
// ---------------------------------------------------------------------------
// Private SDPA backends (dispatcher contracts declared in
// config/native_functions.yaml).  The math backend, the multi-head fast-path
// composite, and the backend selector share one body with the CUDA side
// (composite/AttentionComposite.h); the flash-for-CPU fused kernels are
// CPU-only and stay here.
// ---------------------------------------------------------------------------

std::tuple<Tensor, Tensor> sdpa_math_kernel_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale, bool enable_gqa) {
  return composite::sdpa_math_composite(query, key, value, attn_mask,
                                        dropout_p, is_causal, dropout_mask,
                                        scale, enable_gqa);
}

std::tuple<Tensor, Tensor> native_multi_head_attention_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head, const Tensor& qkv_weight,
    const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias,
    const std::optional<Tensor>& mask, bool need_weights,
    bool average_attn_weights, std::optional<int64_t> mask_type) {
  return composite::native_mha_composite(
      query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
      proj_weight, proj_bias, mask, need_weights, average_attn_weights,
      mask_type);
}

int64_t fused_sdp_choice_cpu(const Tensor& query, const Tensor& key,
                             const Tensor& value,
                             const std::optional<Tensor>& attn_mask,
                             double dropout_p, bool is_causal,
                             std::optional<double> scale, bool enable_gqa) {
  return composite::fused_sdp_choice_common(query, key, value, attn_mask,
                                            dropout_p, is_causal, scale,
                                            enable_gqa);
}

// ---------------------------------------------------------------------------
// Blocked flash attention on the CPU.
//
// The query axis is cut into blocks of `flash_q_split` rows and the key axis
// into blocks of `kFlashKvSplit` columns.  A (query block, key block) pair
// costs two matrix products -- scores = Q K^T and the weighted values P V --
// with a row softmax in between, and the key blocks of one query block are
// merged with the running-max/running-sum recurrence, so the full
// [Tq, Skv] score matrix never exists: the scratch per thread is one
// q_split x kv_split tile.  Work is shared over batch x head x query block.
//
// The backward replays the same tiling.  The probabilities are rebuilt from
// the saved per-row logsumexp as p = exp(s - lse) and
//   dV += P^T dO,  dS = P * (dO V^T - rowsum(dO * O)),
//   dQ += scale * dS K,  dK += scale * dS^T Q,
// shared over batch x key head so every worker owns its gradient slices.
// ---------------------------------------------------------------------------
namespace {

constexpr int64_t kFlashKvSplit = 512;

inline int64_t flash_q_split(int64_t q_len) {
  return q_len >= 768 ? 256 : (q_len >= 192 ? 64 : 32);
}

// Row-major C[m, n] = alpha * op(A)[m, k] @ op(B)[k, n] + beta * C.
template <typename A>
inline void flash_gemm(bool trans_a, bool trans_b, int64_t m, int64_t n,
                       int64_t k, A alpha, const A* a, int64_t lda,
                       const A* b, int64_t ldb, A beta, A* c, int64_t ldc) {
#if defined(USE_MKL) || defined(USE_BLAS)
  const CBLAS_TRANSPOSE ta = trans_a ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE tb = trans_b ? CblasTrans : CblasNoTrans;
  if constexpr (std::is_same_v<A, float>) {
    cblas_sgemm(CblasRowMajor, ta, tb, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, a,
                static_cast<int>(lda), b, static_cast<int>(ldb), beta, c,
                static_cast<int>(ldc));
  } else {
    cblas_dgemm(CblasRowMajor, ta, tb, static_cast<int>(m),
                static_cast<int>(n), static_cast<int>(k), alpha, a,
                static_cast<int>(lda), b, static_cast<int>(ldb), beta, c,
                static_cast<int>(ldc));
  }
#else
  for (int64_t i = 0; i < m; ++i) {
    A* crow = c + i * ldc;
    if (beta == A(0)) {
      std::fill(crow, crow + n, A(0));
    } else if (beta != A(1)) {
      for (int64_t j = 0; j < n; ++j) crow[j] *= beta;
    }
    for (int64_t p = 0; p < k; ++p) {
      const A av = alpha * (trans_a ? a[p * lda + i] : a[i * lda + p]);
      if (trans_b) {
        for (int64_t j = 0; j < n; ++j) crow[j] += av * b[j * ldb + p];
      } else {
        const A* brow = b + p * ldb;
        for (int64_t j = 0; j < n; ++j) crow[j] += av * brow[j];
      }
    }
  }
#endif
}

inline void flash_vexp(float* x, int64_t n) { vexp_f32(x, x, n); }
inline void flash_vexp(double* x, int64_t n) { vexp_f64(x, x, n); }

// An additive mask broadcast to [mask_b, mask_h, Tq, Skv] in the accumulate
// type, where mask_b is 1 or B and mask_h is 1 or H.
struct FlashMask {
  Tensor data;
  int64_t mask_b = 1;
  int64_t mask_h = 1;

  template <typename A>
  const A* rows(int64_t b, int64_t h, int64_t tq, int64_t skv) const {
    if (!data.defined()) return nullptr;
    const int64_t bi = mask_b > 1 ? b : 0;
    const int64_t hi = mask_h > 1 ? h : 0;
    return data.data_ptr<A>() + (bi * mask_h + hi) * tq * skv;
  }
};

FlashMask flash_prepare_mask(const std::optional<Tensor>& attn_mask,
                             DType origin_dtype, DType acc_dtype, int64_t B,
                             int64_t H, int64_t Tq, int64_t Skv,
                             const char* who) {
  FlashMask out;
  if (!attn_mask.has_value() || !attn_mask->defined() ||
      attn_mask->numel() == 0) {
    return out;
  }
  const Tensor& m = *attn_mask;
  if (m.dtype() != DType::Float32 && m.dtype() != origin_dtype) {
    TP_THROW(ValueError, who,
             ": attn_mask must be float32 or the query dtype");
  }
  if (m.dim() != 2 && m.dim() != 4) {
    TP_THROW(ValueError, who, ": attn_mask dim must be 2 or 4");
  }
  const int64_t rows = m.size(-2), cols = m.size(-1);
  if ((rows != Tq && rows != 1) || (cols != Skv && cols != 1)) {
    TP_THROW(ValueError, who,
             ": attn_mask trailing sizes must be {Tq or 1, Skv or 1}");
  }
  if (m.dim() == 4) {
    if ((m.size(0) != B && m.size(0) != 1) ||
        (m.size(1) != H && m.size(1) != 1)) {
      TP_THROW(ValueError, who,
               ": 4D attn_mask leading sizes must be {B or 1, H or 1}");
    }
    out.mask_b = m.size(0);
    out.mask_h = m.size(1);
  }
  out.data = m.to(acc_dtype)
                 .reshape({out.mask_b, out.mask_h, rows, cols})
                 .expand({out.mask_b, out.mask_h, Tq, Skv})
                 .contiguous();
  return out;
}

struct FlashDims {
  int64_t B, H, Hkv, Tq, Skv, D;
};

template <typename A>
void flash_forward(const A* q, const A* k, const A* v, const FlashMask& mask,
                   A* out, A* lse, const FlashDims& dims, A scale,
                   bool is_causal) {
  const int64_t B = dims.B, H = dims.H, Hkv = dims.Hkv;
  const int64_t Tq = dims.Tq, Skv = dims.Skv, D = dims.D;
  const int64_t q_split = std::min(flash_q_split(Tq), Tq);
  const int64_t kv_split = std::min(kFlashKvSplit, Skv);
  const int64_t q_slices = (Tq + q_split - 1) / q_split;
  const int64_t repeat = H / Hkv;
  const A neg_inf = -std::numeric_limits<A>::infinity();

  // Per-thread scratch: score tile, running max, running sum, value tile.
  const size_t per_thread = static_cast<size_t>(q_split) * kv_split +
                            2 * static_cast<size_t>(q_split) +
                            static_cast<size_t>(q_split) * D;
  const int threads = std::max(1, parallel::get_num_threads());
  std::vector<A> scratch(static_cast<size_t>(threads) * per_thread);

  parallel::parallel_for(0, B * H * q_slices, 1, [&](int64_t begin, int64_t end) {
    A* qk = scratch.data() +
            static_cast<size_t>(parallel::get_thread_num()) * per_thread;
    A* qk_max = qk + q_split * kv_split;
    A* qk_sum = qk_max + q_split;
    A* dst = qk_sum + q_split;
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t slice = idx % q_slices;
      const int64_t bh = idx / q_slices;
      const int64_t b = bh / H, h = bh % H;
      const int64_t hkv = h / repeat;
      const int64_t m = slice * q_split;
      const int64_t qb = std::min(q_split, Tq - m);
      const A* qp = q + (bh * Tq + m) * D;
      const A* kp = k + (b * Hkv + hkv) * Skv * D;
      const A* vp = v + (b * Hkv + hkv) * Skv * D;
      const A* mp = mask.template rows<A>(b, h, Tq, Skv);
      std::fill(qk_max, qk_max + qb, neg_inf);
      std::fill(qk_sum, qk_sum + qb, A(0));
      const int64_t num_keys = is_causal ? std::min(m + qb, Skv) : Skv;
      for (int64_t n = 0; n < num_keys; n += kv_split) {
        const int64_t kvb = std::min(kv_split, Skv - n);
        // scores <- Q K^T
        flash_gemm<A>(false, true, qb, kvb, D, A(1), qp, D, kp + n * D, D,
                      A(0), qk, kvb);
        const bool causal_tail = is_causal && num_keys - n <= kv_split;
        for (int64_t row = 0; row < qb; ++row) {
          A* r = qk + row * kvb;
          if (causal_tail) {
            // Row m + row sees keys [0, m + row]; later columns are closed.
            const int64_t first_closed = std::max<int64_t>(m + row - n + 1, 0);
            for (int64_t c = first_closed; c < kvb; ++c) r[c] = neg_inf;
          }
          A mx = neg_inf;
          if (mp != nullptr) {
            const A* mr = mp + (m + row) * Skv + n;
            for (int64_t c = 0; c < kvb; ++c) {
              r[c] = r[c] * scale + mr[c];
              mx = r[c] > mx ? r[c] : mx;
            }
          } else {
            for (int64_t c = 0; c < kvb; ++c) {
              r[c] *= scale;
              mx = r[c] > mx ? r[c] : mx;
            }
          }
          if (qk_max[row] > mx) mx = qk_max[row];
          if (mx == neg_inf) {
            // Nothing visible so far: exp(-inf - -inf) would be nan.
            std::fill(r, r + kvb, A(0));
            continue;
          }
          for (int64_t c = 0; c < kvb; ++c) r[c] -= mx;
          flash_vexp(r, kvb);
          A sum = A(0);
          for (int64_t c = 0; c < kvb; ++c) sum += r[c];
          // Rebase what earlier key blocks accumulated onto the new max.
          const A carry = std::exp(qk_max[row] - mx);
          qk_sum[row] = sum + carry * qk_sum[row];
          qk_max[row] = mx;
          if (n > 0) {
            A* drow = dst + row * D;
            for (int64_t d = 0; d < D; ++d) drow[d] *= carry;
          }
        }
        // values <- values + P V
        flash_gemm<A>(false, false, qb, D, kvb, A(1), qk, kvb, vp + n * D, D,
                      n == 0 ? A(0) : A(1), dst, D);
      }
      A* op = out + (bh * Tq + m) * D;
      A* lp = lse + bh * Tq + m;
      for (int64_t row = 0; row < qb; ++row) {
        // A fully closed row has sum 0; it yields zeros rather than nan.
        const A mx = qk_max[row] == neg_inf ? A(0) : qk_max[row];
        const A sum = qk_sum[row] == A(0) ? A(1) : qk_sum[row];
        const A inv = A(1) / sum;
        const A* drow = dst + row * D;
        A* orow = op + row * D;
        for (int64_t d = 0; d < D; ++d) orow[d] = drow[d] * inv;
        lp[row] = mx + std::log(sum);
      }
    }
  });
}

template <typename A>
void flash_backward(const A* q, const A* k, const A* v, const A* go,
                    const A* out, const A* lse, const FlashMask& mask, A* dq,
                    A* dk, A* dv, const FlashDims& dims, A scale,
                    bool is_causal) {
  const int64_t B = dims.B, H = dims.H, Hkv = dims.Hkv;
  const int64_t Tq = dims.Tq, Skv = dims.Skv, D = dims.D;
  const int64_t q_split = std::min(flash_q_split(Tq), Tq);
  const int64_t kv_split = std::min(kFlashKvSplit, Skv);
  const int64_t repeat = H / Hkv;

  // Per-thread scratch: probability tile, score-gradient tile, row sums.
  const size_t per_thread = 2 * static_cast<size_t>(q_split) * kv_split +
                            static_cast<size_t>(q_split);
  const int threads = std::max(1, parallel::get_num_threads());
  std::vector<A> scratch(static_cast<size_t>(threads) * per_thread);

  parallel::parallel_for(0, B * Hkv, 1, [&](int64_t begin, int64_t end) {
    A* attn = scratch.data() +
              static_cast<size_t>(parallel::get_thread_num()) * per_thread;
    A* gattn = attn + q_split * kv_split;
    A* dsum = gattn + q_split * kv_split;
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t b = idx / Hkv, hkv = idx % Hkv;
      const A* kp = k + idx * Skv * D;
      const A* vp = v + idx * Skv * D;
      A* dkp = dk + idx * Skv * D;
      A* dvp = dv + idx * Skv * D;
      for (int64_t rep = 0; rep < repeat; ++rep) {
        const int64_t h = hkv * repeat + rep;
        const int64_t bh = b * H + h;
        const A* mp = mask.template rows<A>(b, h, Tq, Skv);
        for (int64_t m = 0; m < Tq; m += q_split) {
          const int64_t qb = std::min(q_split, Tq - m);
          const A* qp = q + (bh * Tq + m) * D;
          const A* gp = go + (bh * Tq + m) * D;
          const A* op = out + (bh * Tq + m) * D;
          const A* lp = lse + bh * Tq + m;
          A* dqp = dq + (bh * Tq + m) * D;
          // dsum <- rowsum(dO * O)
          for (int64_t row = 0; row < qb; ++row) {
            const A* grow = gp + row * D;
            const A* orow = op + row * D;
            A acc = A(0);
            for (int64_t d = 0; d < D; ++d) acc += grow[d] * orow[d];
            dsum[row] = acc;
          }
          const int64_t num_keys = is_causal ? std::min(m + qb, Skv) : Skv;
          for (int64_t n = 0; n < num_keys; n += kv_split) {
            const int64_t kvb = std::min(kv_split, Skv - n);
            // attn <- scale * Q K^T
            flash_gemm<A>(false, true, qb, kvb, D, scale, qp, D, kp + n * D,
                          D, A(0), attn, kvb);
            const bool causal_tail = is_causal && num_keys - n <= kv_split;
            for (int64_t row = 0; row < qb; ++row) {
              A* r = attn + row * kvb;
              const A shift = lp[row];
              if (mp != nullptr) {
                const A* mr = mp + (m + row) * Skv + n;
                for (int64_t c = 0; c < kvb; ++c) r[c] = r[c] + mr[c] - shift;
              } else {
                for (int64_t c = 0; c < kvb; ++c) r[c] -= shift;
              }
              // attn <- exp(attn - lse): the forward probabilities.
              flash_vexp(r, kvb);
              if (causal_tail) {
                const int64_t first_closed =
                    std::max<int64_t>(m + row - n + 1, 0);
                for (int64_t c = first_closed; c < kvb; ++c) r[c] = A(0);
              }
            }
            // dV <- dV + P^T dO
            flash_gemm<A>(true, false, kvb, D, qb, A(1), attn, kvb, gp, D,
                          A(1), dvp + n * D, D);
            // gattn <- dO V^T
            flash_gemm<A>(false, true, qb, kvb, D, A(1), gp, D, vp + n * D, D,
                          A(0), gattn, kvb);
            // gattn <- P * (gattn - dsum)
            for (int64_t row = 0; row < qb; ++row) {
              const A* ar = attn + row * kvb;
              A* gr = gattn + row * kvb;
              const A ds = dsum[row];
              for (int64_t c = 0; c < kvb; ++c) gr[c] = ar[c] * (gr[c] - ds);
            }
            // dQ <- dQ + scale * gattn K
            flash_gemm<A>(false, false, qb, D, kvb, scale, gattn, kvb,
                          kp + n * D, D, A(1), dqp, D);
            // dK <- dK + scale * gattn^T Q
            flash_gemm<A>(true, false, kvb, D, qb, scale, gattn, kvb, qp, D,
                          A(1), dkp + n * D, D);
          }
        }
      }
    }
  });
}

FlashDims flash_check_shapes(const Tensor& query, const Tensor& key,
                             const Tensor& value, double dropout_p,
                             const char* who) {
  const DType dtype = query.dtype();
  if (dtype != DType::Float32 && dtype != DType::Float64 &&
      dtype != DType::Float16 && dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError, who,
             ": expected float32/float64/float16/bfloat16");
  }
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    TP_THROW(ValueError, who, ": accept only 4D inputs of shape {B, H, T, K}");
  }
  if (dropout_p != 0.0) {
    TP_THROW(ValueError, who, ": dropout > 0 is not supported");
  }
  if (key.dtype() != dtype || value.dtype() != dtype) {
    TP_THROW(ValueError, who, ": Q/K/V must share one dtype");
  }
  if (value.size(3) != query.size(3) || key.size(3) != value.size(3)) {
    TP_THROW(ValueError, who, ": Q/K/V must share the head size");
  }
  FlashDims dims{query.size(0), query.size(1), key.size(1),
                 query.size(2), key.size(2),   query.size(3)};
  if (key.size(0) != dims.B || value.size(0) != dims.B ||
      value.size(1) != dims.Hkv || value.size(2) != dims.Skv) {
    TP_THROW(ValueError, who, ": key/value shapes must match {B, Hkv, S, D}");
  }
  if (dims.Hkv != dims.H && (dims.Hkv == 0 || dims.H % dims.Hkv != 0)) {
    TP_THROW(ValueError, who,
             ": the query head count must be a multiple of the key head count");
  }
  if (dims.Tq * dims.Skv > static_cast<int64_t>(INT32_MAX) ||
      dims.Skv * dims.D > static_cast<int64_t>(INT32_MAX) ||
      dims.Tq * dims.D > static_cast<int64_t>(INT32_MAX)) {
    TP_THROW(RuntimeError, who, ": shape too large for BLAS ints");
  }
  return dims;
}

inline bool flash_is_empty(const FlashDims& d) {
  return d.B == 0 || d.H == 0 || d.Hkv == 0 || d.Tq == 0 || d.Skv == 0 ||
         d.D == 0;
}

} // namespace

// Fused kernel for the `_scaled_dot_product_flash_attention_for_cpu`
// dispatcher contract: 4D [B, H, Tq, D] query against [B, Hkv, Skv, D]
// key/value (H a multiple of Hkv), optional additive 2D/4D mask, causal flag,
// explicit scale.  Returns the attention output plus the per-row logsumexp
// [B, H, Tq] in the accumulate dtype that the backward kernel replays.  A row
// with no visible key yields zeros.  Dropout is rejected.
std::tuple<Tensor, Tensor> sdpa_flash_cpu_kernel(
    const Tensor& query, const Tensor& key, const Tensor& value,
    double dropout_p, bool is_causal, const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  static const char* who = "sdpa cpu flash";
  const FlashDims dims = flash_check_shapes(query, key, value, dropout_p, who);
  const DType origin_dtype = query.dtype();
  const DType acc_dtype = origin_dtype == DType::Float64 ? DType::Float64
                                                         : DType::Float32;
  const FlashMask mask = flash_prepare_mask(attn_mask, origin_dtype, acc_dtype,
                                            dims.B, dims.H, dims.Tq, dims.Skv,
                                            who);
  if (flash_is_empty(dims)) {
    return {Tensor::zeros({dims.B, dims.H, dims.Tq, dims.D}, origin_dtype,
                          query.device()),
            Tensor::zeros({dims.B, dims.H, dims.Tq}, acc_dtype,
                          query.device())};
  }
  Tensor q = query.to(acc_dtype).contiguous();
  Tensor k = key.to(acc_dtype).contiguous();
  Tensor v = value.to(acc_dtype).contiguous();
  const double scale_val =
      scale.has_value() ? *scale
                        : 1.0 / std::sqrt(static_cast<double>(dims.D));

  Tensor out = Tensor::empty({dims.B, dims.H, dims.Tq, dims.D}, acc_dtype,
                             q.device());
  Tensor lse = Tensor::empty({dims.B, dims.H, dims.Tq}, acc_dtype, q.device());
  if (acc_dtype == DType::Float64) {
    flash_forward<double>(q.data_ptr<double>(), k.data_ptr<double>(),
                          v.data_ptr<double>(), mask, out.data_ptr<double>(),
                          lse.data_ptr<double>(), dims, scale_val, is_causal);
  } else {
    flash_forward<float>(q.data_ptr<float>(), k.data_ptr<float>(),
                         v.data_ptr<float>(), mask, out.data_ptr<float>(),
                         lse.data_ptr<float>(), dims,
                         static_cast<float>(scale_val), is_causal);
  }
  if (acc_dtype != origin_dtype) out = out.to(origin_dtype);
  return {std::move(out), std::move(lse)};
}

// Replay partner of sdpa_flash_cpu_kernel: rebuilds the probabilities from
// the saved logsumexp and emits dQ/dK/dV in the shapes of Q/K/V.
std::tuple<Tensor, Tensor, Tensor> sdpa_flash_backward_cpu_kernel(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    double dropout_p, bool is_causal, const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  if (!grad_out.defined()) {
    return {Tensor(), Tensor(), Tensor()};
  }
  static const char* who = "sdpa cpu flash backward";
  const FlashDims dims = flash_check_shapes(query, key, value, dropout_p, who);
  if (grad_out.dim() != 4 || grad_out.size(0) != dims.B ||
      grad_out.size(1) != dims.H || grad_out.size(2) != dims.Tq ||
      grad_out.size(3) != dims.D || out.dim() != 4 ||
      out.numel() != grad_out.numel() || logsumexp.dim() != 3 ||
      logsumexp.numel() != dims.B * dims.H * dims.Tq) {
    TP_THROW(ValueError, who,
             ": grad_out/out must be {B, H, Tq, D} and logsumexp {B, H, Tq}");
  }
  const DType origin_dtype = query.dtype();
  const DType acc_dtype = origin_dtype == DType::Float64 ? DType::Float64
                                                         : DType::Float32;
  const FlashMask mask = flash_prepare_mask(attn_mask, origin_dtype, acc_dtype,
                                            dims.B, dims.H, dims.Tq, dims.Skv,
                                            who);
  Tensor d_q = Tensor::zeros({dims.B, dims.H, dims.Tq, dims.D}, acc_dtype,
                             query.device());
  Tensor d_k = Tensor::zeros({dims.B, dims.Hkv, dims.Skv, dims.D}, acc_dtype,
                             query.device());
  Tensor d_v = Tensor::zeros({dims.B, dims.Hkv, dims.Skv, dims.D}, acc_dtype,
                             query.device());
  if (!flash_is_empty(dims)) {
    Tensor q = query.to(acc_dtype).contiguous();
    Tensor k = key.to(acc_dtype).contiguous();
    Tensor v = value.to(acc_dtype).contiguous();
    // A broadcast gradient (zero strides) is materialized before the matrix
    // products read it.
    Tensor go = grad_out.to(acc_dtype).contiguous();
    Tensor o = out.to(acc_dtype).contiguous();
    Tensor lse = logsumexp.to(acc_dtype).contiguous();
    const double scale_val =
        scale.has_value() ? *scale
                          : 1.0 / std::sqrt(static_cast<double>(dims.D));
    if (acc_dtype == DType::Float64) {
      flash_backward<double>(
          q.data_ptr<double>(), k.data_ptr<double>(), v.data_ptr<double>(),
          go.data_ptr<double>(), o.data_ptr<double>(), lse.data_ptr<double>(),
          mask, d_q.data_ptr<double>(), d_k.data_ptr<double>(),
          d_v.data_ptr<double>(), dims, scale_val, is_causal);
    } else {
      flash_backward<float>(
          q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
          go.data_ptr<float>(), o.data_ptr<float>(), lse.data_ptr<float>(),
          mask, d_q.data_ptr<float>(), d_k.data_ptr<float>(),
          d_v.data_ptr<float>(), dims, static_cast<float>(scale_val),
          is_causal);
    }
  }
  if (acc_dtype != origin_dtype) {
    d_q = d_q.to(origin_dtype);
    d_k = d_k.to(origin_dtype);
    d_v = d_v.to(origin_dtype);
  }
  return {std::move(d_q), std::move(d_k), std::move(d_v)};
}


TENSORPLAY_LIBRARY_IMPL(CPU, TransformersKernels) {
  m.impl("scaled_dot_product_attention", sdpa_kernel_cpu);
  m.impl("scaled_dot_product_attention_backward", sdpa_backward_kernel_cpu);
  m.impl("_scaled_dot_product_attention_math", sdpa_math_kernel_cpu);
  m.impl("_scaled_dot_product_flash_attention_for_cpu", sdpa_flash_cpu_kernel);
  m.impl("_scaled_dot_product_flash_attention_for_cpu_backward",
         sdpa_flash_backward_cpu_kernel);
  m.impl("_native_multi_head_attention", native_multi_head_attention_cpu);
  m.impl("_fused_sdp_choice", fused_sdp_choice_cpu);
  m.impl("grouped_mm", grouped_mm_cpu);
  m.impl("rotary_embedding", rotary_embedding_cpu);
  m.impl("fused_rope", fused_rope_cpu);
}

} // namespace cpu
} // namespace tensorplay
