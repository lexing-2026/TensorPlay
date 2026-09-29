// CPU scaled-dot-product attention kernels.
// lives under transformers/, not an "llm" grab-bag.
//
// Forward f32/f16/bf16 path: BLAS sgemm for QK^T and PV, fused causal-prefix
// row softmax with runtime-dispatched libmvec vector exp (AVX-512 16-wide ->
// AVX2 8-wide -> scalar).  f64 keeps a serial double reference oracle.

#include <algorithm>
#include <type_traits>
#include <vector>

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "GradMode.h"
#include "LinearAlgebraNames.h"
#include "Parallel.h"

#include "../composite/AttentionComposite.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

// Vectorized expf: AVX-512 16 lanes -> AVX2 8 lanes -> scalar libm tail.
// The target attribute is required because the TU compiles with base
// x86-64 flags (repo convention: per-function targets, cf. VecUnary.h).
#if defined(TP_SDPA_SLEEF)
__attribute__((target("avx2,avx512f")))
#endif
void vexp_f32(const float* x, float* y, int64_t n) {
  int64_t i = 0;
#if defined(TP_SDPA_SLEEF)
  const bool avx512 = __builtin_cpu_supports("avx512f");
  if (avx512) {
    for (; i + 16 <= n; i += 16)
      _mm512_storeu_ps(y + i, tensorplay::tpsleef::exp(_mm512_loadu_ps(x + i)));
  } else if (__builtin_cpu_supports("avx2")) {
    for (; i + 8 <= n; i += 8)
      _mm256_storeu_ps(y + i, tensorplay::tpsleef::exp(_mm256_loadu_ps(x + i)));
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
             c10_style_dtype_name(self.dtype()), " != ",
             c10_style_dtype_name(mat2.dtype()));
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

// Fused flash-style kernel for the `_scaled_dot_product_attention_for_cpu`
// dispatcher contract: 4D [B, H, Tq, D], optional 2D/4D float mask, causal
// flag, explicit scale.  Returns the attention output plus the per-row
// logsumexp (B, H, Tq) in the accumulate dtype that the backward kernel
// replays.  Dropout is rejected, matching the CPU flash contract.
std::tuple<Tensor, Tensor> sdpa_flash_cpu_kernel(
    const Tensor& query, const Tensor& key, const Tensor& value,
    double dropout_p, bool is_causal, const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  const DType origin_dtype = query.dtype();
  if (origin_dtype != DType::Float32 && origin_dtype != DType::Float64 &&
      origin_dtype != DType::Float16 && origin_dtype != DType::BFloat16) {
    TP_THROW(NotImplementedError,
             "sdpa cpu flash: expected float32/float64/float16/bfloat16");
  }
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    TP_THROW(ValueError,
             "sdpa cpu flash: accept only 4D inputs of shape {B, H, T, K}");
  }
  if (dropout_p != 0.0) {
    TP_THROW(ValueError, "sdpa cpu flash: dropout > 0 is not supported");
  }
  if (value.size(3) != query.size(3) || key.size(3) != value.size(3)) {
    TP_THROW(ValueError, "sdpa cpu flash: Q/K/V must share the head size");
  }
  if (attn_mask.has_value()) {
    const Tensor& m = *attn_mask;
    if (m.dtype() != DType::Float32 && m.dtype() != origin_dtype &&
        m.dtype() != DType::Float64) {
      TP_THROW(ValueError,
               "sdpa cpu flash: attn_mask must be float or the query dtype");
    }
    if (m.dim() != 2 && m.dim() != 4) {
      TP_THROW(ValueError, "sdpa cpu flash: attn_mask dim must be 2 or 4");
    }
    if (m.dim() == 4 && (m.size(0) != query.size(0) ||
                         m.size(1) != query.size(1) ||
                         m.size(2) != query.size(2) ||
                         m.size(3) != key.size(2))) {
      TP_THROW(ValueError,
               "sdpa cpu flash: 4D attn_mask must match {B, H, Tq, Skv}");
    }
  }
  const int64_t B = query.size(0), H = query.size(1);
  const int64_t Tq = query.size(2), Skv = key.size(2), D = query.size(3);
  if (key.size(0) != B || key.size(1) != H || value.size(0) != B ||
      value.size(1) != H || value.size(2) != Skv) {
    TP_THROW(ValueError,
             "sdpa cpu flash: key/value shapes must match {B, H, S, D}");
  }
  if (Tq * Skv > static_cast<int64_t>(INT32_MAX) ||
      Skv * D > static_cast<int64_t>(INT32_MAX) ||
      Tq * D > static_cast<int64_t>(INT32_MAX)) {
    TP_THROW(RuntimeError, "sdpa cpu flash: shape too large for BLAS ints");
  }

  const DType acc_dtype = origin_dtype == DType::Float64 ? DType::Float64
                                                         : DType::Float32;
  Tensor q = query.to(acc_dtype).contiguous();
  Tensor k = key.to(acc_dtype).contiguous();
  Tensor v = value.to(acc_dtype).contiguous();
  Tensor mask_f;
  if (attn_mask.has_value()) {
    mask_f = attn_mask->to(acc_dtype).contiguous();
  }
  const double scale_val = scale.has_value()
                               ? *scale
                               : 1.0 / std::sqrt(static_cast<double>(D));

  Tensor out = Tensor::empty({B, H, Tq, D}, acc_dtype, q.device());
  Tensor lse = Tensor::empty({B, H, Tq}, acc_dtype, q.device());

  const bool has_mask2d = mask_f.defined() && mask_f.dim() == 2;
  const bool has_mask4d = mask_f.defined() && mask_f.dim() == 4;
  std::vector<double> scores;

  auto run_head = [&](auto qd, auto kd, auto vd, auto od, auto ld, auto md) {
    using A = std::remove_pointer_t<decltype(qd)>;
    for (int64_t bh = 0; bh < B * H; ++bh) {
      const A* qh = qd + bh * Tq * D;
      const A* kh = kd + bh * Skv * D;
      const A* vh = vd + bh * Skv * D;
      A* oh = od + bh * Tq * D;
      A* lh = ld + bh * Tq;
      const A* mh = has_mask4d ? md + bh * Tq * Skv : nullptr;
      const A* m2 = has_mask2d ? md : nullptr;
      scores.resize(static_cast<size_t>(Tq) * Skv);
      for (int64_t t = 0; t < Tq; ++t) {
        const int64_t visible = is_causal ? std::min(t + 1, Skv) : Skv;
        double mx = -INFINITY;
        for (int64_t j = 0; j < Skv; ++j) {
          double s = -INFINITY;
          if (j < visible) {
            s = 0.0;
            for (int64_t d = 0; d < D; ++d) s += static_cast<double>(qh[t * D + d]) * kh[j * D + d];
            s *= scale_val;
            if (has_mask4d) s += static_cast<double>(mh[t * Skv + j]);
            else if (has_mask2d) s += static_cast<double>(m2[t * Skv + j]);
          }
          scores[static_cast<size_t>(t) * Skv + j] = s;
          mx = std::max(mx, s);
        }
        double total = 0.0;
        for (int64_t j = 0; j < Skv; ++j) {
          double e = std::exp(scores[static_cast<size_t>(t) * Skv + j] - mx);
          scores[static_cast<size_t>(t) * Skv + j] = e;
          total += e;
        }
        lh[t] = static_cast<A>(mx + std::log(total));
        for (int64_t d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < Skv; ++j)
            acc += scores[static_cast<size_t>(t) * Skv + j] *
                   static_cast<double>(vh[j * D + d]);
          oh[t * D + d] = static_cast<A>(acc / total);
        }
      }
    }
  };

  if (acc_dtype == DType::Float64) {
    run_head(q.data_ptr<double>(), k.data_ptr<double>(), v.data_ptr<double>(),
             out.data_ptr<double>(), lse.data_ptr<double>(),
             mask_f.defined() ? mask_f.data_ptr<double>()
                              : static_cast<const double*>(nullptr));
  } else {
    run_head(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
             out.data_ptr<float>(), lse.data_ptr<float>(),
             mask_f.defined() ? mask_f.data_ptr<float>()
                              : static_cast<const float*>(nullptr));
  }
  if (acc_dtype != origin_dtype) out = out.to(origin_dtype);
  return {std::move(out), std::move(lse)};
}

// Replay partner of sdpa_flash_cpu_kernel: rebuilds the probabilities from
// the saved logsumexp and emits dQ/dK/dV.  dS = p * (dP - rowsum(dP * p))
// with dP = dO @ V^T; dQ = scale * dS @ K; dK = scale * dS^T @ Q;
// dV = P^T @ dO.  Masked/causal positions carry p = 0 so no extra masking
// pass is needed.
std::tuple<Tensor, Tensor, Tensor> sdpa_flash_backward_cpu_kernel(
    const Tensor& grad_out, const Tensor& query, const Tensor& key,
    const Tensor& value, const Tensor& out, const Tensor& logsumexp,
    double dropout_p, bool is_causal, const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  (void)out;
  if (!grad_out.defined()) {
    return {Tensor(), Tensor(), Tensor()};
  }
  if (dropout_p != 0.0) {
    TP_THROW(ValueError,
             "sdpa cpu flash backward: dropout > 0 is not supported");
  }
  const DType origin_dtype = query.dtype();
  const int64_t B = query.size(0), H = query.size(1);
  const int64_t Tq = query.size(2), Skv = key.size(2), D = query.size(3);
  if (Tq * Skv > static_cast<int64_t>(INT32_MAX) ||
      Skv * D > static_cast<int64_t>(INT32_MAX) ||
      Tq * D > static_cast<int64_t>(INT32_MAX)) {
    TP_THROW(RuntimeError,
             "sdpa cpu flash backward: shape too large for BLAS ints");
  }
  const DType acc_dtype = origin_dtype == DType::Float64 ? DType::Float64
                                                         : DType::Float32;
  Tensor q = query.to(acc_dtype).contiguous();
  Tensor k = key.to(acc_dtype).contiguous();
  Tensor v = value.to(acc_dtype).contiguous();
  Tensor go = grad_out.to(acc_dtype).contiguous();
  Tensor lse = logsumexp.to(acc_dtype).contiguous();
  Tensor mask_f;
  if (attn_mask.has_value() && attn_mask->defined()) {
    mask_f = attn_mask->to(acc_dtype).contiguous();
  }
  const double scale_val = scale.has_value()
                               ? *scale
                               : 1.0 / std::sqrt(static_cast<double>(D));
  const bool has_mask2d = mask_f.defined() && mask_f.dim() == 2;
  const bool has_mask4d = mask_f.defined() && mask_f.dim() == 4;

  Tensor d_q = Tensor::zeros({B, H, Tq, D}, acc_dtype, q.device());
  Tensor d_k = Tensor::zeros({B, H, Skv, D}, acc_dtype, q.device());
  Tensor d_v = Tensor::zeros({B, H, Skv, D}, acc_dtype, q.device());
  std::vector<double> scores;
  std::vector<double> dp;
  std::vector<double> ds;

  auto run_head = [&](auto qd, auto kd, auto vd, auto god, auto ld, auto md) {
    using A = std::remove_pointer_t<decltype(qd)>;
    for (int64_t bh = 0; bh < B * H; ++bh) {
      const A* qh = qd + bh * Tq * D;
      const A* kh = kd + bh * Skv * D;
      const A* vh = vd + bh * Skv * D;
      const A* gh = god + bh * Tq * D;
      const A* lh = ld + bh * Tq;
      const A* mh = has_mask4d ? md + bh * Tq * Skv : nullptr;
      const A* m2 = has_mask2d ? md : nullptr;
      A* dqh = d_q.data_ptr<A>() + bh * Tq * D;
      A* dkh = d_k.data_ptr<A>() + bh * Skv * D;
      A* dvh = d_v.data_ptr<A>() + bh * Skv * D;
      scores.resize(static_cast<size_t>(Tq) * Skv);
      dp.resize(static_cast<size_t>(Tq) * Skv);
      ds.resize(static_cast<size_t>(Tq) * Skv);
      // Rebuild logits; p = exp(s - lse) recovers the softmax probabilities
      // with masked/causal entries at -inf -> 0.
      for (int64_t t = 0; t < Tq; ++t) {
        const int64_t visible = is_causal ? std::min(t + 1, Skv) : Skv;
        for (int64_t j = 0; j < Skv; ++j) {
          double s = -INFINITY;
          if (j < visible) {
            s = 0.0;
            for (int64_t d = 0; d < D; ++d) s += static_cast<double>(qh[t * D + d]) * kh[j * D + d];
            s *= scale_val;
            if (has_mask4d) s += static_cast<double>(mh[t * Skv + j]);
            else if (has_mask2d) s += static_cast<double>(m2[t * Skv + j]);
          }
          scores[static_cast<size_t>(t) * Skv + j] = s;
        }
      }
      auto prob = [&](int64_t t, int64_t j) -> double {
        return std::exp(scores[static_cast<size_t>(t) * Skv + j] -
                        static_cast<double>(lh[t]));
      };
      // Softmax backward: dS = p * (dP - rowsum(dP * p)), dP = dO @ V^T.
      for (int64_t t = 0; t < Tq; ++t) {
        double row_dot = 0.0;
        for (int64_t j = 0; j < Skv; ++j) {
          double dot = 0.0;
          for (int64_t d = 0; d < D; ++d)
            dot += static_cast<double>(gh[t * D + d]) *
                   static_cast<double>(vh[j * D + d]);
          dp[static_cast<size_t>(t) * Skv + j] = dot;
          row_dot += dot * prob(t, j);
        }
        for (int64_t j = 0; j < Skv; ++j) {
          ds[static_cast<size_t>(t) * Skv + j] =
              prob(t, j) * (dp[static_cast<size_t>(t) * Skv + j] - row_dot) *
              scale_val;
        }
      }
      // dQ = dS @ K
      for (int64_t t = 0; t < Tq; ++t) {
        for (int64_t d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int64_t j = 0; j < Skv; ++j)
            acc += ds[static_cast<size_t>(t) * Skv + j] *
                   static_cast<double>(kh[j * D + d]);
          dqh[t * D + d] = static_cast<A>(acc);
        }
      }
      // dK = dS^T @ Q ; dV = P^T @ dO (per key position j)
      for (int64_t j = 0; j < Skv; ++j) {
        for (int64_t d = 0; d < D; ++d) {
          double kacc = 0.0, vacc = 0.0;
          for (int64_t t = 0; t < Tq; ++t) {
            const double p = prob(t, j);
            kacc += ds[static_cast<size_t>(t) * Skv + j] *
                    static_cast<double>(qh[t * D + d]);
            vacc += p * static_cast<double>(gh[t * D + d]);
          }
          dkh[j * D + d] = static_cast<A>(kacc);
          dvh[j * D + d] = static_cast<A>(vacc);
        }
      }
    }
  };

  if (acc_dtype == DType::Float64) {
    run_head(q.data_ptr<double>(), k.data_ptr<double>(), v.data_ptr<double>(),
             go.data_ptr<double>(), lse.data_ptr<double>(),
             mask_f.defined() ? mask_f.data_ptr<double>()
                              : static_cast<const double*>(nullptr));
  } else {
    run_head(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
             go.data_ptr<float>(), lse.data_ptr<float>(),
             mask_f.defined() ? mask_f.data_ptr<float>()
                              : static_cast<const float*>(nullptr));
  }
  return {d_q.to(origin_dtype), d_k.to(origin_dtype), d_v.to(origin_dtype)};
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
