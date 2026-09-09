// Rotary position embedding (RoPE), fused RoPE and grouped GEMM entry
// points for the CUDA backend.  Split from AttentionKernels.cu (which keeps
// the scaled-dot-product attention kernels) so LLM-side edits do not
// recompile the WMMA attention paths.  The composite dispatcher entries
// resolve through AttentionComposite.h -> TPXOpsGenerated.h; no local
// re-declarations.

#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Allocator.h"
#include "CudaGemm.h"
#include "GradMode.h"
#include "../composite/AttentionComposite.h"

#include <cuda_runtime.h>

#include <optional>
#include <tuple>
#include <vector>


// nvcc mis-resolves the unqualified `tensorplay` in TP_THROW after the
// namespace reopen above (it synthesizes a phantom
// tensorplay::cuda::tensorplay member), so pin the throw target explicitly.

namespace tensorplay {
namespace cuda {

// nvcc mis-resolves the unqualified `tensorplay` in TP_THROW after the
// namespace reopen above (it synthesizes a phantom
// tensorplay::cuda::tensorplay member), so pin the throw target explicitly.
#if defined(TP_THROW)
#undef TP_THROW
#endif
#define TP_THROW(err_type, ...)                                               \
    throw ::tensorplay::err_type(                                             \
        {__FILE__, __func__, __LINE__},                                       \
        ::tensorplay::detail::format_msg(__VA_ARGS__))

// flash-attention's rotary Q/K handling), so keep these kernels beside SDPA
// and grouped attention rather than under an LLM-specific translation unit.
namespace {

#define TP_CUDA_CHECK(condition)                                             \
  do {                                                                       \
    cudaError_t error = condition;                                           \
    if (error != cudaSuccess) {                                              \
      TP_THROW(RuntimeError,                                                 \
               std::string("CUDA Error: ") + cudaGetErrorString(error));     \
    }                                                                        \
  } while (0)

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

template <typename T, typename C, typename Acc>
__device__ inline void rotate_pair(const T* input, T* output, const C* cos,
                                   const C* sin, int64_t input_offset,
                                   int64_t table_offset) {
  const Acc x0 = static_cast<Acc>(input[input_offset]);
  const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
  const Acc c = static_cast<Acc>(cos[table_offset]);
  const Acc s = static_cast<Acc>(sin[table_offset]);
  output[input_offset] = static_cast<T>(x0 * c - x1 * s);
  output[input_offset + 1] = static_cast<T>(x0 * s + x1 * c);
}

template <typename T, typename C, typename Acc>
__global__ void rotary_embedding_kernel(
    const T* input, T* output, const C* cos, const C* sin, int64_t pairs,
    int64_t tokens, int64_t half_dim, int64_t table_rows,
    int64_t position_offset) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t pair = first; pair < pairs; pair += stride) {
    const int64_t row = pair / half_dim;
    const int64_t pair_in_row = pair - row * half_dim;
    const int64_t token = row % tokens;
    const int64_t table_row = table_rows == 1 ? 0 : position_offset + token;
    rotate_pair<T, C, Acc>(
        input, output, cos, sin, row * (2 * half_dim) + 2 * pair_in_row,
        table_row * half_dim + pair_in_row);
  }
}

template <typename T, typename C, typename Acc>
__global__ void fused_rope_kernel(
    const T* query, T* query_out, const T* key, T* key_out,
    const C* cos, const C* sin, int64_t query_pairs, int64_t key_pairs,
    int64_t query_tokens, int64_t key_tokens, int64_t half_dim,
    int64_t table_rows, int64_t position_offset) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t total_pairs = query_pairs + key_pairs;
  for (int64_t pair = first; pair < total_pairs; pair += stride) {
    const T* input = query;
    T* output = query_out;
    int64_t local_pair = pair;
    int64_t tokens = query_tokens;
    if (pair >= query_pairs) {
      local_pair -= query_pairs;
      input = key;
      output = key_out;
      tokens = key_tokens;
    }
    const int64_t row = local_pair / half_dim;
    const int64_t pair_in_row = local_pair - row * half_dim;
    const int64_t token = row % tokens;
    const int64_t table_row = table_rows == 1 ? 0 : position_offset + token;
    rotate_pair<T, C, Acc>(
        input, output, cos, sin, row * (2 * half_dim) + 2 * pair_in_row,
        table_row * half_dim + pair_in_row);
  }
}

// The transformer path presents Q/K as a [B,H,T,D] view over a contiguous
// [B,T,H,D] linear result.  Its innermost D dimension is still contiguous, so
// RoPE can read that view directly instead of first materializing Q and K.
// Keep the generic flat kernel above for other ranks/layouts.
template <typename T, typename C, typename Acc>
__global__ void fused_rope_4d_strided_kernel(
    const T* query, T* query_out, const T* key, T* key_out,
    const C* cos, const C* sin, int64_t query_pairs, int64_t key_pairs,
    int64_t query_tokens, int64_t key_tokens, int64_t half_dim,
    int64_t table_rows, int64_t position_offset,
    int64_t query_batch_stride, int64_t query_head_stride,
    int64_t query_token_stride, int64_t query_out_batch_stride,
    int64_t query_out_head_stride, int64_t query_out_token_stride,
    int64_t query_heads, int64_t key_batch_stride, int64_t key_head_stride,
    int64_t key_token_stride, int64_t key_out_batch_stride,
    int64_t key_out_head_stride, int64_t key_out_token_stride,
    int64_t key_heads) {
  const int64_t first = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t total_pairs = query_pairs + key_pairs;
  for (int64_t pair = first; pair < total_pairs; pair += stride) {
    const T* input = query;
    T* output = query_out;
    int64_t local_pair = pair;
    int64_t tokens = query_tokens;
    int64_t batch_stride = query_batch_stride;
    int64_t head_stride = query_head_stride;
    int64_t token_stride = query_token_stride;
    int64_t out_batch_stride = query_out_batch_stride;
    int64_t out_head_stride = query_out_head_stride;
    int64_t out_token_stride = query_out_token_stride;
    int64_t heads = query_heads;
    if (pair >= query_pairs) {
      local_pair -= query_pairs;
      input = key;
      output = key_out;
      tokens = key_tokens;
      batch_stride = key_batch_stride;
      head_stride = key_head_stride;
      token_stride = key_token_stride;
      out_batch_stride = key_out_batch_stride;
      out_head_stride = key_out_head_stride;
      out_token_stride = key_out_token_stride;
      heads = key_heads;
    }
    const int64_t row = local_pair / half_dim;
    const int64_t pair_in_row = local_pair - row * half_dim;
    const int64_t token = row % tokens;
    const int64_t outer = row / tokens;
    const int64_t batch = outer / heads;
    const int64_t head = outer - batch * heads;
    const int64_t input_offset = batch * batch_stride +
                                 head * head_stride +
                                 token * token_stride +
                                 2 * pair_in_row;
    const int64_t output_offset = batch * out_batch_stride +
                                  head * out_head_stride +
                                  token * out_token_stride +
                                  2 * pair_in_row;
    const int64_t table_row = table_rows == 1 ? 0 : position_offset + token;
    const Acc x0 = static_cast<Acc>(input[input_offset]);
    const Acc x1 = static_cast<Acc>(input[input_offset + 1]);
    const Acc c = static_cast<Acc>(cos[table_row * half_dim + pair_in_row]);
    const Acc s = static_cast<Acc>(sin[table_row * half_dim + pair_in_row]);
    output[output_offset] = static_cast<T>(x0 * c - x1 * s);
    output[output_offset + 1] = static_cast<T>(x0 * s + x1 * c);
  }
}

inline unsigned rope_blocks(int64_t n) {
  constexpr int64_t kThreads = 256;
  return static_cast<unsigned>(std::min<int64_t>(
      (n + kThreads - 1) / kThreads, 65535));
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
  if (output.numel() == 0) return output;
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  rotary_embedding_kernel<T, C, Acc><<<
      rope_blocks(input_c.numel() / 2), 256, 0,
      getCurrentCUDAStream().stream()>>>(
      input_c.data_ptr<T>(), output.data_ptr<T>(), cos_c.data_ptr<C>(),
      sin_c.data_ptr<C>(), input_c.numel() / 2, tokens, table.half_dim,
      table.rows, position_offset);
  TP_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename T, typename C>
std::tuple<Tensor, Tensor> rope_pair_typed(
    const Tensor& query, const Tensor& key, const Tensor& cos,
    const Tensor& sin, const RopeTable& table, int64_t query_tokens,
    int64_t key_tokens, int64_t position_offset) {
  const bool strided_4d = query.dim() == 4 && key.dim() == 4 &&
                          query.stride(3) == 1 && key.stride(3) == 1;
  Tensor query_c = strided_4d || query.is_contiguous()
      ? query : query.contiguous();
  Tensor key_c = strided_4d || key.is_contiguous()
      ? key : key.contiguous();
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
  if (query_pairs + key_pairs == 0) {
    return std::make_tuple(query_out, key_out);
  }
  using Acc = std::conditional_t<std::is_same_v<T, double>, double, float>;
  if (strided_4d) {
    fused_rope_4d_strided_kernel<T, C, Acc><<<
        rope_blocks(query_pairs + key_pairs), 256, 0,
        getCurrentCUDAStream().stream()>>>(
        query_c.data_ptr<T>(), query_out.data_ptr<T>(), key_c.data_ptr<T>(),
        key_out.data_ptr<T>(), cos_c.data_ptr<C>(), sin_c.data_ptr<C>(),
        query_pairs, key_pairs, query_tokens, key_tokens, table.half_dim,
        table.rows, position_offset,
        query_c.stride(0), query_c.stride(1), query_c.stride(2),
        query_out.stride(0), query_out.stride(1), query_out.stride(2),
        query_c.size(1), key_c.stride(0), key_c.stride(1), key_c.stride(2),
        key_out.stride(0), key_out.stride(1), key_out.stride(2),
        key_c.size(1));
  } else {
    fused_rope_kernel<T, C, Acc><<<
        rope_blocks(query_pairs + key_pairs), 256, 0,
        getCurrentCUDAStream().stream()>>>(
        query_c.data_ptr<T>(), query_out.data_ptr<T>(), key_c.data_ptr<T>(),
        key_out.data_ptr<T>(), cos_c.data_ptr<C>(), sin_c.data_ptr<C>(),
        query_pairs, key_pairs, query_tokens, key_tokens, table.half_dim,
        table.rows, position_offset);
  }
  TP_CUDA_CHECK(cudaGetLastError());
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
      return rope_single_typed<T, tensorplay::Half>(
          input, cos, sin, table, tokens, position_offset);
    case DType::BFloat16:
      return rope_single_typed<T, tensorplay::BFloat16>(
          input, cos, sin, table, tokens, position_offset);
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
      return rope_pair_typed<T, tensorplay::Half>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    case DType::BFloat16:
      return rope_pair_typed<T, tensorplay::BFloat16>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported table dtype");
  }
}

} // namespace

Tensor rotary_embedding_cuda(const Tensor& input, const Tensor& cos,
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
      return rope_single_dispatch<tensorplay::Half>(
          input, cos, sin, table, tokens, position_offset);
    case DType::BFloat16:
      return rope_single_dispatch<tensorplay::BFloat16>(
          input, cos, sin, table, tokens, position_offset);
    default:
      TP_THROW(NotImplementedError, "rotary_embedding: unsupported input dtype");
  }
}

std::tuple<Tensor, Tensor> fused_rope_cuda(
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
      return rope_pair_dispatch<tensorplay::Half>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    case DType::BFloat16:
      return rope_pair_dispatch<tensorplay::BFloat16>(
          query, key, cos, sin, table, query_tokens, key_tokens,
          position_offset);
    default:
      TP_THROW(NotImplementedError, "fused_rope: unsupported input dtype");
  }
}

// MoE expert compute on CUDA: ragged grouped GEMM.
//   - No-grad fast path: one C++ pass over groups calling the plan-cached
//     cublasLt entry (gemm_impl) on zero-copy slice views, writing straight
//     into preallocated output -- no per-group dispatcher round-trips, no
//     cat copy pass.  Zero-fill allocation covers uncovered tail rows,
//   - GradMode path: differentiable composite (narrow/mm/cat through the
//     dispatcher) so CIA records inner nodes automatically.
Tensor grouped_mm_cuda(const Tensor& self, const Tensor& mat2,
                       const Tensor& offs) {
  if (self.dim() != 2 || mat2.dim() != 3) {
    TP_THROW(RuntimeError,
             "grouped_mm(): expected 2D self and 3D mat2, got ", self.dim(),
             "D and ", mat2.dim(), "D");
  }
  const int64_t M = self.size(0), K = self.size(1);
  const int64_t G = mat2.size(0);
  const int64_t N = mat2.size(2);
  if (mat2.size(1) != K) {
    TP_THROW(RuntimeError,
             "grouped_mm(): self.size(1) must match mat2.size(1): ", K,
             " vs ", mat2.size(1));
  }
  if (offs.dim() != 1 || offs.numel() != G) {
    TP_THROW(RuntimeError,
             "grouped_mm(): offs must be 1D of length mat2.size(0)=", G,
             ", got ", offs.dim(), "D/", offs.numel(), " elements");
  }
  if (offs.dtype() != DType::Int32 && offs.dtype() != DType::Int64) {
    TP_THROW(TypeError, "grouped_mm(): offs must be int32 or int64");
  }

  // Loop bounds live on the host: read directly when offs is CPU-resident
  // (the common construction site), D2H copy when it lives on the device.
  const int64_t nbytes = G * (offs.dtype() == DType::Int32 ? 4 : 8);
  std::vector<unsigned char> hoff(static_cast<size_t>(nbytes));
  if (offs.device().type() == DeviceType::CUDA) {
    cudaMemcpy(hoff.data(), offs.data_ptr(), nbytes, cudaMemcpyDeviceToHost);
  } else {
    std::memcpy(hoff.data(), offs.data_ptr(), static_cast<size_t>(nbytes));
  }
  auto read_off = [&](int64_t i) -> int64_t {
    return offs.dtype() == DType::Int32
               ? static_cast<int64_t>(
                     reinterpret_cast<const int32_t*>(hoff.data())[i])
               : reinterpret_cast<const int64_t*>(hoff.data())[i];
  };
  int64_t covered = 0;
  for (int64_t g = 0; g < G; ++g) {
    const int64_t end = read_off(g);
    if (end < covered || end > M) {
      TP_THROW(RuntimeError,
               "grouped_mm(): offs must be non-decreasing in [0, M_total=", M,
               "], got offs[", g, "]=", end);
    }
    covered = end;
  }

  const bool needs_grad =
      GradMode::is_enabled() && (self.requires_grad() || mat2.requires_grad());
  if (needs_grad) {
    // Differentiable composite: dispatcher primitives record inner nodes.
    std::vector<Tensor> parts;
    parts.reserve(G + 1);
    int64_t start = 0;
    for (int64_t g = 0; g < G; ++g) {
      const int64_t end = read_off(g);
      const int64_t len = end - start;
      if (len > 0) {
        Tensor wg = tpx::ops::narrow(mat2, 0, g, 1);
        wg = tpx::ops::reshape(wg, {K, N});
        parts.push_back(
            tpx::ops::mm(tpx::ops::narrow(self, 0, start, len), wg));
      }
      start = end;
    }
    if (start < M) {
      parts.push_back(tpx::ops::zeros({M - start, N}, self.dtype(),
                                      self.device(), false, false));
    }
    if (parts.empty()) {
      return tpx::ops::zeros({M, N}, self.dtype(), self.device(), false,
                             false);
    }
    return tpx::ops::cat(parts, 0);
  }

  // Fast path: zero-filled output + per-group cublasLt into slice views.
  Tensor out = Tensor::empty({M, N}, self.dtype(), self.device());
  zero_matmul_output_cuda(out);
  int64_t start = 0;
  for (int64_t g = 0; g < G; ++g) {
    const int64_t end = read_off(g);
    const int64_t len = end - start;
    if (len > 0) {
      Tensor aslice = self.slice(0, start, end);
      Tensor bg = mat2.select(0, g);          // [K, N], zero-copy
      Tensor oslice = out.slice(0, start, end);
      gemm_impl(aslice, bg, oslice, 1.0, 0.0, nullptr);
    }
    start = end;
  }
  return out;
}

std::tuple<Tensor, Tensor> sdpa_math_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
    const std::optional<Tensor>& dropout_mask,
    std::optional<double> scale, bool enable_gqa) {
  return tensorplay::composite::sdpa_math_composite(
      query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale,
      enable_gqa);
}

std::tuple<Tensor, Tensor> native_multi_head_attention_cuda(
    const Tensor& query, const Tensor& key, const Tensor& value,
    int64_t embed_dim, int64_t num_head, const Tensor& qkv_weight,
    const Tensor& qkv_bias, const Tensor& proj_weight, const Tensor& proj_bias,
    const std::optional<Tensor>& mask, bool need_weights,
    bool average_attn_weights, std::optional<int64_t> mask_type) {
  return tensorplay::composite::native_mha_composite(
      query, key, value, embed_dim, num_head, qkv_weight, qkv_bias,
      proj_weight, proj_bias, mask, need_weights, average_attn_weights,
      mask_type);
}

int64_t fused_sdp_choice_cuda(const Tensor& query, const Tensor& key,
                              const Tensor& value,
                              const std::optional<Tensor>& attn_mask,
                              double dropout_p, bool is_causal,
                              std::optional<double> scale, bool enable_gqa) {
  (void)key; (void)value; (void)is_causal; (void)scale;
  return tensorplay::composite::fused_sdp_choice_common(query, attn_mask,
                                                        dropout_p, enable_gqa);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, AttentionRopeKernels) {
  m.impl("_scaled_dot_product_attention_math", sdpa_math_cuda);
  m.impl("_native_multi_head_attention", native_multi_head_attention_cuda);
  m.impl("_fused_sdp_choice", fused_sdp_choice_cuda);
  m.impl("grouped_mm", grouped_mm_cuda);
  m.impl("rotary_embedding", rotary_embedding_cuda);
  m.impl("fused_rope", fused_rope_cuda);
}

} // namespace cuda
} // namespace tensorplay
