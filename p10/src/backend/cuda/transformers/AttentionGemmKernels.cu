#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CudaGemm.h"
#include "Exception.h"
#include "AttentionUtils.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace tensorplay {
namespace cuda {

#define TP_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

namespace {

template <typename DT>
__global__ void sdpa_softmax_kernel(
    DT* __restrict__ scores, int64_t groups, int64_t rows_per_head,
    int64_t queries, int64_t keys, bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= groups * rows_per_head) return;
  // The query index within a head, which is what the causal bound is stated
  // in.  It is the query's own length and not the head's row count, because a
  // group contributes several rows per token -- those rows are the same token,
  // so they share one bound.
  const int64_t token = row % queries;
  // The causal bound is the query row itself, which is the top-left alignment:
  // query row t sees keys <= t.  The two lengths are independent, so this is the
  // same bound the single-length form masked with, unchanged -- a context longer
  // than the query does not shift the diagonal, it is bounded by the row.
  const int64_t causal_limit = token;
  DT* values = scores + row * keys;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;

  float maximum = -INFINITY;
  for (int64_t j = threadIdx.x; j < keys; j += 32) {
    float value = to_float(values[j]);
    if (is_causal && j > causal_limit) value = -INFINITY;
    values[j] = from_float<DT>(value);
    maximum = max(maximum, value);
  }
  maximum = warpReduceMax(maximum);
  maximum = __shfl_sync(full_mask, maximum, 0);

  float total = 0.f;
  for (int64_t j = threadIdx.x; j < keys; j += 32) {
    const float value = to_float(values[j]);
    const float probability = isfinite(value) ? expf(value - maximum) : 0.f;
    values[j] = from_float<DT>(probability);
    total += probability;
  }
  total = warpReduceSum(total);
  total = __shfl_sync(full_mask, total, 0);
  const float inverse = total > 0.f ? 1.f / total : 0.f;
  for (int64_t j = threadIdx.x; j < keys; j += 32)
    values[j] = from_float<DT>(to_float(values[j]) * inverse);

}

// GEMM-backed attention over a query and a context of independent lengths, and
// over grouped heads.  The score matrix is materialized rather than tiled, so
// this is what serves a precision the tiled schedule has no tiles for; within a
// precision the tiled schedule is preferred wherever it applies, because it never
// writes the scores out at all.

}  // namespace

// A group's query heads all read one key head, so the group stands in for the
// query's row axis: every operand then has the same head count, which is what
// lets one linear batch index address all three of them.  Row s of key head hk
// is query head hk * group + s, and the memory order that produces already is
// the order the merged head axis reads in, so both reshapes are views.
template <typename DT>
Tensor sdpa_gemm_native(
    const Tensor& q, const Tensor& k, const Tensor& v,
    int64_t B, int64_t Hq, int64_t Hkv, int64_t Tq, int64_t Tkv, int64_t D,
    bool is_causal) {
  const DType dtype = q.dtype();
  const int64_t group = Hq / Hkv;
  const int64_t rows = group * Tq;
  const Tensor q_packed = q.reshape({B, Hkv, rows, D});

  const int64_t batch = B * Hkv;
  Tensor scores = Tensor::empty({B, Hkv, rows, Tkv}, dtype, q.device());
  Tensor q3 = q_packed.reshape({batch, rows, D});
  // The score product reads the key transposed, and the key is stored the
  // other way round.  Handing the transposition to the product reads the key
  // once; transposing it into a tensor of its own reads it once and writes a
  // second copy of it, and for a wide context that copy is the largest single
  // cost in the call -- it was measured at three quarters of the whole thing.
  Tensor k3 = k.reshape({batch, Tkv, D});
  Tensor scores3 = scores.reshape({batch, rows, Tkv});
  const long long q_stride = rows * D;
  const long long k_stride = Tkv * D;
  const long long score_stride = rows * Tkv;
  const float scale = 1.f / sqrtf(static_cast<float>(D));
  gemm_strided_batched_3d_op(
      q3, k3, scores3, batch, rows, Tkv, D, q_stride, k_stride,
      /*transpose_b=*/true, scale, 0.0);

  // The reduction is one warp per row, and the causal bound it states is the
  // query's own index within a head, so the head's row count and the query
  // length are passed separately: a group contributes several rows per token,
  // and those rows are one token, so they share a bound.
  const int64_t softmax_rows = batch * rows;
  sdpa_softmax_kernel<DT><<<
      static_cast<unsigned>(softmax_rows), 32, 0,
      getCurrentCUDAStream().stream()>>>(
      scores.data_ptr<DT>(), batch, rows, Tq, Tkv, is_causal);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor out = Tensor::empty({B, Hkv, rows, D}, dtype, q.device());
  Tensor scores3_again = scores.reshape({batch, rows, Tkv});
  Tensor v3 = v.reshape({batch, Tkv, D});
  Tensor out3 = out.reshape({batch, rows, D});
  // The value is as long as the context, not as long as the query, so its batch
  // stride is the context's -- sharing the query's would only be right while the
  // two lengths agree.
  const long long v_stride = Tkv * D;
  gemm_strided_batched_3d(
      scores3_again, v3, out3, batch, rows, D, Tkv, score_stride, v_stride,
      1.0, 0.0);
  return out.reshape({B, Hq, Tq, D});
}

template Tensor sdpa_gemm_native<float>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool);
template Tensor sdpa_gemm_native<tensorplay::Half>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool);
template Tensor sdpa_gemm_native<tensorplay::BFloat16>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, bool);

#undef TP_CUDA_CHECK

}
}
