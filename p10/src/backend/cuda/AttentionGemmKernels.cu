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
__global__ void sdpa_transpose_k_kernel(
    const DT* __restrict__ input, DT* __restrict__ output,
    int64_t tokens, int64_t head_dim) {
  constexpr int tile = 32;
  __shared__ DT smem[tile][tile + 1];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int64_t head = static_cast<int64_t>(blockIdx.z);
  const int64_t t0 = static_cast<int64_t>(blockIdx.x) * tile;
  const int64_t d0 = static_cast<int64_t>(blockIdx.y) * tile;
  const int64_t input_head = head * tokens * head_dim;
  const int64_t output_head = head * head_dim * tokens;

  for (int i = 0; i < 4; ++i) {
    const int t = ty + i * 8;
    const int64_t tg = t0 + t;
    const int64_t dg = d0 + tx;
    smem[t][tx] = (tg < tokens && dg < head_dim)
        ? input[input_head + tg * head_dim + dg]
        : from_float<DT>(0.f);
  }
  __syncthreads();
  for (int i = 0; i < 4; ++i) {
    const int d = ty + i * 8;
    const int64_t dg = d0 + d;
    const int64_t tg = t0 + tx;
    if (dg < head_dim && tg < tokens)
      output[output_head + dg * tokens + tg] = smem[tx][d];
  }
}

template <typename DT>
__global__ void sdpa_softmax_kernel(
    DT* __restrict__ scores, int64_t rows, int64_t tokens,
    bool is_causal) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows * tokens) return;
  const int64_t token = row % tokens;
  DT* values = scores + row * tokens;
  constexpr unsigned long long full_mask = 0xffffffffffffffffull;

  float maximum = -INFINITY;
  for (int64_t j = threadIdx.x; j < tokens; j += 32) {
    float value = to_float(values[j]);
    if (is_causal && j > token) value = -INFINITY;
    values[j] = from_float<DT>(value);
    maximum = max(maximum, value);
  }
  maximum = warpReduceMax(maximum);
  maximum = __shfl_sync(full_mask, maximum, 0);

  float total = 0.f;
  for (int64_t j = threadIdx.x; j < tokens; j += 32) {
    const float value = to_float(values[j]);
    const float probability = isfinite(value) ? expf(value - maximum) : 0.f;
    values[j] = from_float<DT>(probability);
    total += probability;
  }
  total = warpReduceSum(total);
  total = __shfl_sync(full_mask, total, 0);
  const float inverse = total > 0.f ? 1.f / total : 0.f;
  for (int64_t j = threadIdx.x; j < tokens; j += 32)
    values[j] = from_float<DT>(to_float(values[j]) * inverse);
}

}

template <typename DT>
Tensor sdpa_gemm_native(
    const Tensor& q, const Tensor& k, const Tensor& v,
    int64_t B, int64_t H, int64_t T, int64_t D, bool is_causal) {
  const DType dtype = q.dtype();
  Tensor kt = Tensor::empty({B, H, D, T}, dtype, q.device());
  dim3 transpose_grid(
      static_cast<unsigned>((T + 31) / 32),
      static_cast<unsigned>((D + 31) / 32),
      static_cast<unsigned>(B * H));
  dim3 transpose_block(32, 8);
  sdpa_transpose_k_kernel<DT><<<
      transpose_grid, transpose_block, 0, getCurrentCUDAStream().stream()>>>(
      k.data_ptr<DT>(), kt.data_ptr<DT>(), T, D);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor scores = Tensor::empty({B, H, T, T}, dtype, q.device());
  Tensor q3 = q.reshape({B * H, T, D});
  Tensor kt3 = kt.reshape({B * H, D, T});
  Tensor scores3 = scores.reshape({B * H, T, T});
  const long long q_stride = T * D;
  const long long kt_stride = D * T;
  const long long score_stride = T * T;
  const float scale = 1.f / sqrtf(static_cast<float>(D));
  gemm_strided_batched_3d(
      q3, kt3, scores3, B * H, T, T, D, q_stride, kt_stride, scale, 0.0);

  const int64_t softmax_rows = B * H * T;
  const unsigned softmax_blocks = static_cast<unsigned>(softmax_rows);
  sdpa_softmax_kernel<DT><<<
      softmax_blocks, 32, 0, getCurrentCUDAStream().stream()>>>(
      scores.data_ptr<DT>(), B * H, T, is_causal);
  TP_CUDA_CHECK(cudaGetLastError());

  Tensor out = Tensor::empty({B, H, T, D}, dtype, q.device());
  Tensor scores3_again = scores.reshape({B * H, T, T});
  Tensor v3 = v.reshape({B * H, T, D});
  Tensor out3 = out.reshape({B * H, T, D});
  gemm_strided_batched_3d(
      scores3_again, v3, out3, B * H, T, D, T, score_stride, q_stride,
      1.0, 0.0);
  return out;
}

template Tensor sdpa_gemm_native<float>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
    int64_t, bool);
template Tensor sdpa_gemm_native<tensorplay::Half>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
    int64_t, bool);
template Tensor sdpa_gemm_native<tensorplay::BFloat16>(
    const Tensor&, const Tensor&, const Tensor&, int64_t, int64_t, int64_t,
    int64_t, bool);

#undef TP_CUDA_CHECK

}
}
