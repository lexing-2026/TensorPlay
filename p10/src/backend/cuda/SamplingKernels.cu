#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Allocator.h"
#include "CUDAContext.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <vector>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)


// LLM hot-path sampling operators.
// References:
//     (renormRowsL1 + prefix-sum binary search; sampleMultinomialOnce block scan)
//     (block bitonic sort for small inputs)
//   - sample:      fused temperature/top-k/top-p decoder sampler (single kernel)

namespace tensorplay {
namespace cuda {

namespace {

#define TP_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

template <typename T>
__device__ inline T warpReduceSum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

template <typename T>
__device__ inline T warpReduceMax(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  return val;
}

template <typename T>
__device__ inline T blockReduceSum(T val, T* smem) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warpReduceSum(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : static_cast<T>(0);
  if (wid == 0) val = warpReduceSum(val);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

template <typename T>
__device__ inline T blockReduceMax(T val, T* smem) {
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  val = warpReduceMax(val);
  if (lane == 0) smem[wid] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x >> 5)) ? smem[lane] : static_cast<T>(-INFINITY);
  if (wid == 0) val = warpReduceMax(val);
  if (threadIdx.x == 0) smem[0] = val;
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// Computes the normalized inclusive prefix sum of every row in a single
// kernel (renormRowsL1 + cumsum fused), so the input tensor is never mutated.
// ---------------------------------------------------------------------------

__global__ void prefix_renorm_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int64_t rows, int64_t cols) {
  extern __shared__ float smem[];  // blockDim floats
  int64_t row = blockIdx.x;
  const float* row_in = in + row * cols;
  float* row_out = out + row * cols;

  float total = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    total += row_in[c];
  total = blockReduceSum(total, smem);

  float carry = 0.f;
  for (int64_t base = 0; base < cols; base += blockDim.x) {
    int64_t chunk = min((int64_t)blockDim.x, cols - base);
    __syncthreads();
    smem[threadIdx.x] = (threadIdx.x < chunk) ? row_in[base + threadIdx.x] : 0.f;
    __syncthreads();
    if (threadIdx.x == 0) {
      float acc = carry;
      for (int64_t i = 0; i < chunk; ++i) {
        acc += smem[i];
        smem[i] = acc;
      }
      carry = acc;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < chunk; i += blockDim.x)
      row_out[base + i] = total > 0.f ? smem[i] / total : (float)(base + i + 1) / (float)cols;
  }
}

__device__ int binary_search_multinomial(const float* cumdist, int64_t size, float val) {
  int lo = 0, hi = (int)size - 1;
  if (val >= cumdist[hi]) return hi;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (cumdist[mid] >= val) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

__global__ void binary_search_sample_kernel(
    const float* __restrict__ cumdist,
    const float* __restrict__ uniforms,
    int64_t* __restrict__ dest,
    int64_t rows, int64_t cols, int64_t samples_per_row) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = rows * samples_per_row;
  if (idx >= total) return;
  int64_t row = idx / samples_per_row;
  int64_t sample = idx % samples_per_row;
  dest[idx] = binary_search_multinomial(cumdist + row * cols, cols, uniforms[idx]);
}

// ---------------------------------------------------------------------------
// One block per row; supports num_samples == 1 (the LLM decode hot path).
// ---------------------------------------------------------------------------

__global__ void scan_sample_kernel(
    const float* __restrict__ dist,
    const float* __restrict__ uniforms,
    int64_t* __restrict__ dest,
    int64_t rows, int64_t cols) {
  extern __shared__ float smem[];  // blockDim + 2 floats
  __shared__ bool found;
  __shared__ int foundPos;

  int64_t row = blockIdx.x;
  const float* row_dist = dist + row * cols;
  int nthreads = blockDim.x;

  float sum = 0.f;
  for (int c = threadIdx.x; c < cols; c += nthreads)
    sum += row_dist[c];
  sum = blockReduceSum(sum, smem);

  if (threadIdx.x == 0) {
    smem[0] = sum;
    smem[1] = uniforms[row];
    foundPos = 0;
  }
  __syncthreads();

  if (smem[0] == 0.f) {
    if (threadIdx.x == 0) dest[row] = 0;
    return;
  }

  float sample = smem[1];
  int chunks = (int)((cols + nthreads - 1) / nthreads);
  float prevHighProb = 0.f;
  found = false;

  for (int chunk = 0; chunk < chunks && !found; ++chunk) {
    int cat = chunk * nthreads + threadIdx.x;
    float dist_val = cat < cols ? row_dist[cat] / smem[0] : 0.f;
    smem[threadIdx.x] = dist_val;
    __syncthreads();

    if (threadIdx.x == 0) {
      for (int i = 1; i < nthreads; ++i)
        smem[i] += smem[i - 1];
    }
    __syncthreads();

    float curBucket = smem[threadIdx.x] + prevHighProb;
    float prevBucket = threadIdx.x == 0 ? prevHighProb : smem[threadIdx.x - 1] + prevHighProb;
    bool inBucket = (cat < cols) && (sample >= prevBucket) && (sample < curBucket) && (dist_val > 0.f);

    if (inBucket) {
      atomicMax(&foundPos, cat);
      found = true;
    }
    prevHighProb += smem[nthreads - 1];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (found) {
      dest[row] = foundPos;
    } else {
      for (int64_t c = cols - 1; c >= 0; --c) {
        if (row_dist[c] > 0.f) {
          dest[row] = c;
          break;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// sample — impl 0 reference pipeline: temp+softmax -> exact top-k mask ->
// exact top-p mask -> multinomial (prefix sum + binary search). Multi-kernel.
// ---------------------------------------------------------------------------

// probs = softmax(logits / temperature), computed in place on a scratch copy.
__global__ void temp_softmax_kernel(float* logits, int64_t rows, int64_t cols, float temperature) {
  extern __shared__ float smem[];  // blockDim floats
  int64_t row = blockIdx.x;
  float* row_logits = logits + row * cols;
  float inv_temp = 1.f / temperature;

  float max_val = -INFINITY;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    max_val = max(max_val, row_logits[c]);
  max_val = blockReduceMax(max_val, smem);

  float sum = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    float e = expf((row_logits[c] - max_val) * inv_temp);
    row_logits[c] = e;
    sum += e;
  }
  sum = blockReduceSum(sum, smem);
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    row_logits[c] /= sum;
}

// Exact top-k filter in place: keeps exactly the k largest entries per row,
__global__ void topk_mask_kernel(float* probs, int64_t rows, int64_t cols, int64_t k) {
  extern __shared__ float smem[];  // 2*blockDim floats + k floats + k int64
  float* red = smem;
  float* rvals = smem + 2 * blockDim.x;
  int64_t* ridx = reinterpret_cast<int64_t*>(rvals + k);
  int64_t row = blockIdx.x;
  float* row_probs = probs + row * cols;

  for (int64_t t = 0; t < k; ++t) {
    float best = -INFINITY;
    int64_t best_idx = -1;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
      if (row_probs[c] > best) {
        best = row_probs[c];
        best_idx = c;
      }
    }
    red[threadIdx.x * 2] = best;
    red[threadIdx.x * 2 + 1] = (float)best_idx;
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int i = 1; i < blockDim.x; ++i) {
        if (red[i * 2] > red[0]) {
          red[0] = red[i * 2];
          red[1] = red[i * 2 + 1];
        }
      }
      rvals[t] = red[0];
      ridx[t] = (int64_t)red[1];
      if (ridx[t] >= 0) row_probs[ridx[t]] = -INFINITY;
    }
    __syncthreads();
  }
  // Zero everything, then restore only the removed (top-k) elements.
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    row_probs[c] = 0.f;
  __syncthreads();
  for (int64_t i = threadIdx.x; i < k; i += blockDim.x)
    if (ridx[i] >= 0) row_probs[ridx[i]] = rvals[i];
}

// Exact top-p filter in place (nucleus sampling): keeps the smallest set of
// value tiers whose cumulative probability >= top_p; ties within a tier are
// kept together.
__global__ void topp_mask_kernel(float* probs, int64_t rows, int64_t cols, float top_p) {
  extern __shared__ float smem[];  // blockDim floats
  int64_t row = blockIdx.x;
  float* row_probs = probs + row * cols;

  // filtered set is the whole distribution when its sum is below the threshold).
  float total = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    total += row_probs[c];
  total = blockReduceSum(total, smem);
  if (total <= top_p) return;

  __shared__ float shcum;
  if (threadIdx.x == 0) shcum = 0.f;
  __syncthreads();

  // Keep the smallest set of value tiers (from the top) whose cumulative
  // probability reaches top_p; zero everything below the crossing tier.
  while (true) {
    float best = -INFINITY;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
      best = max(best, row_probs[c]);
    best = blockReduceMax(best, smem);
    if (best <= 0.f) break;

    float tier = 0.f;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
      if (row_probs[c] == best) tier += row_probs[c];
    tier = blockReduceSum(tier, smem);

    if (threadIdx.x == 0) shcum += tier;
    __syncthreads();
    if (shcum >= top_p) break;

    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
      if (row_probs[c] == best) row_probs[c] = 0.f;
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// sample — impl 1: fused single kernel per row: softmax, top-k/top-p
// selection via iterative argmax into a smem candidate list, then prefix sum
// + binary search over candidates. One launch; requires cols <= 4096.
// ---------------------------------------------------------------------------

__global__ void fused_sample_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ uniforms,
    int64_t* __restrict__ dest,
    int64_t rows, int64_t cols,
    float temperature, int64_t top_k, float top_p) {
  // smem layout: probs[cols] | red[2*blockDim] | cand_vals[kMaxCand] | cand_idxs[kMaxCand]
  extern __shared__ float smem[];
  float* probs = smem;
  float* red = smem + 4096;
  float* cand_vals = red + 2 * blockDim.x;
  int64_t* cand_idxs = reinterpret_cast<int64_t*>(cand_vals + 1024);
  constexpr int kMaxCand = 1024;

  int64_t row = blockIdx.x;
  const float* row_logits = logits + row * cols;
  int nthreads = blockDim.x;
  float inv_temp = 1.f / temperature;

  // --- softmax into shared memory ---
  float max_val = -INFINITY;
  for (int64_t c = threadIdx.x; c < cols; c += nthreads)
    max_val = max(max_val, row_logits[c]);
  max_val = blockReduceMax(max_val, red);

  float sum = 0.f;
  for (int64_t c = threadIdx.x; c < cols; c += nthreads) {
    float e = expf((row_logits[c] - max_val) * inv_temp);
    probs[c] = e;
    sum += e;
  }
  sum = blockReduceSum(sum, red);
  for (int64_t c = threadIdx.x; c < cols; c += nthreads)
    probs[c] /= sum;
  __syncthreads();

  // --- top-k filter: exact k candidates (one occurrence per iteration) ---
  int count = 0;
  if (top_k > 0 && top_k < cols) {
    int64_t nsel = min(top_k, (int64_t)kMaxCand);
    for (int64_t t = 0; t < nsel; ++t) {
      float best = -INFINITY;
      int64_t best_idx = -1;
      for (int64_t c = threadIdx.x; c < cols; c += nthreads) {
        if (probs[c] > best) {
          best = probs[c];
          best_idx = c;
        }
      }
      red[threadIdx.x * 2] = best;
      red[threadIdx.x * 2 + 1] = (float)best_idx;
      __syncthreads();
      if (threadIdx.x == 0) {
        for (int i = 1; i < nthreads; ++i) {
          if (red[i * 2] > red[0]) {
            red[0] = red[i * 2];
            red[1] = red[i * 2 + 1];
          }
        }
        cand_vals[count] = red[0];
        cand_idxs[count] = (int64_t)red[1];
        if (red[1] >= 0.f) probs[(int64_t)red[1]] = 0.f;
      }
      __syncthreads();
      ++count;
    }
  }

  // --- top-p filter over the candidate set (or full distribution) ---
  if (top_p < 1.f) {
    int m = count > 0 ? count : (int)cols;
    // If the candidate mass cannot reach top_p, keep everything.
    float total = 0.f;
    for (int64_t i = threadIdx.x; i < m; i += nthreads) {
      float v = count > 0 ? cand_vals[i] : probs[i];
      total += v;
    }
    total = blockReduceSum(total, red);
    if (total > top_p) {
      __shared__ float shcum;
      if (threadIdx.x == 0) shcum = 0.f;
      __syncthreads();
      // Keep the smallest set of value tiers (from the top) whose cumulative
      // probability reaches top_p; zero everything below the crossing tier.
      while (true) {
        float best = -INFINITY;
        for (int64_t i = threadIdx.x; i < m; i += nthreads) {
          float v = count > 0 ? cand_vals[i] : probs[i];
          best = max(best, v);
        }
        best = blockReduceMax(best, red);
        if (best <= 0.f) break;

        float tier = 0.f;
        for (int64_t i = threadIdx.x; i < m; i += nthreads) {
          float v = count > 0 ? cand_vals[i] : probs[i];
          if (v == best) tier += v;
        }
        tier = blockReduceSum(tier, red);
        if (threadIdx.x == 0) shcum += tier;
        __syncthreads();
        if (shcum >= top_p) break;

        for (int64_t i = threadIdx.x; i < m; i += nthreads) {
          if (count > 0) {
            if (cand_vals[i] == best) cand_vals[i] = 0.f;
          } else {
            if (probs[i] == best) probs[i] = 0.f;
          }
        }
        __syncthreads();
      }
    }
  }

  // --- normalize candidates + prefix sum + binary search (thread 0) ---
  if (threadIdx.x == 0) {
    // If top-k was applied the candidates live in the smem list; otherwise the
    // full softmax distribution (possibly top-p masked) is the candidate set.
    bool topk_applied = (top_k > 0 && top_k < cols);
    int n = topk_applied ? count : (int)cols;
    float* src = topk_applied ? cand_vals : probs;

    float total = 0.f;
    for (int i = 0; i < n; ++i) total += src[i];
    if (n == 0 || total <= 0.f) {
      dest[row] = 0;
      return;
    }
    float acc = 0.f;
    for (int i = 0; i < n; ++i) {
      acc += src[i] / total;
      src[i] = acc;
    }
    float u = uniforms[row];
    int lo = 0, hi = n - 1;
    if (u >= src[hi]) {
      dest[row] = topk_applied ? cand_idxs[hi] : hi;
      return;
    }
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (src[mid] >= u) hi = mid;
      else lo = mid + 1;
    }
    dest[row] = topk_applied ? cand_idxs[lo] : lo;
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

Tensor multinomial_kernel_cuda(const Tensor& self, int64_t num_samples, bool replacement, int64_t impl) {
  if (!replacement) {
    TP_THROW(NotImplementedError, "multinomial: replacement=False is not implemented for CUDA");
  }
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
  if (prob.dtype() != DType::Float32) {
    prob = prob.to(DType::Float32);
  }

  std::vector<int64_t> out_shape = is_1d ? std::vector<int64_t>{num_samples}
                                         : std::vector<int64_t>{rows, num_samples};
  Tensor result = Tensor::empty(out_shape, DType::Int64, prob.device());
  if (num_samples == 0) return result;

  Tensor uniforms = Tensor::empty({rows * num_samples}, DType::Float32, prob.device());
  uniforms.uniform_(0.0, 1.0);

  constexpr int kThreads = 256;
  Tensor cumdist = Tensor::empty({rows, cols}, DType::Float32, prob.device());
  size_t smem = kThreads * sizeof(float);

  if (impl == 1 && num_samples == 1) {
    size_t smem_once = (kThreads + 2) * sizeof(float);
    scan_sample_kernel<<<rows, kThreads, smem_once, getCurrentCUDAStream().stream()>>>(
        prob.data_ptr<float>(), uniforms.data_ptr<float>(), result.data_ptr<int64_t>(), rows, cols);
    TP_CUDA_CHECK(cudaGetLastError());
  } else {
    prefix_renorm_kernel<<<rows, kThreads, smem, getCurrentCUDAStream().stream()>>>(
        prob.data_ptr<float>(), cumdist.data_ptr<float>(), rows, cols);
    TP_CUDA_CHECK(cudaGetLastError());
    int64_t total = rows * num_samples;
    binary_search_sample_kernel<<<(total + kThreads - 1) / kThreads, kThreads, 0, getCurrentCUDAStream().stream()>>>(
        cumdist.data_ptr<float>(), uniforms.data_ptr<float>(), result.data_ptr<int64_t>(),
        rows, cols, num_samples);
    TP_CUDA_CHECK(cudaGetLastError());
  }
  return result;
}

Tensor sample_kernel_cuda(const Tensor& logits, double temperature, int64_t top_k, double top_p, int64_t impl) {
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
  Tensor uniforms = Tensor::empty({rows}, DType::Float32, input.device());
  uniforms.uniform_(0.0, 1.0);

  constexpr int kThreads = 256;

  if (impl == 0) {
    // Reference pipeline on a scratch copy (never mutate the caller's tensor).
    Tensor work = input.clone();
    size_t smem = kThreads * sizeof(float);
    temp_softmax_kernel<<<rows, kThreads, smem, getCurrentCUDAStream().stream()>>>(work.data_ptr<float>(), rows, cols, (float)temperature);
    TP_CUDA_CHECK(cudaGetLastError());
    if (top_k > 0 && top_k < cols) {
      size_t smem_k = kThreads * 2 * sizeof(float) + top_k * (sizeof(float) + sizeof(int64_t));
      topk_mask_kernel<<<rows, kThreads, smem_k, getCurrentCUDAStream().stream()>>>(work.data_ptr<float>(), rows, cols, top_k);
      TP_CUDA_CHECK(cudaGetLastError());
    }
    if (top_p < 1.0) {
      topp_mask_kernel<<<rows, kThreads, smem, getCurrentCUDAStream().stream()>>>(work.data_ptr<float>(), rows, cols, (float)top_p);
      TP_CUDA_CHECK(cudaGetLastError());
    }
    Tensor cumdist = Tensor::empty({rows, cols}, DType::Float32, input.device());
    prefix_renorm_kernel<<<rows, kThreads, smem, getCurrentCUDAStream().stream()>>>(work.data_ptr<float>(), cumdist.data_ptr<float>(), rows, cols);
    TP_CUDA_CHECK(cudaGetLastError());
    binary_search_sample_kernel<<<(rows + kThreads - 1) / kThreads, kThreads, 0, getCurrentCUDAStream().stream()>>>(
        cumdist.data_ptr<float>(), uniforms.data_ptr<float>(), result.data_ptr<int64_t>(),
        rows, cols, 1);
    TP_CUDA_CHECK(cudaGetLastError());
  } else if (impl == 1) {
    if (cols > 4096) {
      TP_THROW(NotImplementedError, "sample impl=1 (fused) supports vocab <= 4096; use impl=0 for larger vocab");
    }
    if (top_k > 1024) {
      TP_THROW(NotImplementedError, "sample impl=1 (fused) supports top_k <= 1024; use impl=0");
    }
    size_t smem = 4096 * sizeof(float) + 2 * kThreads * sizeof(float) +
                  1024 * sizeof(float) + 1024 * sizeof(int64_t);
    fused_sample_kernel<<<rows, kThreads, smem, getCurrentCUDAStream().stream()>>>(
        input.data_ptr<float>(), uniforms.data_ptr<float>(), result.data_ptr<int64_t>(),
        rows, cols, (float)temperature, top_k, (float)top_p);
    TP_CUDA_CHECK(cudaGetLastError());
  } else {
    TP_THROW(RuntimeError, "sample: unknown impl " + std::to_string(impl));
  }
  return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, SamplingKernels) {
  m.impl("multinomial", multinomial_kernel_cuda);
  m.impl("sample", sample_kernel_cuda);
}

} // namespace
} // namespace cuda
} // namespace tensorplay
