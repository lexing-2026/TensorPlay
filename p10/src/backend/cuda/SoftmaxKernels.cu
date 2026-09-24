#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDNNUtils.h"
#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <type_traits>
#include <limits>
#include "OutWrite.h"

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
       TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)


namespace tensorplay {
namespace cuda {

#ifdef USE_CUDNN

Tensor softmax_native_impl(const Tensor& self, int64_t dim, bool log_mode);

// Defined after the native kernels: runs the wave/register tiers for a
// contiguous row along the fast dimension and reports whether one applied.
bool softmax_native_fast_path(const Tensor& self, Tensor& result,
                              int64_t outer_size, int64_t softmax_size,
                              int64_t inner_size, bool log_mode);

Tensor cudnn_softmax(const Tensor& self, int64_t dim, bool log) {
    int64_t ndim = self.dim();
    if (dim < 0) dim += ndim;
    // The DNN softmax call below is only wired for 4-byte element types;
    // reduced-precision inputs would be described with a mismatched element
    // size and read/written out of bounds.  Route them to the native kernel,
    // which accumulates in float and returns the input dtype.
    if (self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        return softmax_native_impl(self, dim, log);
    }
    // The descriptor maps logical dims onto contiguous layout.
    Tensor input = self.is_contiguous() ? self : self.contiguous();

    // Map to NCHW where C is the softmax dim
    // N = outer_size, C = softmax_size, H = inner_size, W = 1
    int64_t outer_size = 1;
    for(int i=0; i<dim; ++i) outer_size *= input.size(i);
    int64_t softmax_size = input.size(dim);
    int64_t inner_size = 1;
    for(int i=dim+1; i<ndim; ++i) inner_size *= input.size(i);

    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(input.shape()), input.dtype(), input.device());

    // A contiguous row along the fast dimension is a single-pass kernel in
    // this unit; the DNN library path stays for strided/spatial layouts.
    if (softmax_native_fast_path(input, result, outer_size, softmax_size,
                                 inner_size, log)) {
        return result;
    }

    cudnnHandle_t handle = CUDAContext::getCudnnHandle();

    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));

    cudnnDataType_t c_dtype = (input.dtype() == DType::Float64) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    // Set 4D descriptor with logical dims
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, c_dtype, (int)outer_size, (int)softmax_size, (int)inner_size, 1));

    cudnnSoftmaxAlgorithm_t algo = log ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL; // Softmax over C

    float alpha = 1.0f, beta = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void *alpha_p = &alpha, *beta_p = &beta;
    if (input.dtype() == DType::Float64) { alpha_p = &alpha_d; beta_p = &beta_d; }

    CUDNN_CHECK(cudnnSoftmaxForward(handle, algo, mode, alpha_p, desc, input.data_ptr(), beta_p, desc, result.data_ptr()));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));

    return result;
}

Tensor softmax_kernel_cudnn(const Tensor& self, int64_t dim, DType dtype) {
    // Ignoring dtype arg for now (assuming input dtype)
    return cudnn_softmax(self, dim, false);
}

Tensor log_softmax_kernel_cudnn(const Tensor& self, int64_t dim, DType dtype) {
    return cudnn_softmax(self, dim, true);
}

#endif  // USE_CUDNN

// --- Native softmax ---
//
// Row softmax over an arbitrary dimension without a DNN-library dependency.
// The softmax dimension is the middle of the (outer, softmax_size, inner)
// view: one thread block walks one row whose consecutive elements sit
// `inner_size` elements apart.  Two passes (max, then sum of exponentials)
// keep the numerics of the library path; reduced-precision inputs compute
// in fp32, matching the upcast the DNN descriptor path applies.

template <typename scalar_t, typename compute_t>
__global__ void softmax_dim_kernel(
    scalar_t* out, const scalar_t* in, int64_t rows, int64_t softmax_size,
    int64_t inner_size, bool log_mode) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;
  const int64_t outer = row / inner_size;
  const int64_t inner = row % inner_size;
  const scalar_t* row_in =
      in + outer * softmax_size * inner_size + inner;
  scalar_t* row_out = out + outer * softmax_size * inner_size + inner;

  __shared__ compute_t tile[1024];
  const int tid = static_cast<int>(threadIdx.x);

  compute_t thread_max = -std::numeric_limits<compute_t>::infinity();
  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    const compute_t v = static_cast<compute_t>(row_in[j * inner_size]);
    thread_max = (v > thread_max) ? v : thread_max;
  }
  tile[tid] = thread_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] = (tile[tid + s] > tile[tid]) ? tile[tid + s] : tile[tid];
    __syncthreads();
  }
  const compute_t row_max = tile[0];

  compute_t thread_sum = compute_t(0);
  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    thread_sum += std::exp(static_cast<compute_t>(row_in[j * inner_size]) - row_max);
  }
  tile[tid] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] += tile[tid + s];
    __syncthreads();
  }
  const compute_t denom = std::log(tile[0]);

  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    const compute_t v =
        static_cast<compute_t>(row_in[j * inner_size]) - row_max - denom;
    row_out[j * inner_size] =
        static_cast<scalar_t>(log_mode ? v : std::exp(v));
  }
}

// --- Wave (warp) softmax for rows along the fast dimension ---
//
// One wave owns one row (or two rows for short rows): every lane keeps a
// strided register slice, and the max / sum reductions run as butterfly
// shuffles with no shared-memory round trip.  The logical wave width is
// clamped to the next power of two of the row length, so a hardware wave can
// hold two independent logical waves when rows are short.  Templates are
// instantiated for both 32- and 64-lane hardware waves and selected from the
// device attribute at launch time.

namespace {

inline int softmax_log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

inline int softmax_wave_size() {
  static int wave = []() {
    int dev = 0, lanes = 32;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&lanes, cudaDevAttrWarpSize, dev);
    return lanes > 0 ? lanes : 32;
  }();
  return wave;
}

template <typename acc_t, int WAVE_SIZE, int WAVE_BATCH, bool kIsMax>
__device__ __forceinline__ void wave_butterfly_reduce(acc_t* values) {
  const unsigned long long mask =
      WAVE_SIZE == 64 ? 0xffffffffffffffffull : 0xffffffffull;
#pragma unroll
  for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WAVE_BATCH; ++i) {
      const acc_t other =
          __shfl_xor_sync(mask, values[i], offset, WAVE_SIZE);
      values[i] = kIsMax ? (other > values[i] ? other : values[i])
                         : (values[i] + other);
    }
  }
}

template <typename scalar_t, typename acc_t, int LOG2_ELEMENTS, bool LOG_MODE,
          int HW_WAVE>
__global__ void softmax_wave_forward(scalar_t* dst, const scalar_t* src,
                                     int batch_count, int stride,
                                     int element_count) {
  constexpr int kNextPow2 = 1 << LOG2_ELEMENTS;
  constexpr int kWAVE = (kNextPow2 < HW_WAVE) ? kNextPow2 : HW_WAVE;
  constexpr int kIterations = kNextPow2 / kWAVE;
  constexpr int kBatchesPerWave = (kNextPow2 <= 128) ? 2 : 1;

  const int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * kBatchesPerWave;
  int local_batches = batch_count - first_batch;
  if (local_batches > kBatchesPerWave) local_batches = kBatchesPerWave;

  const int local_idx = static_cast<int>(threadIdx.x);
  src += first_batch * stride + local_idx;
  dst += first_batch * stride + local_idx;

  acc_t elems[kBatchesPerWave][kIterations];
  for (int i = 0; i < kBatchesPerWave; ++i) {
    const int count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int element_index = local_idx + it * kWAVE;
      elems[i][it] =
          element_index < count
              ? static_cast<acc_t>(src[i * element_count + it * kWAVE])
              : -std::numeric_limits<acc_t>::infinity();
    }
  }

  acc_t max_value[kBatchesPerWave];
#pragma unroll
  for (int i = 0; i < kBatchesPerWave; ++i) {
    max_value[i] = elems[i][0];
#pragma unroll
    for (int it = 1; it < kIterations; ++it) {
      max_value[i] =
          max_value[i] > elems[i][it] ? max_value[i] : elems[i][it];
    }
  }
  wave_butterfly_reduce<acc_t, kWAVE, kBatchesPerWave, true>(max_value);

  acc_t sum[kBatchesPerWave];
#pragma unroll
  for (int i = 0; i < kBatchesPerWave; ++i) {
    sum[i] = acc_t(0);
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      if (LOG_MODE) {
        sum[i] += std::exp(elems[i][it] - max_value[i]);
      } else {
        elems[i][it] = std::exp(elems[i][it] - max_value[i]);
        sum[i] += elems[i][it];
      }
    }
  }
  wave_butterfly_reduce<acc_t, kWAVE, kBatchesPerWave, false>(sum);

#pragma unroll
  for (int i = 0; i < kBatchesPerWave; ++i) {
    if (i >= local_batches) break;
#pragma unroll
    for (int it = 0; it < kIterations; ++it) {
      const int element_index = local_idx + it * kWAVE;
      if (element_index < element_count) {
        if (LOG_MODE) {
          // Elements stay raw: subtract the max and the log normalizer.
          const acc_t v =
              elems[i][it] - max_value[i] - std::log(sum[i]);
          dst[i * element_count + it * kWAVE] = static_cast<scalar_t>(v);
        } else {
          // Elements hold exp(x - max) from the accumulation pass.
          dst[i * element_count + it * kWAVE] =
              static_cast<scalar_t>(elems[i][it] / sum[i]);
        }
      }
    }
  }
}

template <typename scalar_t, typename acc_t, bool LOG_MODE>
void launch_wave_softmax(scalar_t* dst, const scalar_t* src, int64_t batch_count,
                         int64_t element_count, cudaStream_t stream) {
  const int log2_elements = softmax_log2_ceil(static_cast<int>(element_count));
  const int next_pow2 = 1 << log2_elements;
  const int hw_wave = softmax_wave_size();
  const int wave_size = next_pow2 < hw_wave ? next_pow2 : hw_wave;
  const int batches_per_wave = (next_pow2 <= 128) ? 2 : 1;
  constexpr int kThreadsPerBlock = 128;
  const int warps_per_block = kThreadsPerBlock / wave_size;
  const int batches_per_block = warps_per_block * batches_per_wave;
  const int64_t blocks =
      (batch_count + batches_per_block - 1) / batches_per_block;
  dim3 threads(static_cast<unsigned>(wave_size),
               static_cast<unsigned>(warps_per_block), 1);

#define TP_LAUNCH_WAVE_SOFTMAX(L2E)                                          \
  do {                                                                       \
    if (hw_wave == 64) {                                                     \
      softmax_wave_forward<scalar_t, acc_t, L2E, LOG_MODE, 64>               \
          <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(           \
              dst, src, static_cast<int>(batch_count),                       \
              static_cast<int>(element_count),                               \
              static_cast<int>(element_count));                              \
    } else {                                                                 \
      softmax_wave_forward<scalar_t, acc_t, L2E, LOG_MODE, 32>               \
          <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(           \
              dst, src, static_cast<int>(batch_count),                       \
              static_cast<int>(element_count),                               \
              static_cast<int>(element_count));                              \
    }                                                                        \
  } while (0)

  switch (log2_elements) {
    case 0: TP_LAUNCH_WAVE_SOFTMAX(0); break;
    case 1: TP_LAUNCH_WAVE_SOFTMAX(1); break;
    case 2: TP_LAUNCH_WAVE_SOFTMAX(2); break;
    case 3: TP_LAUNCH_WAVE_SOFTMAX(3); break;
    case 4: TP_LAUNCH_WAVE_SOFTMAX(4); break;
    case 5: TP_LAUNCH_WAVE_SOFTMAX(5); break;
    case 6: TP_LAUNCH_WAVE_SOFTMAX(6); break;
    case 7: TP_LAUNCH_WAVE_SOFTMAX(7); break;
    case 8: TP_LAUNCH_WAVE_SOFTMAX(8); break;
    case 9: TP_LAUNCH_WAVE_SOFTMAX(9); break;
    case 10: TP_LAUNCH_WAVE_SOFTMAX(10); break;
    default: TP_LAUNCH_WAVE_SOFTMAX(11); break;
  }
#undef TP_LAUNCH_WAVE_SOFTMAX
  CUDA_CHECK(cudaGetLastError());
}

// The wave kernel covers rows laid out along the fastest dimension with a
// bounded row length; anything else (strided rows, very long rows, huge
// batches) stays on the block kernel below.
template <typename scalar_t, typename acc_t, bool LOG_MODE>
bool try_wave_softmax(const Tensor& self, Tensor& result, int64_t softmax_size,
                      int64_t rows) {
  constexpr int64_t kMaxRowBytes = 8192;
  if (softmax_size <= 0 || softmax_size > 2048) return false;
  if (softmax_size * static_cast<int64_t>(sizeof(scalar_t)) > kMaxRowBytes)
    return false;
  if (rows * softmax_size > static_cast<int64_t>(INT32_MAX)) return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  launch_wave_softmax<scalar_t, acc_t, LOG_MODE>(
      result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), rows,
      softmax_size, getCurrentCUDAStream().stream());
  return true;
}

// --- Register-resident row softmax for long rows ---
//
// One block owns one row and holds its whole slice in registers, so global
// memory sees exactly one read pass and one write pass no matter how long
// the row is.  When the row length divides the packet width and both base
// pointers are 16-byte aligned, every slot moves a vector packet; otherwise
// a scalar-slot variant keeps the same single-pass residency for odd rows.
// Padding slots carry the max identity, which contributes zero to the
// exponential sum and is masked out of the store.

template <typename T, int V>
struct alignas(sizeof(T) * V) SoftmaxPack {
  T v[V];
};

// Row data moves exactly once in each direction and is never revisited, so
// the streaming hints keep it out of the L2 working set of other blocks.
template <typename T, int V>
__device__ __forceinline__ SoftmaxPack<T, V> softmax_load_stream(
    const SoftmaxPack<T, V>* p) {
  SoftmaxPack<T, V> r;
  *reinterpret_cast<uint4*>(&r) = __ldcs(reinterpret_cast<const uint4*>(p));
  return r;
}

template <typename T, int V>
__device__ __forceinline__ void softmax_store_stream(SoftmaxPack<T, V>* p,
                                                     const SoftmaxPack<T, V>& r) {
  __stcs(reinterpret_cast<uint4*>(p), *reinterpret_cast<const uint4*>(&r));
}

template <typename compute_t, bool kIsMax>
__device__ __forceinline__ compute_t softmax_block_reduce(
    compute_t value, compute_t* scratch) {
  constexpr int kWave = 32;
  const unsigned mask = 0xffffffffu;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int warp = static_cast<int>(threadIdx.x) / kWave;
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset /= 2) {
    const compute_t other = __shfl_xor_sync(mask, value, offset, kWave);
    value = kIsMax ? (other > value ? other : value) : (value + other);
  }
  const int warps = static_cast<int>(blockDim.x) / kWave;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps
        ? scratch[lane]
        : (kIsMax ? -std::numeric_limits<compute_t>::infinity()
                  : compute_t(0));
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2) {
      const compute_t other = __shfl_xor_sync(mask, value, offset, kWave);
      value = kIsMax ? (other > value ? other : value) : (value + other);
    }
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Online (max, sum-of-exponentials) pair: combining two partials rescales
// the sums onto the merged max, so a single tree reduction yields the row
// statistics that a separate max pass followed by a sum pass would produce.
template <typename compute_t>
struct SoftmaxMS {
  compute_t m;
  compute_t s;
};

template <typename compute_t>
__device__ __forceinline__ SoftmaxMS<compute_t> softmax_ms_combine(
    SoftmaxMS<compute_t> a, SoftmaxMS<compute_t> b) {
  const compute_t m = a.m > b.m ? a.m : b.m;
  if (m == -std::numeric_limits<compute_t>::infinity()) {
    // Both partials are empty: the sums are already zero, and the max
    // difference would turn inf - inf into NaN.
    return SoftmaxMS<compute_t>{m, a.s + b.s};
  }
  const compute_t sa = a.s * std::exp(a.m - m);
  const compute_t sb = b.s * std::exp(b.m - m);
  return SoftmaxMS<compute_t>{m, sa + sb};
}

template <typename compute_t>
__device__ __forceinline__ SoftmaxMS<compute_t> softmax_block_reduce_ms(
    SoftmaxMS<compute_t> value, SoftmaxMS<compute_t>* scratch) {
  constexpr int kWave = 32;
  const unsigned mask = 0xffffffffu;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int warp = static_cast<int>(threadIdx.x) / kWave;
#pragma unroll
  for (int offset = kWave / 2; offset > 0; offset /= 2) {
    const SoftmaxMS<compute_t> other{
        __shfl_xor_sync(mask, value.m, offset, kWave),
        __shfl_xor_sync(mask, value.s, offset, kWave)};
    value = softmax_ms_combine(value, other);
  }
  const int warps = static_cast<int>(blockDim.x) / kWave;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps
        ? scratch[lane]
        : SoftmaxMS<compute_t>{-std::numeric_limits<compute_t>::infinity(),
                               compute_t(0)};
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2) {
      const SoftmaxMS<compute_t> other{
          __shfl_xor_sync(mask, value.m, offset, kWave),
          __shfl_xor_sync(mask, value.s, offset, kWave)};
      value = softmax_ms_combine(value, other);
    }
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

template <typename scalar_t, typename compute_t, int PACKS, bool LOG_MODE>
__global__ void softmax_reg_packed_kernel(scalar_t* __restrict__ out,
                                          const scalar_t* __restrict__ in,
                                          int classes, int packets) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  constexpr int kWave = 32;
  __shared__ SoftmaxMS<compute_t> reduce_ms[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const scalar_t* row = in + static_cast<int64_t>(blockIdx.x) * classes;
  scalar_t* row_out = out + static_cast<int64_t>(blockIdx.x) * classes;

  SoftmaxPack<scalar_t, kPack> v[PACKS];
  const SoftmaxPack<scalar_t, kPack>* src =
      reinterpret_cast<const SoftmaxPack<scalar_t, kPack>*>(row);
#pragma unroll
  for (int i = 0; i < PACKS; ++i) {
    const int slot = tid + i * static_cast<int>(blockDim.x);
    if (slot < packets) {
      v[i] = softmax_load_stream(src + slot);
    } else {
#pragma unroll
      for (int k = 0; k < kPack; ++k) {
        v[i].v[k] = -std::numeric_limits<scalar_t>::infinity();
      }
    }
  }

  // Thread-local moments, then one online pair reduction: every thread's
  // partial sum is rescaled onto the merged max inside the combine, so a
  // single tree walk yields the same (max, sum) a two-pass flow would.
  compute_t thread_max = -std::numeric_limits<compute_t>::infinity();
#pragma unroll
  for (int i = 0; i < PACKS; ++i) {
#pragma unroll
    for (int k = 0; k < kPack; ++k) {
      const compute_t x = static_cast<compute_t>(v[i].v[k]);
      thread_max = x > thread_max ? x : thread_max;
    }
  }
  compute_t thread_sum = compute_t(0);
#pragma unroll
  for (int i = 0; i < PACKS; ++i) {
#pragma unroll
    for (int k = 0; k < kPack; ++k) {
      thread_sum += std::exp(static_cast<compute_t>(v[i].v[k]) - thread_max);
    }
  }
  const SoftmaxMS<compute_t> ms =
      softmax_block_reduce_ms<compute_t>(
          SoftmaxMS<compute_t>{thread_max, thread_sum}, reduce_ms);
  const compute_t norm = LOG_MODE ? std::log(ms.s) : compute_t(1) / ms.s;

  SoftmaxPack<scalar_t, kPack>* dst =
      reinterpret_cast<SoftmaxPack<scalar_t, kPack>*>(row_out);
#pragma unroll
  for (int i = 0; i < PACKS; ++i) {
    const int slot = tid + i * static_cast<int>(blockDim.x);
    if (slot < packets) {
      SoftmaxPack<scalar_t, kPack> r;
#pragma unroll
      for (int k = 0; k < kPack; ++k) {
        const compute_t x = static_cast<compute_t>(v[i].v[k]) - ms.m;
        r.v[k] = static_cast<scalar_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
      }
      softmax_store_stream(dst + slot, r);
    }
  }
}

template <typename scalar_t, typename compute_t, int REG, bool LOG_MODE>
__global__ void softmax_reg_scalar_kernel(scalar_t* __restrict__ out,
                                          const scalar_t* __restrict__ in,
                                          int classes) {
  constexpr int kWave = 32;
  __shared__ compute_t reduce_max[kWave];
  __shared__ compute_t reduce_sum[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const scalar_t* row = in + static_cast<int64_t>(blockIdx.x) * classes;
  scalar_t* row_out = out + static_cast<int64_t>(blockIdx.x) * classes;

  scalar_t v[REG];
#pragma unroll
  for (int i = 0; i < REG; ++i) {
    const int slot = tid + i * static_cast<int>(blockDim.x);
    v[i] = slot < classes
        ? row[slot]
        : -std::numeric_limits<scalar_t>::infinity();
  }

  compute_t thread_max = -std::numeric_limits<compute_t>::infinity();
#pragma unroll
  for (int i = 0; i < REG; ++i) {
    const compute_t x = static_cast<compute_t>(v[i]);
    thread_max = x > thread_max ? x : thread_max;
  }
  thread_max = softmax_block_reduce<compute_t, true>(thread_max, reduce_max);

  compute_t thread_sum = compute_t(0);
#pragma unroll
  for (int i = 0; i < REG; ++i) {
    thread_sum += std::exp(static_cast<compute_t>(v[i]) - thread_max);
  }
  thread_sum = softmax_block_reduce<compute_t, false>(thread_sum, reduce_sum);
  const compute_t norm =
      LOG_MODE ? std::log(thread_sum) : compute_t(1) / thread_sum;

#pragma unroll
  for (int i = 0; i < REG; ++i) {
    const int slot = tid + i * static_cast<int>(blockDim.x);
    if (slot < classes) {
      const compute_t x = static_cast<compute_t>(v[i]) - thread_max;
      row_out[slot] =
          static_cast<scalar_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
    }
  }
}

// A register-resident block only exists if the instantiation fits the
// per-block register budget at the requested width; the occupancy query
// reads the actual compiled register count instead of guessing.
template <typename KernelT>
bool softmax_launch_feasible(KernelT kernel, int threads) {
  int blocks_per_sm = 0;
  const cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, threads, 0);
  return err == cudaSuccess && blocks_per_sm > 0;
}

template <typename scalar_t, typename compute_t, bool LOG_MODE>
bool try_reg_softmax(const Tensor& self, Tensor& result,
                     int64_t softmax_size, int64_t rows) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  if (softmax_size < 2049 || rows > INT32_MAX) return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  const scalar_t* in = self.data_ptr<scalar_t>();
  scalar_t* out = result.data_ptr<scalar_t>();
  const bool aligned =
      (reinterpret_cast<uintptr_t>(in) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(out) % 16 == 0);
  const auto stream = getCurrentCUDAStream().stream();
  const auto wave_round = [](int64_t n) {
    return (n + 31) / 32 * 32;
  };
  const auto grid = static_cast<unsigned>(rows);

  if (aligned && softmax_size % kPack == 0) {
    // Prefer a few hundred threads with several register slots each: the
    // row fits in registers either way, and smaller blocks raise the
    // resident-block count, which is what saturates these pure streams.
    const int packets = static_cast<int>(softmax_size / kPack);
    if (packets <= 256) {
      const int threads = static_cast<int>(wave_round(packets));
      softmax_reg_packed_kernel<scalar_t, compute_t, 1, LOG_MODE>
          <<<grid, threads, 0, stream>>>(out, in,
                                         static_cast<int>(softmax_size),
                                         packets);
    } else if (packets <= 512) {
      const int threads = static_cast<int>(wave_round((packets + 1) / 2));
      softmax_reg_packed_kernel<scalar_t, compute_t, 2, LOG_MODE>
          <<<grid, threads, 0, stream>>>(out, in,
                                         static_cast<int>(softmax_size),
                                         packets);
    } else if (packets <= 2048) {
      const int threads = static_cast<int>(wave_round((packets + 3) / 4));
      softmax_reg_packed_kernel<scalar_t, compute_t, 4, LOG_MODE>
          <<<grid, threads, 0, stream>>>(out, in,
                                         static_cast<int>(softmax_size),
                                         packets);
    } else if (packets <= 8192) {
      const int threads = static_cast<int>(wave_round((packets + 7) / 8));
      // Wide blocks only launch when the instantiation actually fits the
      // per-block register budget; otherwise the row-wide kernel below
      // takes over.
      if (threads > 512 &&
          !softmax_launch_feasible(
              softmax_reg_packed_kernel<scalar_t, compute_t, 8, LOG_MODE>,
              threads)) {
        return false;
      }
      softmax_reg_packed_kernel<scalar_t, compute_t, 8, LOG_MODE>
          <<<grid, threads, 0, stream>>>(out, in,
                                         static_cast<int>(softmax_size),
                                         packets);
    } else {
      return false;
    }
  } else {
    // Scalar slots cover any row length and alignment with the same
    // small-block shape.
    if (softmax_size > 16384) return false;
    const int threads = static_cast<int>(
        wave_round((softmax_size + 15) / 16));
    if (threads > 512 &&
        !softmax_launch_feasible(
            softmax_reg_scalar_kernel<scalar_t, compute_t, 16, LOG_MODE>,
            threads)) {
      return false;
    }
    softmax_reg_scalar_kernel<scalar_t, compute_t, 16, LOG_MODE>
        <<<grid, threads, 0, stream>>>(out, in,
                                       static_cast<int>(softmax_size));
  }
  CUDA_CHECK(cudaGetLastError());
  return true;
}

// Rows too long to stay in registers: one block per row makes two vector
// passes over memory. The first folds every chunk into an online (max,
// sum-of-exponentials) pair — each chunk's partial sum is rescaled onto the
// merged max when combined — so the whole row is read once for both moments
// instead of once per moment. Total traffic is one read for the moments, one
// read + one write for the output; a per-moment two-pass flow reads the row
// an additional time.
template <typename scalar_t, typename compute_t, bool LOG_MODE>
__global__ void softmax_wide_kernel(scalar_t* __restrict__ out,
                                    const scalar_t* __restrict__ in,
                                    int classes) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  // Elements folded per combine step: fewer combines spend less on the
  // rescaling exponentials, which dominates once exp runs on the double
  // units rather than the special-function units.
  constexpr int kChunkPackets = 16 / kPack < 1 ? 1 : 16 / kPack;
  constexpr int kWave = 32;
  __shared__ SoftmaxMS<compute_t> reduce_ms[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const int packets = classes / kPack;
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * classes;
  const SoftmaxPack<scalar_t, kPack>* src =
      reinterpret_cast<const SoftmaxPack<scalar_t, kPack>*>(in + row_base);
  SoftmaxPack<scalar_t, kPack>* dst =
      reinterpret_cast<SoftmaxPack<scalar_t, kPack>*>(out + row_base);

  SoftmaxMS<compute_t> acc{
      -std::numeric_limits<compute_t>::infinity(), compute_t(0)};
  const scalar_t pad = -std::numeric_limits<scalar_t>::infinity();
  const int stride = static_cast<int>(blockDim.x) * kChunkPackets;
  for (int base = tid; base < packets; base += stride) {
    SoftmaxPack<scalar_t, kPack> vc[kChunkPackets];
#pragma unroll
    for (int c = 0; c < kChunkPackets; ++c) {
      const int slot = base + c * static_cast<int>(blockDim.x);
      vc[c] = slot < packets
          ? softmax_load_stream(src + slot)
          : SoftmaxPack<scalar_t, kPack>{};
      if (slot >= packets) {
#pragma unroll
        for (int k = 0; k < kPack; ++k) vc[c].v[k] = pad;
      }
    }
    compute_t chunk_max = -std::numeric_limits<compute_t>::infinity();
#pragma unroll
    for (int c = 0; c < kChunkPackets; ++c) {
#pragma unroll
      for (int k = 0; k < kPack; ++k) {
        const compute_t x = static_cast<compute_t>(vc[c].v[k]);
        chunk_max = x > chunk_max ? x : chunk_max;
      }
    }
    compute_t chunk_sum = compute_t(0);
    if (chunk_max != -std::numeric_limits<compute_t>::infinity()) {
#pragma unroll
      for (int c = 0; c < kChunkPackets; ++c) {
#pragma unroll
        for (int k = 0; k < kPack; ++k) {
          chunk_sum += std::exp(static_cast<compute_t>(vc[c].v[k]) - chunk_max);
        }
      }
    }
    // A chunk of pure -inf has a NaN relative sum; it contributes zero
    // exponentials to any finite merged max, and an all--inf row ends with
    // the same zero sum a two-pass flow produces.
    acc = softmax_ms_combine(
        acc, SoftmaxMS<compute_t>{chunk_max, chunk_sum});
  }
  const SoftmaxMS<compute_t> ms = softmax_block_reduce_ms<compute_t>(acc, reduce_ms);
  const compute_t norm = LOG_MODE ? std::log(ms.s) : compute_t(1) / ms.s;

  for (int slot = tid; slot < packets; slot += static_cast<int>(blockDim.x)) {
    const SoftmaxPack<scalar_t, kPack> v = softmax_load_stream(src + slot);
    SoftmaxPack<scalar_t, kPack> r;
#pragma unroll
    for (int k = 0; k < kPack; ++k) {
      const compute_t x = static_cast<compute_t>(v.v[k]) - ms.m;
      r.v[k] = static_cast<scalar_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
    }
    softmax_store_stream(dst + slot, r);
  }
}

template <typename scalar_t, typename compute_t, bool LOG_MODE>
bool try_wide_softmax(const Tensor& self, Tensor& result,
                      int64_t softmax_size, int64_t rows) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  if (softmax_size < 2049 || softmax_size > INT32_MAX || rows > INT32_MAX)
    return false;
  if (softmax_size % kPack != 0) return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  const scalar_t* in = self.data_ptr<scalar_t>();
  scalar_t* out = result.data_ptr<scalar_t>();
  if ((reinterpret_cast<uintptr_t>(in) % 16 != 0) ||
      (reinterpret_cast<uintptr_t>(out) % 16 != 0)) {
    return false;
  }
  // Blocks spread one row per block, so rows below a couple of waves per
  // SM leave the rest of the machine idle; wider blocks then put more
  // loads in flight per SM, which is what feeds a pure streaming kernel.
  static thread_local cudaDeviceProp properties{};
  static thread_local int queried_device = -1;
  const int current = currentDevice();
  if (queried_device != current) {
    CUDA_CHECK(cudaGetDeviceProperties(&properties, current));
    queried_device = current;
  }
  const int threads =
      rows * 2 < properties.multiProcessorCount ? 1024 : 512;
  softmax_wide_kernel<scalar_t, compute_t, LOG_MODE>
      <<<static_cast<unsigned>(rows), threads, 0,
         getCurrentCUDAStream().stream()>>>(out, in,
                                            static_cast<int>(softmax_size));
  CUDA_CHECK(cudaGetLastError());
  return true;
}

}  // namespace

template <typename scalar_t, typename compute_t>
void softmax_dim_dispatch(const Tensor& self, Tensor& result, int64_t dim,
                          bool log_mode) {
  // One thread block per (outer, inner) row; the kernel splits blockIdx
  // back into the outer/inner pair.
  int64_t rows = 1;
  for (int64_t i = 0; i < dim; ++i) rows *= self.size(i);
  int64_t inner_size = 1;
  for (int64_t i = dim + 1; i < self.dim(); ++i) inner_size *= self.size(i);
  const int64_t softmax_size = self.size(dim);
  if (inner_size == 1) {
    // Fast dimension: wave-resident kernel, one hardware wave per row.
    const bool wave = log_mode
        ? try_wave_softmax<scalar_t, compute_t, true>(self, result,
                                                      softmax_size, rows)
        : try_wave_softmax<scalar_t, compute_t, false>(self, result,
                                                       softmax_size, rows);
    if (wave) return;
    // Longer contiguous rows keep the whole slice in registers: one read
    // pass and one write pass regardless of row length.
    const bool reg = log_mode
        ? try_reg_softmax<scalar_t, compute_t, true>(self, result,
                                                     softmax_size, rows)
        : try_reg_softmax<scalar_t, compute_t, false>(self, result,
                                                      softmax_size, rows);
    if (reg) return;
    // Rows beyond the register budget: two streaming vector passes, with
    // both moments folded into a single read.
    const bool wide = log_mode
        ? try_wide_softmax<scalar_t, compute_t, true>(self, result,
                                                      softmax_size, rows)
        : try_wide_softmax<scalar_t, compute_t, false>(self, result,
                                                       softmax_size, rows);
    if (wide) return;
  }
  constexpr int kThreads = 256;
  softmax_dim_kernel<scalar_t, compute_t>
      <<<static_cast<unsigned>(rows * inner_size), kThreads, 0,
         getCurrentCUDAStream().stream()>>>(
          result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(),
          rows * inner_size, softmax_size, inner_size, log_mode);
  CUDA_CHECK(cudaGetLastError());
}

bool softmax_native_fast_path(const Tensor& self, Tensor& result,
                              int64_t outer_size, int64_t softmax_size,
                              int64_t inner_size, bool log_mode) {
  if (inner_size != 1 || outer_size == 0 || softmax_size <= 0) return false;
  switch (self.dtype()) {
    case DType::Float32: {
      if (log_mode) {
        return try_wave_softmax<float, float, true>(self, result, softmax_size,
                                                    outer_size) ||
               try_reg_softmax<float, float, true>(self, result, softmax_size,
                                                   outer_size) ||
               try_wide_softmax<float, float, true>(self, result, softmax_size,
                                                    outer_size);
      }
      return try_wave_softmax<float, float, false>(self, result, softmax_size,
                                                   outer_size) ||
             try_reg_softmax<float, float, false>(self, result, softmax_size,
                                                  outer_size) ||
             try_wide_softmax<float, float, false>(self, result, softmax_size,
                                                   outer_size);
    }
    case DType::Float64: {
      if (log_mode) {
        return try_wave_softmax<double, double, true>(self, result,
                                                      softmax_size,
                                                      outer_size) ||
               try_reg_softmax<double, double, true>(self, result,
                                                     softmax_size,
                                                     outer_size) ||
               try_wide_softmax<double, double, true>(self, result,
                                                      softmax_size,
                                                      outer_size);
      }
      return try_wave_softmax<double, double, false>(self, result,
                                                     softmax_size,
                                                     outer_size) ||
             try_reg_softmax<double, double, false>(self, result,
                                                    softmax_size, outer_size) ||
             try_wide_softmax<double, double, false>(self, result,
                                                     softmax_size, outer_size);
    }
    default:
      return false;
  }
}


Tensor softmax_native_impl(const Tensor& self, int64_t dim, bool log_mode) {
  int64_t dim_idx = dim < 0 ? dim + self.dim() : dim;
  Tensor result = Tensor::empty(
      static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
      self.device());
  switch (self.dtype()) {
    case DType::Float32:
      softmax_dim_dispatch<float, float>(self, result, dim_idx, log_mode);
      break;
    case DType::Float64:
      softmax_dim_dispatch<double, double>(self, result, dim_idx, log_mode);
      break;
    case DType::Float16:
      softmax_dim_dispatch<Half, float>(self, result, dim_idx, log_mode);
      break;
    case DType::BFloat16:
      softmax_dim_dispatch<BFloat16, float>(self, result, dim_idx, log_mode);
      break;
    default:
      TP_THROW(NotImplementedError,
               "softmax: unsupported dtype on this GPU backend");
  }
  return result;
}

Tensor softmax_kernel_native(const Tensor& self, int64_t dim, DType dtype) {
  (void)dtype;
  return softmax_native_impl(self, dim, false);
}

Tensor log_softmax_kernel_native(const Tensor& self, int64_t dim, DType dtype) {
  (void)dtype;
  return softmax_native_impl(self, dim, true);
}

namespace {

// One block per (outer, inner) row, matching the forward layout.  The row
// reduction streams twice: once for the shared factor, once for the write.
//   softmax:  grad_in = out * (grad - <grad, out>_dim)
//   log:      grad_in = grad - exp(out) * <grad>_dim
template <typename scalar_t, typename compute_t, bool LOG_MODE>
__global__ void softmax_backward_data_kernel(
    scalar_t* grad_in, const scalar_t* grad, const scalar_t* out,
    int64_t rows, int64_t softmax_size, int64_t inner_size) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) return;
  const int64_t outer = row / inner_size;
  const int64_t inner = row % inner_size;
  const int64_t base = outer * softmax_size * inner_size + inner;
  const scalar_t* row_grad = grad + base;
  const scalar_t* row_out = out + base;
  scalar_t* row_in = grad_in + base;

  __shared__ compute_t tile[1024];
  const int tid = static_cast<int>(threadIdx.x);

  compute_t thread_sum = compute_t(0);
  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    thread_sum += static_cast<compute_t>(row_grad[j * inner_size]) *
                  (LOG_MODE ? compute_t(1)
                            : static_cast<compute_t>(row_out[j * inner_size]));
  }
  tile[tid] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) tile[tid] += tile[tid + s];
    __syncthreads();
  }
  const compute_t factor = tile[0];

  for (int64_t j = tid; j < softmax_size; j += blockDim.x) {
    const compute_t g = static_cast<compute_t>(row_grad[j * inner_size]);
    const compute_t o = static_cast<compute_t>(row_out[j * inner_size]);
    compute_t v;
    if constexpr (LOG_MODE) {
      v = g - std::exp(o) * factor;
    } else {
      v = o * (g - factor);
    }
    row_in[j * inner_size] = static_cast<scalar_t>(v);
  }
}

// Register-resident backward for whole contiguous rows: the gradient and the
// saved output live in registers, so each operand is read once and the result
// written once.  The strided streaming kernel instead re-reads both operands
// for its write pass, costing two extra passes over the row.
template <typename scalar_t, typename compute_t, int REG, bool LOG_MODE>
__global__ void softmax_bwd_reg_kernel(scalar_t* grad_in,
                                       const scalar_t* grad,
                                       const scalar_t* out, int classes) {
  constexpr int kWave = 32;
  __shared__ compute_t reduce[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * classes;

  scalar_t vg[REG], vo[REG];
#pragma unroll
  for (int i = 0; i < REG; ++i) {
    const int slot = tid + i * static_cast<int>(blockDim.x);
    // Zero padding keeps the factor reduction clean; padded slots are
    // skipped on the write pass.
    vg[i] = slot < classes ? grad[base + slot] : scalar_t(0);
    vo[i] = slot < classes ? out[base + slot] : scalar_t(0);
  }

  compute_t thread_sum = compute_t(0);
#pragma unroll
  for (int i = 0; i < REG; ++i) {
    thread_sum += static_cast<compute_t>(vg[i]) *
        (LOG_MODE ? compute_t(1) : static_cast<compute_t>(vo[i]));
  }
  thread_sum = softmax_block_reduce<compute_t, false>(thread_sum, reduce);
  const compute_t factor = thread_sum;

#pragma unroll
  for (int i = 0; i < REG; ++i) {
    const int slot = tid + i * static_cast<int>(blockDim.x);
    if (slot < classes) {
      const compute_t g = static_cast<compute_t>(vg[i]);
      const compute_t o = static_cast<compute_t>(vo[i]);
      grad_in[base + slot] = static_cast<scalar_t>(
          LOG_MODE ? g - std::exp(o) * factor : o * (g - factor));
    }
  }
}

template <typename scalar_t, typename compute_t>
bool try_softmax_bwd_reg(scalar_t* grad_in, const scalar_t* grad,
                         const scalar_t* out, int64_t classes, int64_t rows,
                         bool log_mode, cudaStream_t stream) {
  // REG = 16 slots per thread caps the block at 512 threads for a 8192-wide
  // row; longer rows take the strided kernel.
  constexpr int kReg = 16;
  if (classes < 1 || classes > kReg * 512 || rows > INT32_MAX) return false;
  const int threads = static_cast<int>(
      ((classes + kReg - 1) / kReg + 31) / 32 * 32);
  if (log_mode) {
    softmax_bwd_reg_kernel<scalar_t, compute_t, kReg, true>
        <<<static_cast<unsigned>(rows), threads, 0, stream>>>(
            grad_in, grad, out, static_cast<int>(classes));
  } else {
    softmax_bwd_reg_kernel<scalar_t, compute_t, kReg, false>
        <<<static_cast<unsigned>(rows), threads, 0, stream>>>(
            grad_in, grad, out, static_cast<int>(classes));
  }
  CUDA_CHECK(cudaGetLastError());
  return true;
}

// grad_output drives the result dtype; reduced-width inputs accumulate and
// compute in float.  grad_output may carry float32 for a half input (the
// half_to_float forward path), in which case the result casts back to the
// input dtype.
Tensor softmax_backward_native_impl(const Tensor& grad_output,
                                    const Tensor& output, int64_t dim,
                                    DType input_dtype, bool log_mode) {
  Tensor g = grad_output.dim() == 0 ? grad_output.view({1}) : grad_output;
  Tensor o = output.dim() == 0 ? output.view({1}) : output;
  const int64_t nd = g.dim();
  const int64_t d = dim < 0 ? dim + nd : dim;
  if (d < 0 || d >= nd) {
    TP_THROW(IndexError,
             "dim must be non-negative and less than input dimensions");
  }
  DType result_dtype = g.dtype();
  if (result_dtype != input_dtype && result_dtype == DType::Float32 &&
      input_dtype == DType::Float16) {
    result_dtype = DType::Float16;
  }
  if (!isFloatingType(result_dtype)) {
    TP_THROW(TypeError, "unsupported dtype for softmax backward");
  }

  Tensor result =
      Tensor::empty(static_cast<std::vector<int64_t>>(g.shape()), result_dtype,
                    g.device());
  if (g.numel() == 0) return result;

  Tensor gc = g.contiguous();
  Tensor oc = o.contiguous().to(gc.dtype());
  Tensor result_work = result_dtype == gc.dtype()
                           ? result
                           : Tensor::empty(
                                 static_cast<std::vector<int64_t>>(g.shape()),
                                 gc.dtype(), g.device());

  int64_t outer = 1;
  for (int64_t i = 0; i < d; ++i) outer *= gc.size(i);
  int64_t inner = 1;
  for (int64_t i = d + 1; i < gc.dim(); ++i) inner *= gc.size(i);
  const int64_t dim_size = gc.size(d);
  const int64_t rows = outer * inner;
  const int threads = dim_size < 256 ? 32 : 256;

  // Whole contiguous rows along the fast dimension fit the register tier:
  // one read of each operand and one write of the result.
  if (inner == 1 && gc.is_contiguous() && oc.is_contiguous() &&
      result_work.is_contiguous() && rows > 0) {
    const auto stream = getCurrentCUDAStream().stream();
    bool reg_done = false;
    switch (gc.dtype()) {
      case DType::Float32:
        reg_done = try_softmax_bwd_reg<float, float>(
            result_work.data_ptr<float>(), gc.data_ptr<float>(),
            oc.data_ptr<float>(), dim_size, rows, log_mode, stream);
        break;
      case DType::Float64:
        reg_done = try_softmax_bwd_reg<double, double>(
            result_work.data_ptr<double>(), gc.data_ptr<double>(),
            oc.data_ptr<double>(), dim_size, rows, log_mode, stream);
        break;
      case DType::Float16:
        reg_done = try_softmax_bwd_reg<Half, float>(
            result_work.data_ptr<Half>(), gc.data_ptr<Half>(),
            oc.data_ptr<Half>(), dim_size, rows, log_mode, stream);
        break;
      case DType::BFloat16:
        reg_done = try_softmax_bwd_reg<BFloat16, float>(
            result_work.data_ptr<BFloat16>(), gc.data_ptr<BFloat16>(),
            oc.data_ptr<BFloat16>(), dim_size, rows, log_mode, stream);
        break;
      default:
        break;
    }
    if (reg_done) {
      if (result_work.data_ptr() != result.data_ptr()) {
        result.copy_(result_work);
      }
      return result;
    }
  }

  #define TP_SOFTMAX_BWD_LAUNCH(ctype, acc)                                \
  if (log_mode) {                                                          \
    constexpr bool LOG_MODE = true;                                        \
    softmax_backward_data_kernel<ctype, acc, LOG_MODE>                     \
        <<<static_cast<unsigned>(rows), threads, 0,                        \
           getCurrentCUDAStream().stream()>>>(                             \
            result_work.data_ptr<ctype>(), gc.data_ptr<ctype>(),           \
            oc.data_ptr<ctype>(), rows, dim_size, inner);                  \
  } else {                                                                 \
    constexpr bool LOG_MODE = false;                                       \
    softmax_backward_data_kernel<ctype, acc, LOG_MODE>                     \
        <<<static_cast<unsigned>(rows), threads, 0,                        \
           getCurrentCUDAStream().stream()>>>(                             \
            result_work.data_ptr<ctype>(), gc.data_ptr<ctype>(),           \
            oc.data_ptr<ctype>(), rows, dim_size, inner);                  \
  }

  switch (gc.dtype()) {
    case DType::Float32:
      TP_SOFTMAX_BWD_LAUNCH(float, float)
      break;
    case DType::Float64:
      TP_SOFTMAX_BWD_LAUNCH(double, double)
      break;
    case DType::Float16:
      TP_SOFTMAX_BWD_LAUNCH(Half, float)
      break;
    case DType::BFloat16:
      TP_SOFTMAX_BWD_LAUNCH(BFloat16, float)
      break;
    default:
      TP_THROW(TypeError, "unsupported dtype for softmax backward");
  }
  #undef TP_SOFTMAX_BWD_LAUNCH
  CUDA_CHECK(cudaGetLastError());

  if (result_work.data_ptr() != result.data_ptr()) {
    result.copy_(result_work);
  }
  return result;
}

}  // namespace

Tensor _softmax_backward_data_cuda(const Tensor& grad_output,
                                   const Tensor& output, int64_t dim,
                                   DType input_dtype) {
  return softmax_backward_native_impl(grad_output, output, dim, input_dtype,
                                      /*log_mode=*/false);
}

Tensor& _softmax_backward_data_out_cuda(const Tensor& grad_output,
                                        const Tensor& output, int64_t dim,
                                        DType input_dtype, Tensor& grad_input) {
  write_out(grad_input, softmax_backward_native_impl(grad_output, output, dim,
                                            input_dtype, /*log_mode=*/false));
  return grad_input;
}

Tensor _log_softmax_backward_data_cuda(const Tensor& grad_output,
                                       const Tensor& output, int64_t dim,
                                       DType input_dtype) {
  return softmax_backward_native_impl(grad_output, output, dim, input_dtype,
                                      /*log_mode=*/true);
}

Tensor& _log_softmax_backward_data_out_cuda(const Tensor& grad_output,
                                            const Tensor& output, int64_t dim,
                                            DType input_dtype,
                                            Tensor& grad_input) {
  write_out(grad_input, softmax_backward_native_impl(grad_output, output, dim,
                                            input_dtype, /*log_mode=*/true));
  return grad_input;
}

Tensor& _softmax_out_cuda(const Tensor& self, int64_t dim, bool half_to_float,
                          Tensor& out) {
  if (half_to_float) {
    TP_THROW(RuntimeError,
             "softmax with half to float conversion is not supported on this backend");
  }
  write_out(out, softmax_native_impl(self, dim, false));
  return out;
}

Tensor& _log_softmax_out_cuda(const Tensor& self, int64_t dim,
                              bool half_to_float, Tensor& out) {
  if (half_to_float) {
    TP_THROW(RuntimeError,
             "log_softmax with half to float conversion is not supported on this backend");
  }
  write_out(out, softmax_native_impl(self, dim, true));
  return out;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, SoftmaxKernels) {
    m.impl("_softmax.out", _softmax_out_cuda);
    m.impl("_log_softmax.out", _log_softmax_out_cuda);
    m.impl("_softmax_backward_data", _softmax_backward_data_cuda);
    m.impl("_softmax_backward_data.out", _softmax_backward_data_out_cuda);
    m.impl("_log_softmax_backward_data", _log_softmax_backward_data_cuda);
    m.impl("_log_softmax_backward_data.out", _log_softmax_backward_data_out_cuda);
#if defined(USE_CUDNN) && !defined(USE_ROCM)
    m.impl("softmax", softmax_kernel_cudnn);
    m.impl("log_softmax", log_softmax_kernel_cudnn);
#elif defined(USE_CUDNN) && defined(USE_ROCM)
    m.impl("softmax", softmax_kernel_native);
    m.impl("log_softmax", log_softmax_kernel_native);
#else
    m.impl("softmax", softmax_kernel_native);
    m.impl("log_softmax", log_softmax_kernel_native);
#endif
}

} // namespace cuda
} // namespace tensorplay
