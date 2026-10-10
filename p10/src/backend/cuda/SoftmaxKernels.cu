#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"
#include "Exception.h"
#include "CUDNNUtils.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
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

namespace {

// Softmax over the single element of a 0-dim tensor reduces to a constant:
// probability 1, or log-probability 0.  The fill goes through the dispatched
// factory: the single element lives in device memory, which no host-side
// store may touch.
template <typename scalar_t>
Tensor softmax_scalar_result(const Tensor& self, bool log_mode) {
  return Tensor::full(std::vector<int64_t>{},
                      Scalar(log_mode ? 0.0 : 1.0), self.dtype(), self.device());
}

Tensor softmax_scalar_result_dispatch(const Tensor& self, bool log_mode) {
  switch (self.dtype()) {
    case DType::Float32:
      return softmax_scalar_result<float>(self, log_mode);
    case DType::Float64:
      return softmax_scalar_result<double>(self, log_mode);
    case DType::Float16:
      return softmax_scalar_result<Half>(self, log_mode);
    case DType::BFloat16:
      return softmax_scalar_result<BFloat16>(self, log_mode);
    default:
      TP_THROW(NotImplementedError,
               "softmax: unsupported dtype on this GPU backend");
  }
}

}  // namespace

#ifdef USE_CUDNN

Tensor softmax_native_impl(const Tensor& self, int64_t dim, bool log_mode);

// Defined after the native kernels: runs the wave/register tiers for a
// contiguous row along the fast dimension and reports whether one applied.
bool softmax_native_fast_path(const Tensor& self, Tensor& result,
                              int64_t outer_size, int64_t softmax_size,
                              int64_t inner_size, bool log_mode);

Tensor cudnn_softmax(const Tensor& self, int64_t dim, bool log) {
    int64_t ndim = self.dim();
    if (ndim == 0) {
        return softmax_scalar_result_dispatch(self, log);
    }
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError,
                 "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
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

    // An empty iteration space has nothing to normalize, and the DNN call
    // below does not accept zero-sized descriptors.
    if (input.numel() == 0) {
        return result;
    }

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

template <typename scalar_t, typename compute_t, typename out_t>
__global__ void softmax_dim_kernel(
    out_t* out, const scalar_t* in, int64_t rows, int64_t softmax_size,
    int64_t inner_size, bool log_mode) {
  // The grid is two-dimensional so a row count past the 32-bit launch limit
  // still addresses every (outer, inner) row without an overflowing cast.
  const int64_t row =
      static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  if (row >= rows) return;
  const int64_t outer = row / inner_size;
  const int64_t inner = row % inner_size;
  const scalar_t* row_in =
      in + outer * softmax_size * inner_size + inner;
  out_t* row_out = out + outer * softmax_size * inner_size + inner;

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
  // Every thread has to read the reduced max out of the tile before the sum
  // pass writes over it; without this barrier a thread that is still waiting on
  // its loads picks up another thread's partial sum in place of the max, which
  // silently rescales the whole row.
  __syncthreads();

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
        static_cast<out_t>(log_mode ? v : std::exp(v));
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

// Row length the wave kernel is built around: one lane per wave slot, at most
// this many elements per lane.  32 x 32 = 1024: past it the 64 accumulator
// slots per lane would cost more in lost occupancy than the shared-memory
// round trip they save, and the register-resident row kernel is faster there
// (2048-wide rows measure 0.9x reference on it vs 1.1x on the wave tier).
constexpr int kWaveLanes = 32;
constexpr int kWaveElemsPerLane = 32;

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
          int HW_WAVE, typename out_t>
__global__ void softmax_wave_forward(out_t* dst, const scalar_t* src,
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
          dst[i * element_count + it * kWAVE] = static_cast<out_t>(v);
        } else {
          // Elements hold exp(x - max) from the accumulation pass.
          dst[i * element_count + it * kWAVE] =
              static_cast<out_t>(elems[i][it] / sum[i]);
        }
      }
    }
  }
}

template <typename scalar_t, typename acc_t, bool LOG_MODE, typename out_t>
void launch_wave_softmax(out_t* dst, const scalar_t* src, int64_t batch_count,
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
      softmax_wave_forward<scalar_t, acc_t, L2E, LOG_MODE, 64, out_t>               \
          <<<static_cast<unsigned>(blocks), threads, 0, stream>>>(           \
              dst, src, static_cast<int>(batch_count),                       \
              static_cast<int>(element_count),                               \
              static_cast<int>(element_count));                              \
    } else {                                                                 \
      softmax_wave_forward<scalar_t, acc_t, L2E, LOG_MODE, 32, out_t>               \
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
// batches) stays on the block kernels below.
//
// The row length caps at kMaxRowLength: one lane then holds at most
// kWaveElemsPerLane accumulator slots, past which the register-resident
// slice costs more in lost occupancy than the shared-memory round trip it
// avoids.  The slots live in acc_t, so wider accumulators lower the cap
// through the byte budget below and keep the same per-lane register count.
// Rows past the cap go to the block-per-row kernels, which spread the same
// row over more threads and keep the whole slice in registers at a fraction
// of the pressure.
template <typename scalar_t, typename acc_t, bool LOG_MODE,
          typename out_t = scalar_t>
bool try_wave_softmax(const Tensor& self, Tensor& result, int64_t softmax_size,
                      int64_t rows) {
  constexpr int64_t kMaxRowLength = kWaveLanes * kWaveElemsPerLane;
  constexpr int64_t kMaxRowBytes = 8192;
  if (softmax_size <= 0 || softmax_size > kMaxRowLength) return false;
  if (softmax_size * static_cast<int64_t>(sizeof(acc_t)) > kMaxRowBytes) {
    return false;
  }
  if (rows * softmax_size > static_cast<int64_t>(INT32_MAX)) return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  launch_wave_softmax<scalar_t, acc_t, LOG_MODE, out_t>(
      result.data_ptr<out_t>(), self.data_ptr<scalar_t>(), rows,
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
  const unsigned long long mask = 0xffffffffffffffffull;
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
  const unsigned long long mask = 0xffffffffffffffffull;
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

template <typename scalar_t, typename compute_t, int PACKS, bool LOG_MODE,
          typename out_t>
__global__ void softmax_reg_packed_kernel(out_t* __restrict__ out,
                                          const scalar_t* __restrict__ in,
                                          int classes, int packets) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  constexpr int kWave = 32;
  __shared__ SoftmaxMS<compute_t> reduce_ms[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const scalar_t* row = in + static_cast<int64_t>(blockIdx.x) * classes;
  out_t* row_out = out + static_cast<int64_t>(blockIdx.x) * classes;

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
  // A thread whose slots all landed past the row end keeps the -inf fill, and
  // exp(-inf - -inf) is NaN; such a thread contributes no exponentials, which
  // leaves an all--inf row with a zero sum exactly as a row of real values
  // would report a NaN output.
  compute_t thread_sum = compute_t(0);
  if (thread_max > -std::numeric_limits<compute_t>::infinity()) {
#pragma unroll
    for (int i = 0; i < PACKS; ++i) {
#pragma unroll
      for (int k = 0; k < kPack; ++k) {
        thread_sum += std::exp(static_cast<compute_t>(v[i].v[k]) - thread_max);
      }
    }
  }
  const SoftmaxMS<compute_t> ms =
      softmax_block_reduce_ms<compute_t>(
          SoftmaxMS<compute_t>{thread_max, thread_sum}, reduce_ms);
  const compute_t norm = LOG_MODE ? std::log(ms.s) : compute_t(1) / ms.s;

  if constexpr (std::is_same_v<out_t, scalar_t>) {
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
  } else {
    // A widened output type cannot reuse the input's packet grid; the same
    // slots go out as scalar stores.
#pragma unroll
    for (int i = 0; i < PACKS; ++i) {
      const int slot = tid + i * static_cast<int>(blockDim.x);
      if (slot < packets) {
#pragma unroll
        for (int k = 0; k < kPack; ++k) {
          const compute_t x = static_cast<compute_t>(v[i].v[k]) - ms.m;
          row_out[slot * kPack + k] =
              static_cast<out_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
        }
      }
    }
  }
}

template <typename scalar_t, typename compute_t, int REG, bool LOG_MODE,
          typename out_t>
__global__ void softmax_reg_scalar_kernel(out_t* __restrict__ out,
                                          const scalar_t* __restrict__ in,
                                          int classes) {
  constexpr int kWave = 32;
  __shared__ compute_t reduce_max[kWave];
  __shared__ compute_t reduce_sum[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const scalar_t* row = in + static_cast<int64_t>(blockIdx.x) * classes;
  out_t* row_out = out + static_cast<int64_t>(blockIdx.x) * classes;

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

  // See the packed kernel: an all--inf thread would turn the exponentials
  // into NaN, so it contributes none.
  compute_t thread_sum = compute_t(0);
  if (thread_max > -std::numeric_limits<compute_t>::infinity()) {
#pragma unroll
    for (int i = 0; i < REG; ++i) {
      thread_sum += std::exp(static_cast<compute_t>(v[i]) - thread_max);
    }
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
          static_cast<out_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
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

template <typename scalar_t, typename compute_t, bool LOG_MODE,
          typename out_t = scalar_t>
bool try_reg_softmax(const Tensor& self, Tensor& result,
                     int64_t softmax_size, int64_t rows) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  if (softmax_size <= kWaveLanes * kWaveElemsPerLane || rows > INT32_MAX)
    return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  const scalar_t* in = self.data_ptr<scalar_t>();
  out_t* out = result.data_ptr<out_t>();
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
      softmax_reg_packed_kernel<scalar_t, compute_t, 1, LOG_MODE, out_t>
          <<<grid, threads, 0, stream>>>(out, in,
                                         static_cast<int>(softmax_size),
                                         packets);
    } else if (packets <= 512) {
      const int threads = static_cast<int>(wave_round((packets + 1) / 2));
      softmax_reg_packed_kernel<scalar_t, compute_t, 2, LOG_MODE, out_t>
          <<<grid, threads, 0, stream>>>(out, in,
                                         static_cast<int>(softmax_size),
                                         packets);
    } else if (packets <= 2048) {
      const int threads = static_cast<int>(wave_round((packets + 3) / 4));
      softmax_reg_packed_kernel<scalar_t, compute_t, 4, LOG_MODE, out_t>
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
              softmax_reg_packed_kernel<scalar_t, compute_t, 8, LOG_MODE,
                                        out_t>,
              threads)) {
        return false;
      }
      softmax_reg_packed_kernel<scalar_t, compute_t, 8, LOG_MODE, out_t>
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
            softmax_reg_scalar_kernel<scalar_t, compute_t, 16, LOG_MODE,
                                      out_t>,
            threads)) {
      return false;
    }
    softmax_reg_scalar_kernel<scalar_t, compute_t, 16, LOG_MODE, out_t>
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
// an additional time. Row starts that sit off the 16-byte alignment (odd
// slice offsets, row lengths that are not a packet multiple — typical
// vocabulary sizes) split the row into a scalar head, an aligned packet
// body and a scalar tail; both edges fold into the same online pair and are
// written back scalar.
template <typename scalar_t, typename compute_t, bool LOG_MODE, typename out_t>
__global__ void softmax_wide_kernel(out_t* __restrict__ out,
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
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * classes;
  const scalar_t* row_in = in + row_base;
  out_t* row_out = out + row_base;
  const auto align_head = [](const void* p) {
    return static_cast<int>(
        (16 - reinterpret_cast<uintptr_t>(p) % 16) % 16 /
        static_cast<int>(sizeof(scalar_t)));
  };
  const int head_in = align_head(row_in);
  const int body_in = classes - head_in;
  const int packets = body_in / kPack;
  const int tail_in = body_in - packets * kPack;
  const int edge_in = head_in + tail_in;
  const scalar_t pad = -std::numeric_limits<scalar_t>::infinity();
  const SoftmaxPack<scalar_t, kPack>* src =
      reinterpret_cast<const SoftmaxPack<scalar_t, kPack>*>(row_in + head_in);

  SoftmaxMS<compute_t> acc{
      -std::numeric_limits<compute_t>::infinity(), compute_t(0)};
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
  // The scalar head and tail fold into the same online pair. A lone -inf
  // element carries a zero sum, exactly like a padded packet slot; any NaN
  // poisons the sum through the combine, matching the packet path.
  for (int e = tid; e < edge_in; e += static_cast<int>(blockDim.x)) {
    const int logical = e < head_in ? e : classes - tail_in + (e - head_in);
    const compute_t x = static_cast<compute_t>(row_in[logical]);
    acc = softmax_ms_combine(
        acc, x == -std::numeric_limits<compute_t>::infinity()
                 ? SoftmaxMS<compute_t>{-std::numeric_limits<compute_t>::infinity(),
                                        compute_t(0)}
                 : SoftmaxMS<compute_t>{x, compute_t(1)});
  }
  const SoftmaxMS<compute_t> ms = softmax_block_reduce_ms<compute_t>(acc, reduce_ms);
  const compute_t norm = LOG_MODE ? std::log(ms.s) : compute_t(1) / ms.s;

  if constexpr (std::is_same_v<out_t, scalar_t>) {
    const int head_out = align_head(row_out);
    if (head_in == head_out) {
      // Fresh results share the input's per-row phase, so the write pass
      // keeps the same packet grid as the read.
      auto* dst = reinterpret_cast<SoftmaxPack<scalar_t, kPack>*>(
          row_out + head_out);
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
      for (int e = tid; e < edge_in; e += static_cast<int>(blockDim.x)) {
        const int logical = e < head_in ? e : classes - tail_in + (e - head_in);
        const compute_t x = static_cast<compute_t>(row_in[logical]) - ms.m;
        row_out[logical] =
            static_cast<out_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
      }
    } else {
      // The two row starts disagree on the alignment phase: the re-read
      // stays vectorized, but every store goes out scalar.
      for (int e = tid; e < classes; e += static_cast<int>(blockDim.x)) {
        const compute_t x = static_cast<compute_t>(row_in[e]) - ms.m;
        row_out[e] =
            static_cast<out_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
      }
    }
  } else {
    // A widened output type has its own element size and no shared packet
    // grid; the whole row goes out scalar.
    for (int e = tid; e < classes; e += static_cast<int>(blockDim.x)) {
      const compute_t x = static_cast<compute_t>(row_in[e]) - ms.m;
      row_out[e] =
          static_cast<out_t>(LOG_MODE ? x - norm : std::exp(x) * norm);
    }
  }
}

template <typename scalar_t, typename compute_t, bool LOG_MODE,
          typename out_t = scalar_t>
bool try_wide_softmax(const Tensor& self, Tensor& result,
                      int64_t softmax_size, int64_t rows) {
  if (softmax_size < 2049 || softmax_size > INT32_MAX || rows > INT32_MAX)
    return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  const scalar_t* in = self.data_ptr<scalar_t>();
  out_t* out = result.data_ptr<out_t>();
  // Misaligned row starts and row lengths that are not a packet multiple
  // stay on this kernel: the block splits each row into a scalar head, an
  // aligned packet body and a scalar tail.
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
  softmax_wide_kernel<scalar_t, compute_t, LOG_MODE, out_t>
      <<<static_cast<unsigned>(rows), threads, 0,
         getCurrentCUDAStream().stream()>>>(out, in,
                                            static_cast<int>(softmax_size));
  CUDA_CHECK(cudaGetLastError());
  return true;
}

// --- Spatial softmax for slices that sit off the fast dimension ---
//
// The row kernels above address a slice with a stride of `inner_size`, so
// threads of a warp land `inner_size` elements apart and every 32-element
// transaction carries one useful value.  Here the roles flip: threads that
// share threadIdx.y each own one inner slice (consecutive threads read
// consecutive addresses), and the x-threads of the block cooperatively
// reduce the softmax dim of that slice.

// Block reduction along the x axis only.  Threads that share threadIdx.y
// collapse their partials through a halved-stride tree inside their own
// scratch row; other y rows use disjoint scratch.  blockDim.x must be a
// power of two, which the launch configuration below guarantees.
template <typename compute_t, bool kIsMax>
__device__ __forceinline__ compute_t softmax_spatial_reduce_x(
    compute_t* shared, compute_t value) {
  compute_t* row = shared + static_cast<int64_t>(threadIdx.y) * blockDim.x;
  const int x = static_cast<int>(threadIdx.x);
  // A leading barrier keeps the previous pass's reads of the scratch row
  // ahead of this pass's writes; every thread of the block arrives.
  __syncthreads();
  row[x] = value;
  int offset = static_cast<int>(blockDim.x) / 2;
  while (offset > 0) {
    __syncthreads();
    if (x < offset) {
      const compute_t other = row[x + offset];
      value = kIsMax ? (other > value ? other : value) : (value + other);
      row[x] = value;
    }
    offset >>= 1;
  }
  __syncthreads();
  const compute_t result = row[0];
  // No trailing barrier: the next pass opens with one, and a thread cannot
  // reach it before it has read row[0] here.
  return result;
}

template <typename scalar_t, typename compute_t, bool LOG_MODE, typename out_t>
__global__ void softmax_spatial_kernel(
    out_t* __restrict__ out, const scalar_t* __restrict__ in,
    int64_t outer_size, int64_t dim_size, int64_t inner_size) {
  extern __shared__ unsigned char softmax_spatial_smem[];
  compute_t* shared = reinterpret_cast<compute_t*>(softmax_spatial_smem);
  const int64_t outer_stride = dim_size * inner_size;
  const int64_t dim_stride = inner_size;
  // The inner axis is tiled in whole rows of y-threads.  Threads past the
  // last element ride through the barriers with identity partials, so every
  // thread of the block executes the same barrier sequence; their loads and
  // stores are gated off and their results discarded.
  const int64_t inner_tiles =
      (inner_size + static_cast<int64_t>(blockDim.y) - 1) /
      static_cast<int64_t>(blockDim.y);

  for (int64_t outer_index = static_cast<int64_t>(blockIdx.x);
       outer_index < outer_size;
       outer_index += static_cast<int64_t>(gridDim.x)) {
    const int64_t outer_offset = outer_index * outer_stride;
    for (int64_t tile = static_cast<int64_t>(blockIdx.y); tile < inner_tiles;
         tile += static_cast<int64_t>(gridDim.y)) {
      const int64_t inner_index =
          tile * static_cast<int64_t>(blockDim.y) + threadIdx.y;
      const bool active = inner_index < inner_size;
      // Inactive threads still need a valid base address for the loads they
      // will run and discard.
      const int64_t data_offset = outer_offset + (active ? inner_index : 0);
      const int x = static_cast<int>(threadIdx.x);

      if (blockDim.x > 1) {
        compute_t thread_max = -std::numeric_limits<compute_t>::infinity();
        if (active) {
          for (int64_t d = x; d < dim_size;
               d += static_cast<int64_t>(blockDim.x)) {
            const compute_t v =
                static_cast<compute_t>(in[data_offset + d * dim_stride]);
            thread_max = v > thread_max ? v : thread_max;
          }
        }
        const compute_t row_max =
            softmax_spatial_reduce_x<compute_t, true>(shared, thread_max);

        compute_t thread_sum = compute_t(0);
        if (active) {
          for (int64_t d = x; d < dim_size;
               d += static_cast<int64_t>(blockDim.x)) {
            thread_sum +=
                std::exp(static_cast<compute_t>(in[data_offset + d * dim_stride]) -
                         row_max);
          }
        }
        const compute_t denom = std::log(
            softmax_spatial_reduce_x<compute_t, false>(shared, thread_sum));

        if (active) {
          for (int64_t d = x; d < dim_size;
               d += static_cast<int64_t>(blockDim.x)) {
            const compute_t v =
                static_cast<compute_t>(in[data_offset + d * dim_stride]) -
                row_max - denom;
            out[data_offset + d * dim_stride] =
                static_cast<out_t>(LOG_MODE ? v : std::exp(v));
          }
        }
      } else {
        // One thread per slice: no block reduction to coordinate.
        compute_t thread_max = -std::numeric_limits<compute_t>::infinity();
        if (active) {
          for (int64_t d = 0; d < dim_size; ++d) {
            const compute_t v =
                static_cast<compute_t>(in[data_offset + d * dim_stride]);
            thread_max = v > thread_max ? v : thread_max;
          }
        }
        compute_t thread_sum = compute_t(0);
        if (active) {
          for (int64_t d = 0; d < dim_size; ++d) {
            thread_sum +=
                std::exp(static_cast<compute_t>(in[data_offset + d * dim_stride]) -
                         thread_max);
          }
        }
        const compute_t denom = std::log(thread_sum);
        if (active) {
          for (int64_t d = 0; d < dim_size; ++d) {
            const compute_t v =
                static_cast<compute_t>(in[data_offset + d * dim_stride]) -
                thread_max - denom;
            out[data_offset + d * dim_stride] =
                static_cast<out_t>(LOG_MODE ? v : std::exp(v));
          }
        }
      }
    }
  }
}

template <typename scalar_t, typename compute_t, bool LOG_MODE,
          typename out_t = scalar_t>
bool try_spatial_softmax(const Tensor& self, Tensor& result,
                         int64_t outer_size, int64_t softmax_size,
                         int64_t inner_size) {
  if (outer_size == 0 || softmax_size == 0 || inner_size == 0) return false;
  if (!self.is_contiguous() || !result.is_contiguous()) return false;
  // Block geometry: y-threads cover the inner axis; x-threads team up on the
  // reduction only while the inner axis alone cannot fill a block and the
  // dim is long enough to feed a team.
  constexpr int kMaxThreads = 1024;
  const int inner_threads =
      static_cast<int>(inner_size < kMaxThreads ? inner_size : kMaxThreads);
  int dim_threads = 1;
  if (inner_threads <= 64 && softmax_size >= 64) {
    while (inner_threads * dim_threads <= kMaxThreads &&
           dim_threads <= softmax_size) {
      dim_threads *= 2;
    }
    dim_threads /= 2;
  }
  dim3 block(static_cast<unsigned>(dim_threads),
             static_cast<unsigned>(inner_threads));

  // Occupancy probe: shrink the block until the driver reports at least one
  // resident block for this instantiation and shared footprint.  An exhausted
  // (1, 1) block falls back to the caller's generic kernel.
  int per_sm = 0;
  auto kernel = softmax_spatial_kernel<scalar_t, compute_t, LOG_MODE, out_t>;
  size_t smem = 0;
  while (true) {
    smem = block.x > 1
        ? static_cast<size_t>(block.x) * block.y * sizeof(compute_t)
        : 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, kernel, static_cast<int>(block.x * block.y), smem);
    if (err == cudaSuccess && per_sm > 0) break;
    // A failed probe leaves a sticky per-thread error that would poison the
    // next call; clear it before retrying with a smaller block.
    if (err != cudaSuccess) cudaGetLastError();
    if (block.x == 1 && block.y == 1) return false;
    if (block.y > 1) {
      block.y /= 2;
    } else {
      block.x /= 2;
    }
  }
  static thread_local cudaDeviceProp properties{};
  static thread_local int queried_device = -1;
  const int current = currentDevice();
  if (queried_device != current) {
    CUDA_CHECK(cudaGetDeviceProperties(&properties, current));
    queried_device = current;
  }
  const int64_t max_active =
      static_cast<int64_t>(per_sm) * properties.multiProcessorCount;
  // Tile the inner axis first and spend the remaining resident blocks on the
  // outer axis; both counts stay far below the hardware launch limits after
  // the caps, so the narrowing casts below are safe.
  int64_t inner_blocks = (inner_size + block.y - 1) / block.y;
  if (inner_blocks > max_active) inner_blocks = max_active;
  int64_t outer_blocks = (max_active + inner_blocks - 1) / inner_blocks;
  if (outer_blocks > outer_size) outer_blocks = outer_size;
  const dim3 grid(static_cast<unsigned>(outer_blocks),
                  static_cast<unsigned>(inner_blocks));
  smem = block.x > 1
      ? static_cast<size_t>(block.x) * block.y * sizeof(compute_t)
      : 0;
  softmax_spatial_kernel<scalar_t, compute_t, LOG_MODE, out_t>
      <<<grid, block, smem, getCurrentCUDAStream().stream()>>>(
          result.data_ptr<out_t>(), self.data_ptr<scalar_t>(), outer_size,
          softmax_size, inner_size);
  CUDA_CHECK(cudaGetLastError());
  return true;
}

}  // namespace

template <typename scalar_t, typename compute_t, typename out_t = scalar_t>
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
        ? try_wave_softmax<scalar_t, compute_t, true, out_t>(self, result,
                                                             softmax_size,
                                                             rows)
        : try_wave_softmax<scalar_t, compute_t, false, out_t>(self, result,
                                                              softmax_size,
                                                              rows);
    if (wave) return;
    // Longer contiguous rows keep the whole slice in registers: one read
    // pass and one write pass regardless of row length.
    const bool reg = log_mode
        ? try_reg_softmax<scalar_t, compute_t, true, out_t>(self, result,
                                                            softmax_size,
                                                            rows)
        : try_reg_softmax<scalar_t, compute_t, false, out_t>(self, result,
                                                             softmax_size,
                                                             rows);
    if (reg) return;
    // Rows beyond the register budget: two streaming vector passes, with
    // both moments folded into a single read.
    const bool wide = log_mode
        ? try_wide_softmax<scalar_t, compute_t, true, out_t>(self, result,
                                                             softmax_size,
                                                             rows)
        : try_wide_softmax<scalar_t, compute_t, false, out_t>(self, result,
                                                              softmax_size,
                                                              rows);
    if (wide) return;
  }
  if (inner_size != 1) {
    // Spatial tier: consecutive threads own consecutive inner slices, so the
    // strided loads of the generic kernel below become coalesced, and the
    // inner axis finally contributes parallelism of its own.
    const bool spatial = log_mode
        ? try_spatial_softmax<scalar_t, compute_t, true, out_t>(self, result,
                                                                rows,
                                                                softmax_size,
                                                                inner_size)
        : try_spatial_softmax<scalar_t, compute_t, false, out_t>(self, result,
                                                                 rows,
                                                                 softmax_size,
                                                                 inner_size);
    if (spatial) return;
  }
  constexpr int kThreads = 256;
  const int64_t launch_rows = rows * inner_size;
  const unsigned grid_x =
      launch_rows > INT32_MAX ? static_cast<unsigned>(INT32_MAX)
                              : static_cast<unsigned>(launch_rows);
  const unsigned grid_y =
      static_cast<unsigned>((launch_rows + grid_x - 1) / grid_x);
  softmax_dim_kernel<scalar_t, compute_t, out_t>
      <<<dim3(grid_x, grid_y), kThreads, 0,
         getCurrentCUDAStream().stream()>>>(
          result.data_ptr<out_t>(), self.data_ptr<scalar_t>(),
          launch_rows, softmax_size, inner_size, log_mode);
  CUDA_CHECK(cudaGetLastError());
}

bool softmax_native_fast_path(const Tensor& self, Tensor& result,
                              int64_t outer_size, int64_t softmax_size,
                              int64_t inner_size, bool log_mode) {
  if (outer_size == 0 || softmax_size <= 0) return false;
  if (inner_size != 1) {
    // Rows strided along the fast dimension: the native spatial tier first,
    // the DNN library call below stays as the fallback for the layouts the
    // launch probe declines.
    switch (self.dtype()) {
      case DType::Float32:
        return log_mode
            ? try_spatial_softmax<float, float, true>(self, result,
                                                      outer_size, softmax_size,
                                                      inner_size)
            : try_spatial_softmax<float, float, false>(self, result,
                                                       outer_size,
                                                       softmax_size,
                                                       inner_size);
      case DType::Float64:
        return log_mode
            ? try_spatial_softmax<double, double, true>(self, result,
                                                        outer_size,
                                                        softmax_size,
                                                        inner_size)
            : try_spatial_softmax<double, double, false>(self, result,
                                                         outer_size,
                                                         softmax_size,
                                                         inner_size);
      default:
        return false;
    }
  }
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
  if (self.dim() == 0) {
    return softmax_scalar_result_dispatch(self, log_mode);
  }
  if (dim < 0) dim += self.dim();
  if (dim < 0 || dim >= self.dim()) {
    TP_THROW(RuntimeError,
             "Dimension out of range (expected to be in range of [",
             -self.dim(), ", ", self.dim() - 1, "], but got ",
             dim - self.dim(), ")");
  }
  // The row kernels below address the input with a contiguous-layout index.
  Tensor input = self.is_contiguous() ? self : self.contiguous();
  Tensor result = Tensor::empty(
      static_cast<std::vector<int64_t>>(input.shape()), input.dtype(),
      input.device());
  if (input.numel() == 0) {
    return result;
  }
  switch (input.dtype()) {
    case DType::Float32:
      softmax_dim_dispatch<float, float>(input, result, dim, log_mode);
      break;
    case DType::Float64:
      softmax_dim_dispatch<double, double>(input, result, dim, log_mode);
      break;
    case DType::Float16:
      softmax_dim_dispatch<Half, float>(input, result, dim, log_mode);
      break;
    case DType::BFloat16:
      softmax_dim_dispatch<BFloat16, float>(input, result, dim, log_mode);
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

// The half_to_float out-variant: reduced-precision input, fp32 result.  The
// accumulator is float throughout and every store lands in the wider output
// directly, so no intermediate rounding through the input dtype occurs.
// Writes go into the caller's output tensor.
void softmax_half_to_float_native(const Tensor& self, Tensor& out,
                                  int64_t dim, bool log_mode) {
  if (self.dim() == 0) {
    write_out(out, softmax_scalar_result_dispatch(self, log_mode));
    return;
  }
  if (dim < 0) dim += self.dim();
  if (dim < 0 || dim >= self.dim()) {
    TP_THROW(RuntimeError,
             "Dimension out of range (expected to be in range of [",
             -self.dim(), ", ", self.dim() - 1, "], but got ",
             dim - self.dim(), ")");
  }
  Tensor input = self.is_contiguous() ? self : self.contiguous();
  if (input.numel() == 0) return;
  switch (input.dtype()) {
    case DType::Float16:
      softmax_dim_dispatch<Half, float, float>(input, out, dim, log_mode);
      break;
    default:
      TP_THROW(NotImplementedError,
               "softmax: unsupported dtype on this GPU backend");
  }
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
  const int64_t row =
      static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
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

// Wide-row backward: rows beyond the register tier stream both operands in
// aligned packets with streaming hints, so both passes over the row stay
// bandwidth-bound.  Packet bodies require the three row starts to share one
// alignment phase; a mismatched row keeps the same two-pass shape with scalar
// accesses.
template <typename scalar_t, typename compute_t, bool LOG_MODE>
__global__ void softmax_bwd_wide_kernel(scalar_t* __restrict__ grad_in,
                                        const scalar_t* __restrict__ grad,
                                        const scalar_t* __restrict__ out,
                                        int classes) {
  constexpr int kPack = 16 / static_cast<int>(sizeof(scalar_t));
  constexpr int kChunkPackets = 16 / kPack < 1 ? 1 : 16 / kPack;
  constexpr int kWave = 32;
  __shared__ compute_t reduce[kWave];
  const int tid = static_cast<int>(threadIdx.x);
  const int64_t row_base = static_cast<int64_t>(blockIdx.x) * classes;
  const scalar_t* row_grad = grad + row_base;
  const scalar_t* row_out = out + row_base;
  scalar_t* row_in = grad_in + row_base;

  const auto align_head = [](const void* p) {
    return static_cast<int>(
        (16 - reinterpret_cast<uintptr_t>(p) % 16) % 16 /
        static_cast<int>(sizeof(scalar_t)));
  };
  const int head = align_head(row_grad);
  const bool packed =
      head == align_head(row_out) && head == align_head(row_in);
  const int body = classes - head;
  const int packets = body / kPack;
  const int tail = body - packets * kPack;
  const int edge = head + tail;
  const SoftmaxPack<scalar_t, kPack>* src_g =
      reinterpret_cast<const SoftmaxPack<scalar_t, kPack>*>(row_grad + head);
  const SoftmaxPack<scalar_t, kPack>* src_o =
      reinterpret_cast<const SoftmaxPack<scalar_t, kPack>*>(row_out + head);

  compute_t thread_sum = compute_t(0);
  if (packed) {
    const int stride = static_cast<int>(blockDim.x) * kChunkPackets;
    for (int base = tid; base < packets; base += stride) {
#pragma unroll
      for (int c = 0; c < kChunkPackets; ++c) {
        const int slot = base + c * static_cast<int>(blockDim.x);
        if (slot < packets) {
          const SoftmaxPack<scalar_t, kPack> vg =
              softmax_load_stream(src_g + slot);
          if constexpr (LOG_MODE) {
#pragma unroll
            for (int k = 0; k < kPack; ++k) {
              thread_sum += static_cast<compute_t>(vg.v[k]);
            }
          } else {
            const SoftmaxPack<scalar_t, kPack> vo =
                softmax_load_stream(src_o + slot);
#pragma unroll
            for (int k = 0; k < kPack; ++k) {
              thread_sum += static_cast<compute_t>(vg.v[k]) *
                            static_cast<compute_t>(vo.v[k]);
            }
          }
        }
      }
    }
  }
  for (int e = tid; e < (packed ? edge : classes);
       e += static_cast<int>(blockDim.x)) {
    const int logical =
        packed ? (e < head ? e : classes - tail + (e - head)) : e;
    const compute_t g = static_cast<compute_t>(row_grad[logical]);
    thread_sum += LOG_MODE
        ? g
        : g * static_cast<compute_t>(row_out[logical]);
  }
  thread_sum = softmax_block_reduce<compute_t, false>(thread_sum, reduce);
  const compute_t factor = thread_sum;

  if (packed) {
    auto* dst = reinterpret_cast<SoftmaxPack<scalar_t, kPack>*>(row_in + head);
    for (int slot = tid; slot < packets; slot += static_cast<int>(blockDim.x)) {
      const SoftmaxPack<scalar_t, kPack> vg =
          softmax_load_stream(src_g + slot);
      const SoftmaxPack<scalar_t, kPack> vo =
          softmax_load_stream(src_o + slot);
      SoftmaxPack<scalar_t, kPack> r;
#pragma unroll
      for (int k = 0; k < kPack; ++k) {
        const compute_t g = static_cast<compute_t>(vg.v[k]);
        const compute_t o = static_cast<compute_t>(vo.v[k]);
        r.v[k] = static_cast<scalar_t>(
            LOG_MODE ? g - std::exp(o) * factor : o * (g - factor));
      }
      softmax_store_stream(dst + slot, r);
    }
  }
  for (int e = tid; e < (packed ? edge : classes);
       e += static_cast<int>(blockDim.x)) {
    const int logical =
        packed ? (e < head ? e : classes - tail + (e - head)) : e;
    const compute_t g = static_cast<compute_t>(row_grad[logical]);
    const compute_t o = static_cast<compute_t>(row_out[logical]);
    row_in[logical] = static_cast<scalar_t>(
        LOG_MODE ? g - std::exp(o) * factor : o * (g - factor));
  }
}

// Takes the contiguous rows the register tier declines; the caller falls
// back to the strided kernel when the launch geometry does not fit.
template <typename scalar_t, typename compute_t>
bool try_softmax_bwd_wide(scalar_t* grad_in, const scalar_t* grad,
                          const scalar_t* out, int64_t classes, int64_t rows,
                          bool log_mode, cudaStream_t stream) {
  constexpr int kRegCap = 16 * 512;
  if (classes <= kRegCap || classes > INT32_MAX || rows > INT32_MAX)
    return false;
  static thread_local cudaDeviceProp properties{};
  static thread_local int queried_device = -1;
  const int current = currentDevice();
  if (queried_device != current) {
    CUDA_CHECK(cudaGetDeviceProperties(&properties, current));
    queried_device = current;
  }
  const int threads =
      rows * 2 < properties.multiProcessorCount ? 1024 : 512;
  if (log_mode) {
    softmax_bwd_wide_kernel<scalar_t, compute_t, true>
        <<<static_cast<unsigned>(rows), threads, 0, stream>>>(
            grad_in, grad, out, static_cast<int>(classes));
  } else {
    softmax_bwd_wide_kernel<scalar_t, compute_t, false>
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
                       oc.data_ptr<float>(), dim_size, rows, log_mode,
                       stream) ||
                   try_softmax_bwd_wide<float, float>(
                       result_work.data_ptr<float>(), gc.data_ptr<float>(),
                       oc.data_ptr<float>(), dim_size, rows, log_mode,
                       stream);
        break;
      case DType::Float64:
        reg_done = try_softmax_bwd_reg<double, double>(
                       result_work.data_ptr<double>(), gc.data_ptr<double>(),
                       oc.data_ptr<double>(), dim_size, rows, log_mode,
                       stream) ||
                   try_softmax_bwd_wide<double, double>(
                       result_work.data_ptr<double>(), gc.data_ptr<double>(),
                       oc.data_ptr<double>(), dim_size, rows, log_mode,
                       stream);
        break;
      case DType::Float16:
        reg_done = try_softmax_bwd_reg<Half, float>(
                       result_work.data_ptr<Half>(), gc.data_ptr<Half>(),
                       oc.data_ptr<Half>(), dim_size, rows, log_mode,
                       stream) ||
                   try_softmax_bwd_wide<Half, float>(
                       result_work.data_ptr<Half>(), gc.data_ptr<Half>(),
                       oc.data_ptr<Half>(), dim_size, rows, log_mode, stream);
        break;
      case DType::BFloat16:
        reg_done = try_softmax_bwd_reg<BFloat16, float>(
                       result_work.data_ptr<BFloat16>(), gc.data_ptr<BFloat16>(),
                       oc.data_ptr<BFloat16>(), dim_size, rows, log_mode,
                       stream) ||
                   try_softmax_bwd_wide<BFloat16, float>(
                       result_work.data_ptr<BFloat16>(), gc.data_ptr<BFloat16>(),
                       oc.data_ptr<BFloat16>(), dim_size, rows, log_mode,
                       stream);
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

  const unsigned bwd_grid_x =
      rows > INT32_MAX ? static_cast<unsigned>(INT32_MAX)
                       : static_cast<unsigned>(rows);
  const unsigned bwd_grid_y =
      static_cast<unsigned>((rows + bwd_grid_x - 1) / bwd_grid_x);

  #define TP_SOFTMAX_BWD_LAUNCH(ctype, acc)                                \
  if (log_mode) {                                                          \
    constexpr bool LOG_MODE = true;                                        \
    softmax_backward_data_kernel<ctype, acc, LOG_MODE>                     \
        <<<dim3(bwd_grid_x, bwd_grid_y), threads, 0,                       \
           getCurrentCUDAStream().stream()>>>(                             \
            result_work.data_ptr<ctype>(), gc.data_ptr<ctype>(),           \
            oc.data_ptr<ctype>(), rows, dim_size, inner);                  \
  } else {                                                                 \
    constexpr bool LOG_MODE = false;                                       \
    softmax_backward_data_kernel<ctype, acc, LOG_MODE>                     \
        <<<dim3(bwd_grid_x, bwd_grid_y), threads, 0,                       \
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
    if (self.dtype() != DType::Float16) {
      TP_THROW(RuntimeError, "conversion is supported for Half type only");
    }
    if (out.dtype() != DType::Float32) {
      TP_THROW(TypeError,
               "half_to_float softmax expects a float32 out tensor");
    }
    softmax_half_to_float_native(self, out, dim, /*log_mode=*/false);
    return out;
  }
  write_out(out, softmax_native_impl(self, dim, false));
  return out;
}

Tensor& _log_softmax_out_cuda(const Tensor& self, int64_t dim,
                              bool half_to_float, Tensor& out) {
  if (half_to_float) {
    if (self.dtype() != DType::Float16) {
      TP_THROW(RuntimeError, "conversion is supported for Half type only");
    }
    if (out.dtype() != DType::Float32) {
      TP_THROW(TypeError,
               "half_to_float log_softmax expects a float32 out tensor");
    }
    softmax_half_to_float_native(self, out, dim, /*log_mode=*/true);
    return out;
  }
  write_out(out, softmax_native_impl(self, dim, true));
  return out;
}

namespace ops = tensorplay::tpx::ops;

namespace {

// Fused masked softmax for rows laid out along the fastest dimension: the
// mask is consulted while the row statistics are gathered, so dropped
// entries never materialize as -inf logits and no masked_fill temporary is
// needed.  One warp owns one row; each lane strides the row with a 32-wide
// step, so the row must stay within the per-lane register budget of the
// caller's launch bound.  A row whose entries are all dropped carries no
// probability mass and answers zero; a row whose kept entries are all -inf
// still divides zero mass and answers NaN, like the unfused rewrite.
template <typename scalar_t, typename compute_t>
__global__ void masked_softmax_wave_kernel(scalar_t* __restrict__ out,
                                           const scalar_t* __restrict__ in,
                                           const bool* __restrict__ mask,
                                           int elements, int64_t rows) {
  constexpr int kWave = 32;
  const unsigned long long full = 0xffffffffffffffffull;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int warp = static_cast<int>(threadIdx.x) / kWave;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWave;
  const int64_t block_rows = static_cast<int64_t>(gridDim.x) * warps_per_block;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
       row < rows; row += block_rows) {
    const int64_t base = row * elements;
    const scalar_t* row_in = in + base;
    const bool* row_mask = mask + base;
    scalar_t* row_out = out + base;

    compute_t m = -std::numeric_limits<compute_t>::infinity();
    unsigned kept = 0u;
    for (int j = lane; j < elements; j += kWave) {
      if (row_mask[j]) continue;
      kept = 1u;
      const compute_t x = static_cast<compute_t>(row_in[j]);
      m = x > m ? x : m;
    }
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2) {
      const compute_t other = __shfl_xor_sync(full, m, offset, kWave);
      m = other > m ? other : m;
      kept |= __shfl_xor_sync(full, kept, offset, kWave);
    }
    if (!kept) {
      // Every entry dropped: no probability mass, answer zero everywhere.
      for (int j = lane; j < elements; j += kWave) row_out[j] = scalar_t(0);
      continue;
    }
    compute_t s = compute_t(0);
    for (int j = lane; j < elements; j += kWave) {
      if (row_mask[j]) continue;
      s += std::exp(static_cast<compute_t>(row_in[j]) - m);
    }
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2) {
      s += __shfl_xor_sync(full, s, offset, kWave);
    }
    const compute_t inv = compute_t(1) / s;
    for (int j = lane; j < elements; j += kWave) {
      row_out[j] = row_mask[j]
          ? scalar_t(0)
          : static_cast<scalar_t>(
                std::exp(static_cast<compute_t>(row_in[j]) - m) * inv);
    }
  }
}

// Fused masked backward: the dot product runs over the kept entries only and
// dropped positions answer zero, matching the unfused rewrite where both
// operands are zeroed under the mask first.
template <typename scalar_t, typename compute_t>
__global__ void masked_softmax_bwd_wave_kernel(
    scalar_t* __restrict__ grad_in, const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ out, const bool* __restrict__ mask,
    int elements, int64_t rows) {
  constexpr int kWave = 32;
  const unsigned long long full = 0xffffffffffffffffull;
  const int lane = static_cast<int>(threadIdx.x) % kWave;
  const int warp = static_cast<int>(threadIdx.x) / kWave;
  const int warps_per_block = static_cast<int>(blockDim.x) / kWave;
  const int64_t block_rows = static_cast<int64_t>(gridDim.x) * warps_per_block;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
       row < rows; row += block_rows) {
    const int64_t base = row * elements;
    const scalar_t* row_grad = grad + base;
    const scalar_t* row_out = out + base;
    const bool* row_mask = mask + base;
    scalar_t* row_in = grad_in + base;

    compute_t partial = compute_t(0);
    for (int j = lane; j < elements; j += kWave) {
      if (row_mask[j]) continue;
      partial += static_cast<compute_t>(row_grad[j]) *
                 static_cast<compute_t>(row_out[j]);
    }
#pragma unroll
    for (int offset = kWave / 2; offset > 0; offset /= 2) {
      partial += __shfl_xor_sync(full, partial, offset, kWave);
    }
    const compute_t dot = partial;
    for (int j = lane; j < elements; j += kWave) {
      if (row_mask[j]) {
        row_in[j] = scalar_t(0);
        continue;
      }
      const compute_t g = static_cast<compute_t>(row_grad[j]);
      const compute_t o = static_cast<compute_t>(row_out[j]);
      row_in[j] = static_cast<scalar_t>(o * (g - dot));
    }
  }
}

// The mask a caller may hand over for a padding mask (type 1): one row per
// batch entry, (B, L), covering every head and query of a (B, H, L, L)
// input.  Anything else is used as it is and left to broadcasting.
Tensor masked_softmax_mask_view(const Tensor& self, const Tensor& mask,
                                std::optional<int64_t> mask_type) {
  if (mask_type.has_value() && *mask_type == 1 && mask.dim() == 2 &&
      self.dim() == 4) {
    TP_CHECK(self.size(0) == mask.size(0) && self.size(2) == mask.size(1),
             "For mask_type == 1 mask shape should be (B, L)");
    return ops::view(mask,
                     {mask.size(0), 1, 1, mask.size(1)});
  }
  return mask;
}

template <typename scalar_t, typename compute_t>
bool launch_masked_softmax_wave(const Tensor& self, Tensor& result,
                                const Tensor& mask, int64_t elements) {
  const int64_t rows = self.numel() / elements;
  constexpr int kWave = 32;
  const int threads = 128;
  const int warps_per_block = threads / kWave;
  const unsigned grid = static_cast<unsigned>(
      (rows + warps_per_block - 1) / warps_per_block);
  masked_softmax_wave_kernel<scalar_t, compute_t>
      <<<grid, threads, 0, getCurrentCUDAStream().stream()>>>(
          result.data_ptr<scalar_t>(), self.data_ptr<scalar_t>(),
          mask.data_ptr<bool>(), static_cast<int>(elements), rows);
  CUDA_CHECK(cudaGetLastError());
  return true;
}

template <typename scalar_t, typename compute_t>
bool launch_masked_softmax_bwd_wave(const Tensor& grad, const Tensor& output,
                                    const Tensor& mask, Tensor& result,
                                    int64_t elements) {
  const int64_t rows = grad.numel() / elements;
  constexpr int kWave = 32;
  const int threads = 128;
  const int warps_per_block = threads / kWave;
  const unsigned grid = static_cast<unsigned>(
      (rows + warps_per_block - 1) / warps_per_block);
  masked_softmax_bwd_wave_kernel<scalar_t, compute_t>
      <<<grid, threads, 0, getCurrentCUDAStream().stream()>>>(
          result.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
          output.data_ptr<scalar_t>(), mask.data_ptr<bool>(),
          static_cast<int>(elements), rows);
  CUDA_CHECK(cudaGetLastError());
  return true;
}

bool masked_softmax_can_fuse(const Tensor& self, const Tensor& mask,
                             int64_t d) {
  if (self.dim() < 1 || d != self.dim() - 1) return false;
  const int64_t elements = self.size(d);
  if (elements <= 0 || elements > 1024) return false;
  if (elements * static_cast<int64_t>(self.itemsize()) > 8192) return false;
  if (self.numel() == 0 || self.numel() / elements > INT32_MAX) return false;
  if (!self.is_contiguous() || !mask.is_contiguous()) return false;
  if (mask.dim() != self.dim()) return false;
  return static_cast<std::vector<int64_t>>(mask.shape()) ==
         static_cast<std::vector<int64_t>>(self.shape());
}

}  // namespace

Tensor _masked_softmax_cuda(const Tensor& self, const Tensor& mask,
                            std::optional<int64_t> dim,
                            std::optional<int64_t> mask_type) {
  TP_CHECK(mask.dtype() == DType::Bool, "Mask should be a boolean tensor");
  const int64_t nd = self.dim();
  int64_t d = -1;
  if (dim.has_value()) {
    d = *dim < 0 ? *dim + nd : *dim;
  }
  if (nd >= 1 && (dim.has_value() ? d : nd - 1) == nd - 1 &&
      masked_softmax_can_fuse(self, mask, nd - 1)) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(self.shape()), self.dtype(),
        self.device());
    const int64_t elements = self.size(nd - 1);
    switch (self.dtype()) {
      case DType::Float32:
        launch_masked_softmax_wave<float, float>(self, result, mask, elements);
        break;
      case DType::Float64:
        launch_masked_softmax_wave<double, double>(self, result, mask,
                                                   elements);
        break;
      case DType::Float16:
        launch_masked_softmax_wave<Half, float>(self, result, mask, elements);
        break;
      case DType::BFloat16:
        launch_masked_softmax_wave<BFloat16, float>(self, result, mask,
                                                    elements);
        break;
      default:
        TP_THROW(NotImplementedError,
                 "masked_softmax: unsupported dtype on this GPU backend");
    }
    return result;
  }
  // Same rewrite the backend-neutral composite performs: mask the logits to
  // -inf, normalize, then answer zero wherever the mask drops an entry.
  const Tensor dropped = masked_softmax_mask_view(self, mask, mask_type);
  Tensor neg_inf =
      ops::full_like(self, Scalar(-std::numeric_limits<double>::infinity()));
  Tensor out = ops::softmax(ops::where(dropped, neg_inf, self), dim.value_or(-1),
                            DType::Undefined);
  return ops::where(dropped, ops::zeros_like(self), out);
}

Tensor _masked_softmax_backward_cuda(const Tensor& grad_output,
                                     const Tensor& output, const Tensor& mask,
                                     std::optional<int64_t> dim) {
  TP_CHECK(mask.dtype() == DType::Bool, "Mask should be a boolean tensor");
  const int64_t nd = grad_output.dim();
  int64_t d = -1;
  if (dim.has_value()) {
    d = *dim < 0 ? *dim + nd : *dim;
  }
  const bool can_fuse =
      nd >= 1 && (dim.has_value() ? d : nd - 1) == nd - 1 &&
      grad_output.is_contiguous() && output.is_contiguous() &&
      mask.is_contiguous() && mask.dim() == nd &&
      static_cast<std::vector<int64_t>>(mask.shape()) ==
          static_cast<std::vector<int64_t>>(grad_output.shape()) &&
      static_cast<std::vector<int64_t>>(output.shape()) ==
          static_cast<std::vector<int64_t>>(grad_output.shape());
  if (can_fuse) {
    const int64_t elements = grad_output.size(nd - 1);
    if (elements > 0 && elements <= 1024 &&
        elements * static_cast<int64_t>(grad_output.itemsize()) <= 8192 &&
        grad_output.numel() / elements <= INT32_MAX &&
        grad_output.dtype() == output.dtype()) {
      Tensor result = Tensor::empty(
          static_cast<std::vector<int64_t>>(grad_output.shape()),
          grad_output.dtype(), grad_output.device());
      switch (grad_output.dtype()) {
        case DType::Float32:
          launch_masked_softmax_bwd_wave<float, float>(
              grad_output, output, mask, result, elements);
          break;
        case DType::Float64:
          launch_masked_softmax_bwd_wave<double, double>(
              grad_output, output, mask, result, elements);
          break;
        case DType::Float16:
          launch_masked_softmax_bwd_wave<Half, float>(grad_output, output,
                                                      mask, result, elements);
          break;
        case DType::BFloat16:
          launch_masked_softmax_bwd_wave<BFloat16, float>(grad_output, output,
                                                          mask, result,
                                                          elements);
          break;
        default:
          TP_THROW(NotImplementedError,
                   "masked_softmax: unsupported dtype on this GPU backend");
      }
      return result;
    }
  }
  // Unfused rewrite: zero both operands under the mask, take the dot over
  // the kept entries, and let dropped positions pass nothing through.
  Tensor g = ops::where(mask, ops::zeros_like(grad_output), grad_output);
  Tensor o = ops::where(mask, ops::zeros_like(output), output);
  Tensor dot = ops::sum(ops::mul(g, o), {dim.value_or(-1)}, true);
  return ops::where(mask, ops::zeros_like(grad_output),
                    ops::mul(o, ops::sub(g, dot)));
}


TENSORPLAY_LIBRARY_IMPL(CUDA, SoftmaxKernels) {
    m.impl("_masked_softmax", _masked_softmax_cuda);
    m.impl("_masked_softmax_backward", _masked_softmax_backward_cuda);
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
