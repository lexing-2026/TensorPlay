// Stateless key derivation: producing a fresh (seed, offset) from an existing
// one, without any state carried between calls.
//
// The other random kernels here draw from the stateful generator, which
// reserves counter values per launch and therefore needs a host-side generator
// and cannot be read at a position chosen by the caller.  What a graph needs is
// the opposite: a position chosen by the caller, read the same way however many
// values came before.  These four are that -- a (seed, offset) pair is a
// function of its own, so the same pair gives the same values on every run and
// on every device.

#include "Tensor.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "StatelessPhilox4x32.cuh"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

// One thread per (split, key) pair, walking them in a grid-stride loop so the
// launch is bounded by the device rather than by the number of pairs.
__global__ void philox_key_split_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    int64_t num_splits) {
  int64_t total = num_keys * num_splits;
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; tid < total; tid += stride) {
    int64_t split_idx = tid / num_keys;
    int64_t key_idx = tid % num_keys;

    uint64_t seed = input[key_idx * 2];
    uint64_t offset = input[key_idx * 2 + 1];

    // Read four values at a position derived from this split, and use them as
    // the next key: which is what makes the splits independent of each other
    // rather than four reads of one stream.
    uint4 r = philox_4x32(seed, offset + static_cast<uint64_t>(split_idx));
    int64_t out = (split_idx * num_keys + key_idx) * 2;
    philox_derive_key(r, &output[out], &output[out + 1]);
  }
}

__device__ __forceinline__ void philox_key_fold_in_impl(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    uint64_t data) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (; idx < num_keys; idx += stride) {
    uint64_t seed = input[idx * 2];
    uint64_t offset = input[idx * 2 + 1];
    uint4 r = philox_4x32(seed, offset + data);
    philox_derive_key(r, &output[idx * 2], &output[idx * 2 + 1]);
  }
}

// The folded-in value arrives as a number baked into the launch, which a
// captured graph would freeze at whatever it was when the graph was recorded.
__global__ void philox_key_fold_in_scalar_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    uint64_t data) {
  philox_key_fold_in_impl(input, output, num_keys, data);
}

// The folded-in value is read from device memory at execution time instead, so
// a captured graph reads whatever is there when it runs.
__global__ void philox_key_fold_in_tensor_kernel(
    const uint64_t* __restrict__ input,
    uint64_t* __restrict__ output,
    int64_t num_keys,
    const uint64_t* __restrict__ data) {
  philox_key_fold_in_impl(input, output, num_keys, data[0]);
}

void check_key_shape(const Tensor& key, const char* name) {
  if (!(key.dim() >= 1 && key.size(-1) == 2)) {
    TP_THROW(RuntimeError, std::string(name) +
             ": key must have shape (*batch, 2), got a shape whose last "
             "dimension is not 2");
  }
  if (key.scalar_type() != kUInt64) {
    TP_THROW(RuntimeError, std::string(name) +
             ": key must be 64-bit unsigned, got some other type");
  }
}

void check_launch(cudaError_t error) {
  if (error != cudaSuccess) {
    TP_THROW(RuntimeError,
             std::string("CUDA Error: ") + cudaGetErrorString(error));
  }
}

} // anonymous namespace

Tensor _philox_key_split_cuda(const Tensor& key, int64_t num_splits) {
  check_key_shape(key, "_philox_key_split");
  if (num_splits <= 0) {
    TP_THROW(RuntimeError, "_philox_key_split: num_splits must be positive");
  }

  // The splits are a new dimension outermost, so that splitting a batch of
  // keys puts the choice of split outside the batch rather than inside it.
  std::vector<int64_t> output_sizes;
  output_sizes.reserve(key.dim() + 1);
  output_sizes.push_back(num_splits);
  for (int64_t i = 0; i < key.dim(); i++) {
    output_sizes.push_back(key.size(i));
  }
  Tensor output = Tensor::empty(output_sizes, key.dtype(), key.device());

  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  int64_t total_threads = num_keys * num_splits;
  constexpr int block_size = 256;
  int num_blocks = static_cast<int>(
      std::min((total_threads + block_size - 1) / block_size,
               static_cast<int64_t>(65535)));

  auto key_contig = key.contiguous();
  philox_key_split_kernel<<<num_blocks, block_size, 0,
                            getCurrentCUDAStream().stream()>>>(
      key_contig.data_ptr<uint64_t>(), output.data_ptr<uint64_t>(), num_keys,
      num_splits);
  check_launch(cudaGetLastError());

  return output;
}

Tensor _philox_key_fold_in_cuda(const Tensor& key, int64_t data) {
  check_key_shape(key, "_philox_key_fold_in");
  Tensor output = Tensor::empty(std::vector<int64_t>(key.shape().vec()),
                                 key.dtype(), key.device());
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  constexpr int block_size = 256;
  int num_blocks =
      static_cast<int>((num_keys + block_size - 1) / block_size);

  auto key_contig = key.contiguous();
  // The schema carries the value as a signed number; the offset it is folded
  // into is not signed, so it is read as the same bits.
  philox_key_fold_in_scalar_kernel<<<num_blocks, block_size, 0,
                                    getCurrentCUDAStream().stream()>>>(
      key_contig.data_ptr<uint64_t>(), output.data_ptr<uint64_t>(), num_keys,
      static_cast<uint64_t>(data));
  check_launch(cudaGetLastError());

  return output;
}

Tensor _philox_key_fold_in_tensor_cuda(const Tensor& key, const Tensor& data) {
  check_key_shape(key, "_philox_key_fold_in");
  if (data.scalar_type() != kUInt64) {
    TP_THROW(RuntimeError,
             "_philox_key_fold_in: data must be 64-bit unsigned");
  }
  if (data.numel() != 1) {
    TP_THROW(RuntimeError,
             "_philox_key_fold_in: data must be a single value");
  }
  Tensor output = Tensor::empty(std::vector<int64_t>(key.shape().vec()),
                                 key.dtype(), key.device());
  int64_t num_keys = key.numel() / 2;
  if (num_keys == 0) {
    return output;
  }

  constexpr int block_size = 256;
  int num_blocks =
      static_cast<int>((num_keys + block_size - 1) / block_size);

  auto key_contig = key.contiguous();
  philox_key_fold_in_tensor_kernel<<<num_blocks, block_size, 0,
                                    getCurrentCUDAStream().stream()>>>(
      key_contig.data_ptr<uint64_t>(), output.data_ptr<uint64_t>(), num_keys,
      data.const_data_ptr<uint64_t>());
  check_launch(cudaGetLastError());

  return output;
}

// ---- turning four drawn numbers into a distribution ----------------------
//
// A drawn number is a whole number in a fixed range, not a fraction of one, so
// turning it into a real number is a scale by one over that range: the value
// that comes out is in [0, 1) and can never be the top of the range, which is
// what makes it safe to scale and shift afterwards without ever landing on the
// end the caller said the values are less than.

namespace {

template <typename T, typename V>
__device__ __forceinline__ T uniform_real(V val, T from, T to) {
  // Only as many of the drawn bits as the result type holds exactly: a
  // single-precision number holds 24 consecutive integers, so using more of
  // them would round neighbouring draws onto the same value.
  using bits_t = typename std::conditional<sizeof(V) == 8, uint64_t, uint32_t>::type;
  constexpr bits_t kMask = (static_cast<bits_t>(1) << std::numeric_limits<T>::digits) - 1;
  constexpr T kDivisor =
      static_cast<T>(1) / (static_cast<bits_t>(1) << std::numeric_limits<T>::digits);
  T x = static_cast<T>(static_cast<bits_t>(val) & kMask) * kDivisor;
  return x * (to - from) + from;
}

// Four drawn numbers become four standard normals: two of them give a radius
// and two give an angle, and a point at that radius and angle is one of the two
// answers -- the other being the same point turned by a quarter turn, which is
// free once the sine and the cosine are both needed anyway.  Two radii come out
// of four numbers this way, so all four are used.
__device__ __forceinline__ void box_muller_float(uint4 r, float* out) {
  constexpr float kInvRange = 2.3283064365386963e-10f;  // 2^-32
  constexpr float kTwoPi = 6.2831853071795864f;
  // The first of each pair is shifted off zero before its logarithm, because a
  // logarithm of zero is not a number and the draw can be zero.
  float u1 = fmaf(static_cast<float>(r.x), kInvRange, kInvRange * 0.5f);
  float u2 = fmaf(static_cast<float>(r.y), kInvRange, kInvRange * 0.5f);
  float u3 = fmaf(static_cast<float>(r.z), kInvRange, kInvRange * 0.5f);
  float u4 = fmaf(static_cast<float>(r.w), kInvRange, kInvRange * 0.5f);

  float radius1 = sqrtf(-2.0f * logf(u1));
  float radius2 = sqrtf(-2.0f * logf(u3));
  out[0] = radius1 * cosf(kTwoPi * u2);
  out[1] = radius1 * sinf(kTwoPi * u2);
  out[2] = radius2 * cosf(kTwoPi * u4);
  out[3] = radius2 * sinf(kTwoPi * u4);
}

// Two doubles out of four drawn numbers, where each of the two needs a
// fraction of the precision a whole number carries -- so the four are paired
// up before being scaled rather than each being scaled on its own.
__device__ __forceinline__ void box_muller_double(uint4 r, double* out) {
  constexpr double kInvRange = 2.3283064365386963e-10;  // 2^-32
  constexpr double kTwoPi = 6.2831853071795864;
  double u1 = fma(static_cast<double>(r.x), kInvRange,
                  static_cast<double>(r.y) * kInvRange * kInvRange +
                      kInvRange * kInvRange * 0.5);
  double u2 = fma(static_cast<double>(r.z), kInvRange,
                  static_cast<double>(r.w) * kInvRange * kInvRange +
                      kInvRange * kInvRange * 0.5);

  double radius = sqrt(-2.0 * log(u1));
  out[0] = radius * cos(kTwoPi * u2);
  out[1] = radius * sin(kTwoPi * u2);
}

// One thread per group of drawn numbers.  A group of four is one read of the
// stream, and an element takes one value from the group it falls in, so the
// position the stream is read at is the element's position divided by the size
// of a group -- a position a thread is not going to read must not be consumed,
// or the values would depend on how many threads there were.
template <typename scalar_t, int elems_per_call, typename sample_func_t,
          typename param_func_t>
__global__ void philox_single_key_kernel(
    scalar_t* __restrict__ output,
    const uint64_t* __restrict__ key,
    int64_t numel,
    sample_func_t sample_func,
    param_func_t param_func) {
  constexpr int64_t kGroup = elems_per_call;
  const int64_t group_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x;
  const int64_t groups = (numel + kGroup - 1) / kGroup;
  if (group_idx >= groups) {
    return;
  }
  uint4 r = philox_4x32(key[0], key[1] + static_cast<uint64_t>(group_idx) * kGroup);
  auto sample = sample_func(r);
  const int64_t base = group_idx * kGroup;
  #pragma unroll
  for (int j = 0; j < kGroup; j++) {
    if (base + j < numel) {
      output[base + j] = param_func(sample[j]);
    }
  }
}

void check_distribution_shapes(
    const Tensor& self, const Tensor& key, const char* name) {
  if (!self.is_floating_point()) {
    TP_THROW(RuntimeError, std::string(name) +
             ": the destination must hold real numbers");
  }
  if (self.device() != key.device()) {
    TP_THROW(RuntimeError,
             std::string(name) + ": destination and key must be on one device");
  }
  if (!(key.dim() == 1 && key.size(0) == 2)) {
    TP_THROW(RuntimeError, std::string(name) +
             ": key must be one (seed, offset) pair; a batch of them is a "
             "different operation and not yet written");
  }
}

} // anonymous namespace

Tensor _philox_uniform_cuda_(Tensor& self, const Tensor& key, double low,
                              double high) {
  check_distribution_shapes(self, key, "_philox_uniform_");
  if (self.numel() == 0) {
    return self;
  }
  auto key_contig = key.contiguous();
  const int64_t n = self.numel();
  constexpr int block_size = 256;
  const int num_blocks =
      static_cast<int>((n + block_size - 1) / block_size);
  cudaStream_t stream = getCurrentCUDAStream().stream();
  const uint64_t* kp = key_contig.data_ptr<uint64_t>();

  if (self.scalar_type() == kFloat) {
    philox_single_key_kernel<float, 4>
        <<<num_blocks, block_size, 0, stream>>>(
            self.data_ptr<float>(), kp, n,
            [] __device__(uint4 r) {
              // The drawn numbers are handed on as they are: turning one into
              // a real number here and back again inside the transform would
              // round it twice.
              uint32_t s[4];
              #pragma unroll
              for (int j = 0; j < 4; j++) s[j] = (&r.x)[j];
              return s;
            },
            [low, high] __device__(uint32_t v) {
              return uniform_real<float>(v, static_cast<float>(low),
                                         static_cast<float>(high));
            });
  } else if (self.scalar_type() == kDouble) {
    philox_single_key_kernel<double, 2>
        <<<num_blocks, block_size, 0, stream>>>(
            self.data_ptr<double>(), kp, n,
            [] __device__(uint4 r) {
              // A whole number carries more precision than a real number can
              // keep, so each of the two is a pair of the four packed into one
              // rather than one of them scaled.
              uint64_t s[2];
              s[0] = (static_cast<uint64_t>(r.x) << 32) | r.y;
              s[1] = (static_cast<uint64_t>(r.z) << 32) | r.w;
              return s;
            },
            [low, high] __device__(uint64_t v) {
              return uniform_real<double>(v, low, high);
            });
  } else {
    TP_THROW(RuntimeError,
             "_philox_uniform_: only single and double precision are written");
  }
  check_launch(cudaGetLastError());
  return self;
}

Tensor _philox_normal_cuda_(Tensor& self, const Tensor& key, double mean,
                            double stddev) {
  check_distribution_shapes(self, key, "_philox_normal_");
  if (self.numel() == 0) {
    return self;
  }
  auto key_contig = key.contiguous();
  const int64_t n = self.numel();
  constexpr int block_size = 256;
  const int num_blocks =
      static_cast<int>((n + block_size - 1) / block_size);
  cudaStream_t stream = getCurrentCUDAStream().stream();
  const uint64_t* kp = key_contig.data_ptr<uint64_t>();

  if (self.scalar_type() == kFloat) {
    philox_single_key_kernel<float, 4><<<num_blocks, block_size, 0, stream>>>(
        self.data_ptr<float>(), kp, n,
        [] __device__(uint4 r) {
          float s[4];
          box_muller_float(r, s);
          return s;
        },
        [mean, stddev] __device__(float v) {
          return static_cast<float>(v * stddev + mean);
        });
  } else if (self.scalar_type() == kDouble) {
    philox_single_key_kernel<double, 2><<<num_blocks, block_size, 0, stream>>>(
        self.data_ptr<double>(), kp, n,
        [] __device__(uint4 r) {
          double s[2];
          box_muller_double(r, s);
          return s;
        },
        [mean, stddev] __device__(double v) {
          return v * stddev + mean;
        });
  } else {
    TP_THROW(RuntimeError,
             "_philox_normal_: only single and double precision are written");
  }
  check_launch(cudaGetLastError());
  return self;
}

} // namespace cuda
} // namespace tensorplay
