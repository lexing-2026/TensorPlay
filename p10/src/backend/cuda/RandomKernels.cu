#include "Tensor.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "CUDAGenerator.h"
#include "Exception.h"
#include "Utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>

namespace tensorplay {
namespace cuda {

// Random number generation mirrors torch's CUDA distribution kernels
// (ATen/native/cuda/DistributionTemplates.h): a grid-stride kernel where each
// thread drives an independent curandStatePhilox4_32_10_t subsequence seeded
// with (seed, thread index, offset). The (seed, offset) pair comes from the
// host-side generator, which reserves counter values per launch, so results
// are independent of launch geometry and reproducible across runs. During a
// CUDA graph capture the pair instead lives in device memory owned by the
// capturing graph (PhiloxCudaState pointer mode) and is refreshed before each
// replay, so replays draw fresh randoms (see CUDAGenerator.h).

namespace {

// Unpacks PhiloxCudaState into the effective (seed, offset) this launch
// consumes; in pointer mode the device buffer is dereferenced at kernel
// execution time, both during capture and on every later replay.
__device__ inline void philox_unpack(const PhiloxCudaState& state,
                                     uint64_t* seed, uint64_t* offset) {
    if (state.captured) {
        *seed = *state.seed_dev;
        *offset = *state.offset_dev + state.offset_intragraph;
    } else {
        *seed = state.seed;
        *offset = state.offset;
    }
}

constexpr uint32_t kBlockSize = 256;
// curand device API consumes at most 4 counter values per call.
constexpr uint64_t kMaxGeneratorOffsetsPerCall = 4;

uint32_t deviceAttribute(cudaDeviceAttr attr) {
    int value = 0;
    int device_index = 0;
    cudaGetDevice(&device_index);
    cudaError_t error = cudaDeviceGetAttribute(&value, attr, device_index);
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("cudaDeviceGetAttribute failed: ") +
                 cudaGetErrorString(error));
    }
    return static_cast<uint32_t>(value);
}

// Utility function that calculates the proper philox_offset, mirroring
// torch's calc_execution_policy.
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(int64_t total_elements,
                                                       uint32_t unroll_factor) {
    const uint64_t numel = static_cast<uint64_t>(total_elements);
    const uint32_t block_size = kBlockSize;
    dim3 dim_block(block_size);
    dim3 grid(static_cast<uint32_t>((numel + block_size - 1) / block_size));
    const uint32_t blocks_per_sm =
        deviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor) / block_size;
    grid.x = std::min(deviceAttribute(cudaDevAttrMultiProcessorCount) * blocks_per_sm,
                      grid.x);
    // Number of randoms generated per thread, as philox counter increments.
    const uint64_t counter_offset =
        ((numel - 1) / (block_size * grid.x * unroll_factor) + 1) *
        kMaxGeneratorOffsetsPerCall;
    return std::make_tuple(counter_offset, grid, dim_block);
}

template <typename scalar_t, typename dist_return_t, int unroll_factor,
          typename dist_t, typename transform_t>
__global__ void distribution_elementwise_grid_stride_kernel(
        int64_t numel, PhiloxCudaState philox_args, scalar_t* out_data,
        dist_t dist_func, transform_t transform_func) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size =
        ((numel - 1) / (total_threads * unroll_factor) + 1) * total_threads * unroll_factor;
    for (int64_t linear_index = idx; linear_index < rounded_size;
         linear_index += total_threads * unroll_factor) {
        auto rand = dist_func(&state);
        #pragma unroll
        for (int ii = 0; ii < unroll_factor; ii++) {
            int64_t li = linear_index + total_threads * ii;
            if (li < numel) {
                out_data[li] = transform_func((&rand.x)[ii]);
            }
        }
    }
}

template <typename scalar_t, typename dist_return_t, int unroll_factor,
          typename dist_t, typename transform_t>
void distribution_nullary_kernel(scalar_t* out_data, int64_t numel,
                                 dist_t dist_func, transform_t transform_func) {
    if (numel == 0) return;
    auto policy = calc_execution_policy(numel, unroll_factor);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    auto philox_args = philox_cuda_state(counter_offset);
    cudaStream_t stream = getCurrentCUDAStream().stream();
    distribution_elementwise_grid_stride_kernel<scalar_t, dist_return_t, unroll_factor>
        <<<grid, block, 0, stream>>>(numel, philox_args, out_data,
                                     dist_func, transform_func);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
    }
}

} // namespace

Tensor rand_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();

    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        // curand_uniform4 yields (0, 1]; torch maps it linearly to [from, to).
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [] __device__ (float rand) { return rand; });
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [] __device__ (double rand) { return rand; });
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Same consumption as float32 (accscalar float), cast down like torch.
        if (dtype == DType::Float16) {
            Half* data = t.data_ptr<Half>();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [] __device__ (float rand) { return static_cast<Half>(rand); });
        } else {
            BFloat16* data = t.data_ptr<BFloat16>();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [] __device__ (float rand) { return static_cast<BFloat16>(rand); });
        }
    } else if (dtype == DType::ComplexFloat || dtype == DType::ComplexDouble) {
        // torch parity: each component draws U[0,1) independently -- sample
        // the interleaved component buffer as a real array.
        const int64_t comps = t.numel() * 2;
        if (dtype == DType::ComplexFloat) {
            float* raw = static_cast<float*>(t.data_ptr());
            distribution_nullary_kernel<float, float4, 4>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [] __device__ (float v) { return v; });
        } else {
            double* raw = static_cast<double*>(t.data_ptr());
            distribution_nullary_kernel<double, double2, 2>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
                [] __device__ (double v) { return v; });
        }
    } else {
         TP_THROW(NotImplementedError, "rand() only supports floating dtypes on CUDA for now");
    }
    return t;
}

Tensor randn_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();

    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [] __device__ (float rand) { return rand; });
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [] __device__ (double rand) { return rand; });
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        if (dtype == DType::Float16) {
            Half* data = t.data_ptr<Half>();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float rand) { return static_cast<Half>(rand); });
        } else {
            BFloat16* data = t.data_ptr<BFloat16>();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float rand) { return static_cast<BFloat16>(rand); });
        }
    } else if (dtype == DType::ComplexFloat || dtype == DType::ComplexDouble) {
        // ATen normal_impl_ parity: view_as_real(self) with std/sqrt(2); the
        // standard-normal factory is N(0, 1/sqrt(2)) per component.
        constexpr float kInvSqrt2f = 0.70710678118654752f;
        constexpr double kInvSqrt2 = 0.70710678118654752440;
        const int64_t comps = t.numel() * 2;
        if (dtype == DType::ComplexFloat) {
            float* raw = static_cast<float*>(t.data_ptr());
            distribution_nullary_kernel<float, float4, 4>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float v) { return v * kInvSqrt2f; });
        } else {
            double* raw = static_cast<double*>(t.data_ptr());
            distribution_nullary_kernel<double, double2, 2>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
                [] __device__ (double v) { return v * kInvSqrt2; });
        }
    } else {
         TP_THROW(NotImplementedError, "randn() only supports floating dtypes on CUDA for now");
    }
    return t;
}

Tensor rand_like_kernel_cuda(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();

    Device target_device = device.has_value() ? *device : self.device();

    return rand_kernel_cuda(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
}

Tensor randn_like_kernel_cuda(const Tensor& self, DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device target_device = device.has_value() ? *device : self.device();
    return randn_kernel_cuda(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
}

Tensor& uniform_kernel_cuda(Tensor& self, double from, double to) {
    int64_t n = self.numel();
    if (n == 0) return self;
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float lo = static_cast<float>(from);
        const float hi = static_cast<float>(to);
        // curand_uniform4 yields (0, 1]; torch maps it linearly to [from, to).
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi] __device__ (float rand) { return lo + (hi - lo) * rand; });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [from, to] __device__ (double rand) { return from + (to - from) * rand; });
    } else {
        TP_THROW(NotImplementedError, "uniform_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

Tensor& normal_kernel_cuda(Tensor& self, double mean, double std) {
    int64_t n = self.numel();
    if (n == 0) return self;
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) { return mu + sigma * rand; });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [mean, std] __device__ (double rand) { return mean + std * rand; });
    } else {
        TP_THROW(NotImplementedError, "normal_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

// Port of at::native::templates::cuda::exponential_kernel
// (aten/src/ATen/native/cuda/DistributionTemplates.h:561) +
// transformation::exponential CUDA branch
// (aten/src/ATen/core/TransformationHelper.h:128): curand_uniform yields
// (0, 1]; log(1) is 0 and the exponential distribution excludes 0, so values
// within epsilon/2 of 1 clamp their log to -epsilon/2.
Tensor& exponential_kernel_cuda(Tensor& self, double lambd) {
    int64_t n = self.numel();
    if (n == 0) return self;
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float lambda = static_cast<float>(lambd);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lambda] __device__ (float val) {
                float log = val >= 1.f - kEps / 2 ? -kEps / 2 : __logf(val);
                return -1.f / lambda * log;
            });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        const double lambda = lambd;
        constexpr double kEps = std::numeric_limits<double>::epsilon();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [lambda] __device__ (double val) {
                double log = val >= 1. - kEps / 2 ? -kEps / 2 : ::log(val);
                return -1. / lambda * log;
            });
    } else if (self.dtype() == DType::Float16 || self.dtype() == DType::BFloat16) {
        // torch dispatches half/bfloat16 through accscalar_t = float.
        if (self.dtype() == DType::Float16) {
            Half* data = self.data_ptr<Half>();
            const float lambda = static_cast<float>(lambd);
            constexpr float kEps = std::numeric_limits<float>::epsilon();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [lambda] __device__ (float val) {
                    float log = val >= 1.f - kEps / 2 ? -kEps / 2 : __logf(val);
                    return static_cast<Half>(-1.f / lambda * log);
                });
        } else {
            BFloat16* data = self.data_ptr<BFloat16>();
            const float lambda = static_cast<float>(lambd);
            constexpr float kEps = std::numeric_limits<float>::epsilon();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
                [lambda] __device__ (float val) {
                    float log = val >= 1.f - kEps / 2 ? -kEps / 2 : __logf(val);
                    return static_cast<BFloat16>(-1.f / lambda * log);
                });
        }
    } else {
        TP_THROW(NotImplementedError, "exponential_() only supports floating dtypes on CUDA for now");
    }
    return self;
}

// --- torch-aligned distribution ports (DistributionKernels.cu + -----------
// --- TransformationHelper.h): random_ / randint / geometric_ / -------------
// --- log_normal_ / cauchy_ / poisson / randperm ----------------------------

Tensor& geometric_kernel_cuda(Tensor& self, double p) {
    int64_t n = self.numel();
    if (n == 0) return self;
    if (!(p > 0.0 && p < 1.0)) {
        TP_THROW(RuntimeError, "geometric_(): p must be in the interval (0, 1)");
    }
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float pf = static_cast<float>(p);
        // transformation::geometric: ceil(log(u)/log1p(-p)); curand_uniform
        // yields (0, 1] so log(val) is finite and <= 0.
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [pf] __device__ (float val) {
                return ::ceilf(::logf(val) / ::log1pf(-pf));
            });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [p] __device__ (double val) {
                return ::ceil(::log(val) / ::log1p(-p));
            });
    } else {
        TP_THROW(NotImplementedError, "geometric_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

Tensor& log_normal_kernel_cuda(Tensor& self, double mean, double std) {
    int64_t n = self.numel();
    if (n == 0) return self;
    if (std <= 0.0) {
        TP_THROW(RuntimeError, "log_normal_(): std must be positive");
    }
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        // transformation::log_normal: exp(normal_draw).
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) { return ::expf(mu + sigma * rand); });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [mean, std] __device__ (double rand) { return ::exp(mean + std * rand); });
    } else {
        TP_THROW(NotImplementedError, "log_normal_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

Tensor& cauchy_kernel_cuda(Tensor& self, double median, double sigma) {
    int64_t n = self.numel();
    if (n == 0) return self;
    if (sigma <= 0.0) {
        TP_THROW(RuntimeError, "cauchy_(): sigma must be positive");
    }
    constexpr double kPi = 3.14159265358979323846;
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float med = static_cast<float>(median);
        const float sig = static_cast<float>(sigma);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        // transformation::cauchy (float overload): clip val into
        // [eps, 1-eps] because tanf overflows at the open boundaries.
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [med, sig] __device__ (float val) {
                val = val > 1.f - kEps ? 1.f - kEps : val;
                val = val < kEps ? kEps : val;
                return med + sig * ::tanf(static_cast<float>(M_PI) * (val - 0.5f));
            });
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [median, sigma] __device__ (double val) {
                return median + sigma * ::tan(kPi * (val - 0.5));
            });
    } else {
        TP_THROW(NotImplementedError, "cauchy_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

// From-to integer draw kernels (torch random_from_to_kernel /
// random_from_to_64_kernel, DistributionTemplates.h:292-336).  Int64 pairs two
// adjacent philox words per draw like torch's random64.

__global__ void randint32_fill_impl(int64_t numel, PhiloxCudaState philox_args,
                                    uint64_t range, int64_t base,
                                    unsigned int* out_data) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size =
        ((numel - 1) / (total_threads * 4) + 1) * total_threads * 4;
    for (int64_t linear_index = idx; linear_index < rounded_size;
         linear_index += total_threads * 4) {
        uint4 rand = curand4(&state);
        #pragma unroll
        for (int ii = 0; ii < 4; ii++) {
            int64_t li = linear_index + total_threads * ii;
            if (li < numel) {
                out_data[li] = static_cast<unsigned int>(
                    ((uint64_t)(&rand.x)[ii]) % range + (uint64_t)base);
            }
        }
    }
}

__global__ void randint64_fill_impl(int64_t numel, PhiloxCudaState philox_args,
                                    uint64_t range, int64_t base,
                                    int64_t* out_data) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const int64_t total_threads = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const int64_t rounded_size =
        ((numel - 1) / (total_threads * 2) + 1) * total_threads * 2;
    for (int64_t linear_index = idx; linear_index < rounded_size;
         linear_index += total_threads * 2) {
        uint4 rand = curand4(&state);
        #pragma unroll
        for (int ii = 0; ii < 2; ii++) {
            int64_t li = linear_index + total_threads * ii;
            if (li < numel) {
                const unsigned int* words = (&rand.x) + 2 * ii;
                uint64_t val = ((uint64_t)words[0] << 32) | words[1];
                out_data[li] = static_cast<int64_t>(val % range + (uint64_t)base);
            }
        }
    }
}

static void randint_fill_dispatch(Tensor& t, int64_t low, int64_t high) {
    int64_t n = t.numel();
    if (n == 0) return;
    const uint64_t range = static_cast<uint64_t>(high - low);
    const int64_t base = low;

    auto launch = [&](auto kernel, auto* ptr, uint32_t unroll_factor) {
        auto policy = calc_execution_policy(n, unroll_factor);
        const uint64_t counter_offset = std::get<0>(policy);
        const dim3 grid = std::get<1>(policy);
        const dim3 block = std::get<2>(policy);
        auto philox_args = philox_cuda_state(counter_offset);
        kernel<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, range, base, ptr);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            TP_THROW(RuntimeError, std::string("CUDA randint Error: ") +
                        cudaGetErrorString(error));
        }
    };

    switch (t.dtype()) {
        case DType::Int64:
            launch(randint64_fill_impl, t.data_ptr<int64_t>(), 2u);
            break;
        case DType::Int32:
            launch(randint32_fill_impl,
                   reinterpret_cast<unsigned int*>(t.data_ptr<int32_t>()), 4u);
            break;
        default:
            TP_THROW(NotImplementedError, "randint: only Int64/Int32 supported on CUDA");
    }
}

Tensor randint_kernel_cuda(int64_t low, int64_t high, const std::vector<int64_t>& size,
                           DType dtype, Device device) {
    if (low >= high) {
        TP_THROW(RuntimeError, "randint(): low must be less than high");
    }
    Tensor t(size, dtype, device);
    randint_fill_dispatch(t, low, high);
    return t;
}

Tensor randint_like_kernel_cuda(const Tensor& self, int64_t low, int64_t high,
                                DType dtype, std::optional<Device> device) {
    if (dtype == DType::Undefined) dtype = self.dtype();
    Device target_device = device.has_value() ? *device : self.device();
    Tensor t = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), dtype, target_device);
    randint_fill_dispatch(t, low, high);
    return t;
}

Tensor& random_kernel_cuda(Tensor& self, int64_t low, int64_t high) {
    int64_t n = self.numel();
    if (n == 0) return self;
    const bool full_range = (low == 0 && high == 0);
    if (!full_range && low >= high) {
        TP_THROW(RuntimeError, "random_(): upper bound must be larger than lower bound");
    }

    if (full_range && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        // torch random_() on floats fills [0, 1).
        return uniform_kernel_cuda(self, 0.0, 1.0);
    }
    if (full_range) {
        // Full-range ints: uniform_int_full_range casts raw draws; approximate
        // with the widest from-to interval of the dtype.
        if (self.dtype() == DType::Int64) {
            low = 0; high = std::numeric_limits<int64_t>::max();
        } else if (self.dtype() == DType::Int32) {
            low = 0; high = std::numeric_limits<int32_t>::max();
        } else {
            TP_THROW(NotImplementedError,
                     "random_() only supports Int64/Int32/Float32/Float64 on CUDA for now");
        }
    }
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const uint64_t range = static_cast<uint64_t>(high - low);
        const int64_t base = low;
        // Integral-valued floats in [low, high), matching CPU semantics.
        distribution_nullary_kernel<float, uint4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand4(state); },
            [range, base] __device__ (unsigned int val) {
                return static_cast<float>(static_cast<int64_t>(val % range) + base);
            });
    } else {
        randint_fill_dispatch(self, low, high);
    }
    return self;
}

// poisson (torch poisson_kernel): one thread per element since each element
// carries its own lambda and consumes a variable number of philox counters.
template <typename scalar_t>
__global__ void poisson_fill_impl(int64_t numel, PhiloxCudaState philox_args,
                                  const scalar_t* in_data, scalar_t* out_data) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    out_data[idx] = static_cast<scalar_t>(
        curand_poisson(&state, static_cast<double>(in_data[idx])));
}

Tensor poisson_kernel_cuda(const Tensor& self) {
    Tensor t(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return t;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    // Each thread runs curand_poisson which consumes lambda-dependent counters;
    // reserve generously so concurrent calls never share counter slices.
    const uint64_t counter_offset = 16u *
        ((static_cast<uint64_t>(n) + threads * blocks - 1) / (threads * blocks) + 1) *
        kMaxGeneratorOffsetsPerCall;
    auto philox_args = philox_cuda_state(counter_offset);

    if (self.dtype() == DType::Float32) {
        poisson_fill_impl<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, self.data_ptr<float>(), t.data_ptr<float>());
    } else if (self.dtype() == DType::Float64) {
        poisson_fill_impl<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, self.data_ptr<double>(), t.data_ptr<double>());
    } else {
        TP_THROW(NotImplementedError, "poisson() only supports Float32/Float64 on CUDA for now");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA poisson Error: ") +
                    cudaGetErrorString(error));
    }
    return t;
}

// bernoulli_ (torch bernoulli_tensor_cuda_): p taken elementwise from self;
// one thread per element keeps the threshold read race-free.
template <typename scalar_t>
__global__ void bernoulli_fill_impl(int64_t numel, PhiloxCudaState philox_args,
                                    const scalar_t* in_data, scalar_t* out_data) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, offset, &state);
    const float u = curand_uniform(&state);
    out_data[idx] = static_cast<scalar_t>(
        u < static_cast<float>(in_data[idx]) ? scalar_t(1) : scalar_t(0));
}

Tensor& bernoulli_inplace_kernel_cuda(Tensor& self) {
    int64_t n = self.numel();
    if (n == 0) return self;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    auto philox_args = philox_cuda_state(kMaxGeneratorOffsetsPerCall *
        ((static_cast<uint64_t>(n) + threads * blocks - 1) / (threads * blocks) + 1));

    if (self.dtype() == DType::Float32) {
        bernoulli_fill_impl<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, self.data_ptr<float>(), self.data_ptr<float>());
    } else if (self.dtype() == DType::Float64) {
        bernoulli_fill_impl<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, self.data_ptr<double>(), self.data_ptr<double>());
    } else {
        TP_THROW(NotImplementedError, "bernoulli_() only supports Float32/Float64 on CUDA for now");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA bernoulli_ Error: ") +
                    cudaGetErrorString(error));
    }
    return self;
}

// randperm: fp32 keys sorted by argsort (torch uses random keys + sort too,
// aten/src/ATen/native/cuda/TensorFactories.cu randperm_handle_duplicate_keys).
Tensor randperm_kernel_cuda(int64_t n, DType dtype, Device device) {
    if (dtype != DType::Int64 && dtype != DType::Int32) {
        TP_THROW(NotImplementedError, "randperm() only supports Int64/Int32 on CUDA");
    }
    Tensor idx(std::vector<int64_t>{n}, DType::Int64, device);
    if (n == 0) {
        if (dtype == DType::Int32) {
            Tensor out = Tensor::empty({n}, DType::Int32, device);
            return out;
        }
        return idx;
    }
    // Distinct-with-probability-1 fp32 keys; ties are astronomically unlikely.
    Tensor keys = Tensor::empty({n}, DType::Float32, device);
    float* data = keys.data_ptr<float>();
    distribution_nullary_kernel<float, float4, 4>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        [] __device__ (float val) { return val; });
    extern Tensor argsort_cuda(const Tensor& self, int64_t dim, bool descending);
    idx = argsort_cuda(keys, 0, false);
    if (dtype == DType::Int64) return idx;
    Tensor out = Tensor::empty({n}, DType::Int32, device);
    out.copy_(idx);
    return out;
}

// --- Stub-ABI adapters (schema-level optional<DType>/optional<Device>) ------
namespace {

Tensor rand_stub_cuda(const std::vector<int64_t>& size, std::optional<DType> dtype,
                      std::optional<Device> device) {
    return rand_kernel_cuda(size, dtype.value_or(DType::Float32),
                            device.value_or(Device(DeviceType::CUDA)));
}

Tensor randn_stub_cuda(const std::vector<int64_t>& size, std::optional<DType> dtype,
                       std::optional<Device> device) {
    return randn_kernel_cuda(size, dtype.value_or(DType::Float32),
                             device.value_or(Device(DeviceType::CUDA)));
}

Tensor randint_stub_cuda(int64_t low, int64_t high, const std::vector<int64_t>& size,
                         DType dtype, std::optional<Device> device) {
    return randint_kernel_cuda(low, high, size, dtype,
                               device.value_or(Device(DeviceType::CUDA)));
}

Tensor randperm_stub_cuda(int64_t n, DType dtype, std::optional<Device> device) {
    return randperm_kernel_cuda(n, dtype, device.value_or(Device(DeviceType::CUDA)));
}

// bernoulli(p) -- ATen DistributionTemplates.h bernoulli_impl: result is a
// fresh tensor of the same shape/dtype, filled by the same philox kernel that
// backs bernoulli_, with p read from `self` (separate in/out pointers).
Tensor bernoulli_out_kernel_cuda(const Tensor& self) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
    int64_t n = self.numel();
    if (n == 0) return out;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    auto philox_args = philox_cuda_state(kMaxGeneratorOffsetsPerCall *
        ((static_cast<uint64_t>(n) + threads * blocks - 1) / (threads * blocks) + 1));

    if (self.dtype() == DType::Float32) {
        bernoulli_fill_impl<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, self.data_ptr<float>(), out.data_ptr<float>());
    } else if (self.dtype() == DType::Float64) {
        bernoulli_fill_impl<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            n, philox_args, self.data_ptr<double>(), out.data_ptr<double>());
    } else {
        TP_THROW(NotImplementedError, "bernoulli() only supports Float32/Float64 on CUDA for now");
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA bernoulli Error: ") +
                    cudaGetErrorString(error));
    }
    return out;
}

// normal(mean, std) -- ATen DistributionTemplates.h normal_out_impl
// (tensor-tensor): ret = empty(infer_size(mean.sizes(), std.sizes()));
// ret.normal_(0, 1); ret.mul_(std).add_(mean);
Tensor normal_broadcast_kernel_cuda(const Tensor& mean, const Tensor& std) {
    std::vector<int64_t> shape = broadcast_shapes(
        static_cast<std::vector<int64_t>>(mean.shape()),
        static_cast<std::vector<int64_t>>(std.shape()));
    Tensor out(shape, mean.dtype(), mean.device());
    if (out.numel() == 0) return out;
    out.normal_(0.0, 1.0);
    out.mul_(std).add_(mean);
    return out;
}

} // anonymous namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, RandomKernels) {
    m.impl("rand", rand_stub_cuda);
    m.impl("randn", randn_stub_cuda);
    m.impl("rand_like", rand_like_kernel_cuda);
    m.impl("randn_like", randn_like_kernel_cuda);
    m.impl("uniform_", uniform_kernel_cuda);
    m.impl("normal_", normal_kernel_cuda);
    m.impl("exponential_", exponential_kernel_cuda);
    m.impl("geometric_", geometric_kernel_cuda);
    m.impl("log_normal_", log_normal_kernel_cuda);
    m.impl("cauchy_", cauchy_kernel_cuda);
    m.impl("random_", random_kernel_cuda);
    m.impl("bernoulli_", bernoulli_inplace_kernel_cuda);
    m.impl("bernoulli", bernoulli_out_kernel_cuda);
    m.impl("normal", normal_broadcast_kernel_cuda);
    m.impl("poisson", poisson_kernel_cuda);
    m.impl("randint", randint_stub_cuda);
    m.impl("randint_like", randint_like_kernel_cuda);
    m.impl("randperm", randperm_stub_cuda);
}

} // namespace cuda
} // namespace tensorplay
