#include "Tensor.h"
#include "Dispatcher.h"
#include "CUDAGenerator.h"
#include "Exception.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
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
// are independent of launch geometry and reproducible across runs.

namespace {

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
        int64_t numel, uint64_t seed, uint64_t offset, scalar_t* out_data,
        dist_t dist_func, transform_t transform_func) {
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
    auto philox_args = philox_engine_inputs(counter_offset);
    distribution_elementwise_grid_stride_kernel<scalar_t, dist_return_t, unroll_factor>
        <<<grid, block>>>(numel, philox_args.first, philox_args.second, out_data,
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

TENSORPLAY_LIBRARY_IMPL(CUDA, RandomKernels) {
    m.impl("rand", rand_kernel_cuda);
    m.impl("randn", randn_kernel_cuda);
    m.impl("rand_like", rand_like_kernel_cuda);
    m.impl("randn_like", randn_like_kernel_cuda);
    m.impl("uniform_", uniform_kernel_cuda);
    m.impl("normal_", normal_kernel_cuda);
    m.impl("exponential_", exponential_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
