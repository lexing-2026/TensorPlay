#include "Tensor.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "CUDAGenerator.h"
#include "Generator.h"
#include "Exception.h"
#include "Utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tensorplay {
namespace cuda {

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

// Utility function that calculates the proper philox_offset for the
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
                                 dist_t dist_func, transform_t transform_func,
                                 std::optional<Generator> generator = std::nullopt) {
    if (numel == 0) return;
    auto policy = calc_execution_policy(numel, unroll_factor);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    PhiloxCudaState philox_args;
    if (generator.has_value()) {
        philox_args.seed = generator->random64();
        philox_args.offset = 0;
    } else {
        philox_args = philox_cuda_state(counter_offset);
    }
    cudaStream_t stream = getCurrentCUDAStream().stream();
    distribution_elementwise_grid_stride_kernel<scalar_t, dist_return_t, unroll_factor>
        <<<grid, block, 0, stream>>>(numel, philox_args, out_data,
                                     dist_func, transform_func);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error));
    }
}


// In-place sampling writes through a contiguous buffer: non-contiguous
// destinations are filled via a contiguous temporary copied back with the
// strided copy kernel.  A stride-0 dimension with more than one element
// aliases the whole dimension, makes the draw order observable, and is
// rejected.
template <typename FillFn>
Tensor& fill_via_contiguous(Tensor& self, FillFn&& fill) {
    const auto sizes = static_cast<std::vector<int64_t>>(self.shape());
    const auto strides = self.strides();
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (strides[i] == 0 && sizes[i] > 1) {
            TP_THROW(RuntimeError,
                     "unsupported operation: more than one element of the written-to tensor "
                     "refers to a single memory location. Please clone() the tensor before "
                     "performing the operation.");
        }
    }
    Tensor tmp = Tensor::empty(sizes, self.dtype(), self.device());
    fill(tmp);
    self.copy_(tmp);
    return self;
}

} // namespace

Tensor rand_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();

    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
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

Tensor randn_kernel_cuda(const std::vector<int64_t>& size, DType dtype, Device device,
                         std::optional<Generator> generator = std::nullopt) {
    Tensor t = Tensor::empty(size, dtype, device);
    int64_t n = t.numel();

    if (dtype == DType::Float32) {
        float* data = t.data_ptr<float>();
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [] __device__ (float rand) { return rand; }, std::move(generator));
    } else if (dtype == DType::Float64) {
        double* data = t.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [] __device__ (double rand) { return rand; }, std::move(generator));
    } else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        if (dtype == DType::Float16) {
            Half* data = t.data_ptr<Half>();
            distribution_nullary_kernel<Half, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float rand) { return static_cast<Half>(rand); },
                std::move(generator));
        } else {
            BFloat16* data = t.data_ptr<BFloat16>();
            distribution_nullary_kernel<BFloat16, float4, 4>(
                data, n,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float rand) { return static_cast<BFloat16>(rand); },
                std::move(generator));
        }
    } else if (dtype == DType::ComplexFloat || dtype == DType::ComplexDouble) {
        // standard-normal factory is N(0, 1/sqrt(2)) per component.
        constexpr float kInvSqrt2f = 0.70710678118654752f;
        constexpr double kInvSqrt2 = 0.70710678118654752440;
        const int64_t comps = t.numel() * 2;
        if (dtype == DType::ComplexFloat) {
            float* raw = static_cast<float*>(t.data_ptr());
            distribution_nullary_kernel<float, float4, 4>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
                [] __device__ (float v) { return v * kInvSqrt2f; },
                std::move(generator));
        } else {
            double* raw = static_cast<double*>(t.data_ptr());
            distribution_nullary_kernel<double, double2, 2>(
                raw, comps,
                [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
                [] __device__ (double v) { return v * kInvSqrt2; },
                std::move(generator));
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

Tensor& uniform_kernel_cuda(Tensor& self, double from, double to,
                            std::optional<Generator> generator) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return uniform_kernel_cuda(t, from, to, generator);
        });
    }
    int64_t n = self.numel();
    // Bounds of the [from, to) range against the destination dtype.
    if (self.dtype() == DType::Float32) {
        TP_THROW_IF(!(from >= -std::numeric_limits<float>::max() && from <= std::numeric_limits<float>::max()),
                    RuntimeError, "from is out of bounds for float");
        TP_THROW_IF(!(to >= -std::numeric_limits<float>::max() && to <= std::numeric_limits<float>::max()),
                    RuntimeError, "to is out of bounds for float");
    } else if (self.dtype() == DType::Float64) {
        TP_THROW_IF(!(from >= -std::numeric_limits<double>::max() && from <= std::numeric_limits<double>::max()),
                    RuntimeError, "from is out of bounds for double");
        TP_THROW_IF(!(to >= -std::numeric_limits<double>::max() && to <= std::numeric_limits<double>::max()),
                    RuntimeError, "to is out of bounds for double");
    }
    TP_THROW_IF(from > to, RuntimeError,
                "uniform_ expects to return a [from, to) range, but found from=",
                from, " > to=", to);
    TP_THROW_IF((to - from) > (self.dtype() == DType::Float32
                                   ? static_cast<double>(std::numeric_limits<float>::max())
                                   : std::numeric_limits<double>::max()),
                RuntimeError,
                "uniform_ expects to-from <= std::numeric_limits<",
                self.dtype() == DType::Float32 ? "float" : "double",
                ">::max(), but found to=", to, " and from=", from,
                " which result in to-from to exceed the limit");
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float lo = static_cast<float>(from);
        const float hi = static_cast<float>(to);
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi] __device__ (float rand) { return lo + (hi - lo) * rand; },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [from, to] __device__ (double rand) { return from + (to - from) * rand; },
            std::move(generator));
    } else {
        TP_THROW(NotImplementedError, "uniform_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

Tensor& normal_kernel_cuda(Tensor& self, double mean, double std,
                           std::optional<Generator> generator) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return normal_kernel_cuda(t, mean, std, std::move(generator));
        });
    }
    int64_t n = self.numel();
    if (std < 0.0) {
        TP_THROW(RuntimeError, "normal expects std >= 0.0, but found std ", std);
    }
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) { return mu + sigma * rand; },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [mean, std] __device__ (double rand) { return mean + std * rand; },
            std::move(generator));
    } else {
        TP_THROW(NotImplementedError, "normal_() only supports Float32/Float64 on CUDA for now");
    }
    return self;
}

// transformation::exponential CUDA branch
// (0, 1]; log(1) is 0 and the exponential distribution excludes 0, so values
// within epsilon/2 of 1 clamp their log to -epsilon/2.
Tensor& exponential_kernel_cuda(Tensor& self, double lambd) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return exponential_kernel_cuda(t, lambd);
        });
    }
    int64_t n = self.numel();
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

// --- TransformationHelper.h): random_ / randint / geometric_ / -------------
// --- log_normal_ / cauchy_ / poisson / randperm ----------------------------

Tensor& geometric_kernel_cuda(Tensor& self, double p) {
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return geometric_kernel_cuda(t, p); });
    }
    int64_t n = self.numel();
    if (!(p > 0.0 && p < 1.0)) {
        TP_THROW(RuntimeError, "geometric_ expects p to be in (0, 1), but got p=", p);
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
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return log_normal_kernel_cuda(t, mean, std); });
    }
    int64_t n = self.numel();
    if (std <= 0.0) {
        TP_THROW(RuntimeError, "log_normal_ expects std > 0.0, but found std=", std);
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
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return cauchy_kernel_cuda(t, median, sigma); });
    }
    int64_t n = self.numel();
    if (sigma <= 0.0) {
        TP_THROW(RuntimeError, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
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

// random_from_to_64_kernel, DistributionTemplates.h:292-336).  Int64 pairs two

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
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) { return random_kernel_cuda(t, low, high); });
    }
    int64_t n = self.numel();
    const bool full_range = (low == 0 && high == 0);
    if (!full_range && low >= high) {
        TP_THROW(RuntimeError, "random_ expects 'from' to be less than 'to', but got from=", low, " >= to=", high);
    }

    if (full_range && (self.dtype() == DType::Float32 || self.dtype() == DType::Float64)) {
        return uniform_kernel_cuda(self, 0.0, 1.0, std::nullopt);
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

template <typename output_t, typename probability_t>
__global__ void bernoulli_tensor_fill_impl(
        int64_t numel, PhiloxCudaState philox_args,
        const probability_t* probability, output_t* output) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    const int64_t thread_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t linear_index = thread_index;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_index, offset, &state);
    for (; linear_index < numel; linear_index += thread_count * 4) {
        const float4 random = curand_uniform4(&state);
        const int64_t i0 = linear_index;
        const int64_t i1 = linear_index + thread_count;
        const int64_t i2 = linear_index + thread_count * 2;
        const int64_t i3 = linear_index + thread_count * 3;
        if (i0 < numel) {
            const double p = static_cast<double>(probability[i0]);
            assert(0.0 <= p && p <= 1.0);
            output[i0] = static_cast<output_t>(random.x <= p);
        }
        if (i1 < numel) {
            const double p = static_cast<double>(probability[i1]);
            assert(0.0 <= p && p <= 1.0);
            output[i1] = static_cast<output_t>(random.y <= p);
        }
        if (i2 < numel) {
            const double p = static_cast<double>(probability[i2]);
            assert(0.0 <= p && p <= 1.0);
            output[i2] = static_cast<output_t>(random.z <= p);
        }
        if (i3 < numel) {
            const double p = static_cast<double>(probability[i3]);
            assert(0.0 <= p && p <= 1.0);
            output[i3] = static_cast<output_t>(random.w <= p);
        }
    }
}

template <typename output_t>
__global__ void bernoulli_scalar_fill_impl(
        int64_t numel, PhiloxCudaState philox_args, double p,
        output_t* output) {
    uint64_t seed;
    uint64_t offset;
    philox_unpack(philox_args, &seed, &offset);
    const int64_t thread_index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t thread_count =
        static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t linear_index = thread_index;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, thread_index, offset, &state);
    for (; linear_index < numel; linear_index += thread_count * 4) {
        const float4 random = curand_uniform4(&state);
        const int64_t i0 = linear_index;
        const int64_t i1 = linear_index + thread_count;
        const int64_t i2 = linear_index + thread_count * 2;
        const int64_t i3 = linear_index + thread_count * 3;
        if (i0 < numel) output[i0] = static_cast<output_t>(random.x <= p);
        if (i1 < numel) output[i1] = static_cast<output_t>(random.y <= p);
        if (i2 < numel) output[i2] = static_cast<output_t>(random.z <= p);
        if (i3 < numel) output[i3] = static_cast<output_t>(random.w <= p);
    }
}

PhiloxCudaState bernoulli_philox_state(
        std::optional<Generator> generator, uint64_t increment) {
    if (generator.has_value()) {
        PhiloxCudaState state;
        state.seed = generator->random64();
        state.offset = 0;
        return state;
    }
    return philox_cuda_state(increment);
}

template <typename output_t, typename probability_t>
void launch_bernoulli_tensor(const Tensor& probability, Tensor& output,
                             std::optional<Generator> generator) {
    const auto policy = calc_execution_policy(output.numel(), 4);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    const PhiloxCudaState philox_args =
        bernoulli_philox_state(std::move(generator), counter_offset);
    bernoulli_tensor_fill_impl<output_t, probability_t>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output.numel(), philox_args, probability.data_ptr<probability_t>(),
            output.data_ptr<output_t>());
}

template <typename probability_t>
void launch_bernoulli_tensor_for_output(
        const Tensor& probability, Tensor& output,
        std::optional<Generator> generator) {
#define TP_BERNOULLI_TENSOR_OUTPUT_CASE(ctype, name) \
    case DType::name: \
        launch_bernoulli_tensor<ctype, probability_t>( \
            probability, output, std::move(generator)); \
        break;
    switch (output.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BERNOULLI_TENSOR_OUTPUT_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "bernoulli_ output dtype is not supported");
    }
#undef TP_BERNOULLI_TENSOR_OUTPUT_CASE
}

Tensor prepare_bernoulli_probabilities(const Tensor& output,
                                       const Tensor& probabilities) {
    if (!isFloatingType(probabilities.dtype())) {
        TP_THROW(TypeError,
                 "bernoulli_ probability tensor must have a floating dtype");
    }
    const DType probability_dtype =
        output.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
    Tensor probability = probabilities.dtype() == probability_dtype
        ? probabilities
        : probabilities.to(probability_dtype);
    if (probability.shape() != output.shape()) {
        probability = probability.expand(
            static_cast<std::vector<int64_t>>(output.shape()));
    }
    return probability.is_contiguous() ? probability : probability.contiguous();
}

Tensor& bernoulli_tensor_inplace_kernel_cuda(
        Tensor& self, const Tensor& probabilities,
        std::optional<Generator> generator) {
    if (self.device() != probabilities.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli_: probability tensor must be on the same device");
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return bernoulli_tensor_inplace_kernel_cuda(
                t, probabilities, std::move(generator));
        });
    }

    Tensor probability = prepare_bernoulli_probabilities(self, probabilities);
    if (self.dtype() == DType::Float64) {
        launch_bernoulli_tensor_for_output<double>(
            probability, self, std::move(generator));
    } else {
        launch_bernoulli_tensor_for_output<float>(
            probability, self, std::move(generator));
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA bernoulli_ Error: ") +
                    cudaGetErrorString(error));
    }
    return self;
}

template <typename output_t>
void launch_bernoulli_scalar(Tensor& output, double p,
                             std::optional<Generator> generator) {
    const auto policy = calc_execution_policy(output.numel(), 4);
    const uint64_t counter_offset = std::get<0>(policy);
    const dim3 grid = std::get<1>(policy);
    const dim3 block = std::get<2>(policy);
    const PhiloxCudaState philox_args =
        bernoulli_philox_state(std::move(generator), counter_offset);
    bernoulli_scalar_fill_impl<output_t>
        <<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            output.numel(), philox_args, p, output.data_ptr<output_t>());
}

Tensor& bernoulli_scalar_inplace_kernel_cuda(
        Tensor& self, double p, std::optional<Generator> generator) {
    if (!(p >= 0.0 && p <= 1.0)) {
        TP_THROW(ValueError, "bernoulli_ expects p to be in [0, 1]");
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return bernoulli_scalar_inplace_kernel_cuda(
                t, p, std::move(generator));
        });
    }
#define TP_BERNOULLI_SCALAR_OUTPUT_CASE(ctype, name) \
    case DType::name: \
        launch_bernoulli_scalar<ctype>(self, p, std::move(generator)); \
        break;
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_BERNOULLI_SCALAR_OUTPUT_CASE)
        default:
            TP_THROW(NotImplementedError,
                     "bernoulli_ output dtype is not supported");
    }
#undef TP_BERNOULLI_SCALAR_OUTPUT_CASE

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError, std::string("CUDA bernoulli_ Error: ") +
                    cudaGetErrorString(error));
    }
    return self;
}

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

Tensor randn_generator_stub_cuda(const std::vector<int64_t>& size,
                                 std::optional<Generator> generator,
                                 std::optional<DType> dtype,
                                 std::optional<int64_t> layout,
                                 std::optional<Device> device,
                                 std::optional<bool> pin_memory) {
    if (layout.has_value() && *layout != 2) {
        TP_THROW(NotImplementedError,
                 "randn is only implemented for strided (dense) layout tensors");
    }
    if (pin_memory.value_or(false)) {
        TP_THROW(RuntimeError,
                 "pin_memory is not valid for CUDA random tensor outputs");
    }
    return randn_kernel_cuda(size, dtype.value_or(DType::Float32),
                             device.value_or(Device(DeviceType::CUDA)),
                             std::move(generator));
}

Tensor randint_stub_cuda(int64_t low, int64_t high, const std::vector<int64_t>& size,
                         DType dtype, std::optional<Device> device) {
    return randint_kernel_cuda(low, high, size, dtype,
                               device.value_or(Device(DeviceType::CUDA)));
}

Tensor randperm_stub_cuda(int64_t n, DType dtype, std::optional<Device> device) {
    return randperm_kernel_cuda(n, dtype, device.value_or(Device(DeviceType::CUDA)));
}

Tensor bernoulli_kernel_cuda(const Tensor& self,
                             std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()),
               self.dtype(), self.device());
    bernoulli_tensor_inplace_kernel_cuda(out, self, std::move(generator));
    return out;
}

Tensor& bernoulli_out_kernel_cuda(const Tensor& self,
                                  std::optional<Generator> generator,
                                  Tensor& out) {
    if (self.device() != out.device()) {
        TP_THROW(DeviceMismatchError,
                 "bernoulli: output must be on the same device as input");
    }
    out.resize_(static_cast<std::vector<int64_t>>(self.shape()));
    return bernoulli_tensor_inplace_kernel_cuda(out, self,
                                                std::move(generator));
}

Tensor bernoulli_p_kernel_cuda(const Tensor& self, double p,
                               std::optional<Generator> generator) {
    Tensor out(static_cast<std::vector<int64_t>>(self.shape()),
               self.dtype(), self.device());
    bernoulli_scalar_inplace_kernel_cuda(out, p, std::move(generator));
    return out;
}

} // anonymous namespace

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

/*
 * The registration table is kept at the end of the translation unit so every
 * overload has a complete schema-level function type before registration.
 */

TENSORPLAY_LIBRARY_IMPL(CUDA, RandomKernels) {
    m.impl("rand", rand_stub_cuda);
    m.impl("randn", randn_stub_cuda);
    m.impl("randn.generator", randn_generator_stub_cuda);
    m.impl("rand_like", rand_like_kernel_cuda);
    m.impl("randn_like", randn_like_kernel_cuda);
    m.impl("uniform_", uniform_kernel_cuda);
    m.impl("normal_", normal_kernel_cuda);
    m.impl("exponential_", exponential_kernel_cuda);
    m.impl("geometric_", geometric_kernel_cuda);
    m.impl("log_normal_", log_normal_kernel_cuda);
    m.impl("cauchy_", cauchy_kernel_cuda);
    m.impl("random_", random_kernel_cuda);
    m.impl("bernoulli", bernoulli_kernel_cuda);
    m.impl("bernoulli.out", bernoulli_out_kernel_cuda);
    m.impl("bernoulli.p", bernoulli_p_kernel_cuda);
    m.impl("bernoulli_.Tensor", bernoulli_tensor_inplace_kernel_cuda);
    m.impl("bernoulli_.float", bernoulli_scalar_inplace_kernel_cuda);
    m.impl("normal", normal_broadcast_kernel_cuda);
    m.impl("poisson", poisson_kernel_cuda);
    m.impl("randint", randint_stub_cuda);
    m.impl("randint_like", randint_like_kernel_cuda);
    m.impl("randperm", randperm_stub_cuda);
}

} // namespace cuda
} // namespace tensorplay
