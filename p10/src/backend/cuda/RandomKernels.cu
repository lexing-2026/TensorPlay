#include "Tensor.h"
#include "CUDARuntime.h"
#include "Dispatcher.h"
#include "CUDAGenerator.h"
#include "Generator.h"
#include "Exception.h"
#include "RandomCommon.cuh"
#include "Utils.h"
#include "DistributionDispatch.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
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

constexpr uint32_t kBlockSize = 256;

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
    if (isComplexType(self.dtype())) {
        Tensor real_view = tpx::ops::view_as_real(self);
        uniform_kernel_cuda(real_view, from, to, std::move(generator));
        return self;
    }
    if (self.dtype() != DType::Float16 && self.dtype() != DType::BFloat16 &&
        self.dtype() != DType::Float32 && self.dtype() != DType::Float64) {
        TP_THROW(NotImplementedError, "uniform_() only supports floating dtypes on CUDA");
    }
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return uniform_kernel_cuda(t, from, to, std::move(generator));
        });
    }
    const double dtype_max = self.dtype() == DType::Float16
        ? distribution::fp_dtype_max<Half>()
        : self.dtype() == DType::BFloat16
            ? distribution::fp_dtype_max<BFloat16>()
            : self.dtype() == DType::Float32
                ? distribution::fp_dtype_max<float>()
                : distribution::fp_dtype_max<double>();
    const char* dtype_name = distribution::bounds_dtype_name(self.dtype());
    TP_THROW_IF(!(from >= -dtype_max && from <= dtype_max), RuntimeError,
                "from is out of bounds for ", dtype_name);
    TP_THROW_IF(!(to >= -dtype_max && to <= dtype_max), RuntimeError,
                "to is out of bounds for ", dtype_name);
    TP_THROW_IF(from > to, RuntimeError,
                "uniform_ expects to return a [from, to) range, but found from=",
                from, " > to=", to);
    TP_THROW_IF((to - from) > dtype_max,
                RuntimeError,
                "uniform_ expects to-from <= std::numeric_limits<",
                dtype_name,
                ">::max(), but found to=", to, " and from=", from,
                " which result in to-from to exceed the limit");
    from = std::clamp(from, -dtype_max, dtype_max);
    to = std::clamp(to, -dtype_max, dtype_max);

    if (self.numel() == 0) return self;
    int64_t n = self.numel();
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float lo = static_cast<float>(from);
        const float hi = static_cast<float>(to);
        const float range = hi - lo;
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi, range] __device__ (float rand) {
                const float value = static_cast<float>(lo + range * rand);
                return value == hi ? lo : value;
            },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        const double range = to - from;
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [from, to, range] __device__ (double rand) {
                const double value = from + range * rand;
                return value == to ? from : value;
            },
            std::move(generator));
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const Half lo = static_cast<Half>(from);
        const Half hi = static_cast<Half>(to);
        const float lo_value = static_cast<float>(lo);
        const float hi_value = static_cast<float>(hi);
        const float range = hi_value - lo_value;
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi, lo_value, range] __device__ (float rand) {
                const Half value = static_cast<Half>(lo_value + range * rand);
                return value == hi ? lo : value;
            },
            std::move(generator));
    } else {
        BFloat16* data = self.data_ptr<BFloat16>();
        const BFloat16 lo = static_cast<BFloat16>(from);
        const BFloat16 hi = static_cast<BFloat16>(to);
        const float lo_value = static_cast<float>(lo);
        const float hi_value = static_cast<float>(hi);
        const float range = hi_value - lo_value;
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [lo, hi, lo_value, range] __device__ (float rand) {
                const BFloat16 value = static_cast<BFloat16>(lo_value + range * rand);
                return value == hi ? lo : value;
            },
            std::move(generator));
    }
    return self;
}

Tensor& normal_kernel_cuda(Tensor& self, double mean, double std,
                           std::optional<Generator> generator) {
    if (!(std >= 0.0)) {
        TP_THROW(RuntimeError, "normal expects std >= 0.0, but found std ", std);
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return normal_kernel_cuda(t, mean, std, std::move(generator));
        });
    }
    int64_t n = self.numel();
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
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<Half>(mu + sigma * rand);
            },
            std::move(generator));
    } else if (self.dtype() == DType::BFloat16) {
        BFloat16* data = self.data_ptr<BFloat16>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<BFloat16>(mu + sigma * rand);
            },
            std::move(generator));
    } else {
        TP_THROW(NotImplementedError,
                 "normal_() only supports floating dtypes on CUDA");
    }
    return self;
}

// transformation::exponential CUDA branch
// (0, 1]; log(1) is 0 and the exponential distribution excludes 0, so values
// within epsilon/2 of 1 clamp their log to -epsilon/2.
Tensor& exponential_kernel_cuda(Tensor& self, double lambd,
                                std::optional<Generator> generator) {
    if (!(lambd > 0.0)) {
        TP_THROW(RuntimeError,
                 "exponential_ expects lambda > 0.0, but found lambda=", lambd);
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return exponential_kernel_cuda(t, lambd, std::move(generator));
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
            },
            std::move(generator));
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
            },
            std::move(generator));
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
                },
                std::move(generator));
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
                },
                std::move(generator));
        }
    } else {
        TP_THROW(NotImplementedError, "exponential_() only supports floating dtypes on CUDA for now");
    }
    return self;
}

// --- TransformationHelper.h): random_ / randint / geometric_ / -------------
// --- log_normal_ / cauchy_ / poisson / randperm ----------------------------

template <typename scalar_t>
void geometric_fill_float_cuda(scalar_t* data, int64_t n, float p,
                               std::optional<Generator> generator) {
    distribution_nullary_kernel<scalar_t, float4, 4>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
        [p] __device__ (float val) {
            return static_cast<scalar_t>(::ceilf(::logf(val) / ::log1pf(-p)));
        },
        std::move(generator));
}

void geometric_fill_double_cuda(double* data, int64_t n, double p,
                                std::optional<Generator> generator) {
    distribution_nullary_kernel<double, double2, 2>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
        [p] __device__ (double val) {
            return ::ceil(::log(val) / ::log1p(-p));
        },
        std::move(generator));
}

Tensor& geometric_kernel_cuda(Tensor& self, double p,
                              std::optional<Generator> generator) {
    TP_THROW_IF(!(p > 0.0 && p < 1.0), RuntimeError,
                "geometric_ expects p to be in (0, 1), but got p=", p);
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return geometric_kernel_cuda(t, p, std::move(generator));
        });
    }

    const int64_t n = self.numel();
    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        if constexpr (std::is_same_v<scalar_t, double>) {
            geometric_fill_double_cuda(self.data_ptr<double>(), n, p,
                                       std::move(generator));
        } else {
            geometric_fill_float_cuda<scalar_t>(
                self.data_ptr<scalar_t>(), n, static_cast<float>(p),
                std::move(generator));
        }
    });
    return self;
}

Tensor& log_normal_kernel_cuda(Tensor& self, double mean, double std,
                               std::optional<Generator> generator) {
    if (!(std > 0.0)) {
        TP_THROW(RuntimeError,
                 "log_normal_ expects std > 0.0, but found std=", std);
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return log_normal_kernel_cuda(t, mean, std, std::move(generator));
        });
    }
    int64_t n = self.numel();
    if (self.dtype() == DType::Float32) {
        float* data = self.data_ptr<float>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        // transformation::log_normal: exp(normal_draw).
        distribution_nullary_kernel<float, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) { return ::expf(mu + sigma * rand); },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal2_double(state); },
            [mean, std] __device__ (double rand) { return ::exp(mean + std * rand); },
            std::move(generator));
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<Half>(::expf(mu + sigma * rand));
            },
            std::move(generator));
    } else if (self.dtype() == DType::BFloat16) {
        BFloat16* data = self.data_ptr<BFloat16>();
        const float mu = static_cast<float>(mean);
        const float sigma = static_cast<float>(std);
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_normal4(state); },
            [mu, sigma] __device__ (float rand) {
                return static_cast<BFloat16>(::expf(mu + sigma * rand));
            },
            std::move(generator));
    } else {
        TP_THROW(NotImplementedError, "log_normal_() only supports floating dtypes on CUDA");
    }
    return self;
}

Tensor& cauchy_kernel_cuda(Tensor& self, double median, double sigma,
                           std::optional<Generator> generator) {
    if (!(sigma > 0.0)) {
        TP_THROW(RuntimeError,
                 "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
    }
    if (self.numel() == 0) return self;
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return cauchy_kernel_cuda(t, median, sigma, std::move(generator));
        });
    }
    int64_t n = self.numel();
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
                return med + sig * ::tanf(static_cast<float>(kPi) * (val - 0.5f));
            },
            std::move(generator));
    } else if (self.dtype() == DType::Float64) {
        double* data = self.data_ptr<double>();
        distribution_nullary_kernel<double, double2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform2_double(state); },
            [median, sigma] __device__ (double val) {
                return median + sigma * ::tan(kPi * (val - 0.5));
            },
            std::move(generator));
    } else if (self.dtype() == DType::Float16) {
        Half* data = self.data_ptr<Half>();
        const float med = static_cast<float>(median);
        const float sig = static_cast<float>(sigma);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        distribution_nullary_kernel<Half, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [med, sig] __device__ (float val) {
                val = val > 1.f - kEps ? 1.f - kEps : val;
                val = val < kEps ? kEps : val;
                return static_cast<Half>(med + sig * ::tanf(
                    3.14159265358979323846f * (val - 0.5f)));
            },
            std::move(generator));
    } else if (self.dtype() == DType::BFloat16) {
        BFloat16* data = self.data_ptr<BFloat16>();
        const float med = static_cast<float>(median);
        const float sig = static_cast<float>(sigma);
        constexpr float kEps = std::numeric_limits<float>::epsilon();
        distribution_nullary_kernel<BFloat16, float4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); },
            [med, sig] __device__ (float val) {
                val = val > 1.f - kEps ? 1.f - kEps : val;
                val = val < kEps ? kEps : val;
                return static_cast<BFloat16>(med + sig * ::tanf(
                    3.14159265358979323846f * (val - 0.5f)));
            },
            std::move(generator));
    } else {
        TP_THROW(NotImplementedError, "cauchy_() only supports floating dtypes on CUDA");
    }
    return self;
}

template <typename scalar_t>
void launch_random_range_cuda(scalar_t* data, int64_t n,
                              uint64_t range, int64_t base,
                              std::optional<Generator> generator = std::nullopt) {
    if (range >= (1ULL << 28)) {
        distribution_nullary_kernel<scalar_t, ulonglong2, 2>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
                ulonglong2 random;
                uint4 words = curand4(state);
                random.x = (static_cast<uint64_t>(words.x) << 32) | words.y;
                random.y = (static_cast<uint64_t>(words.z) << 32) | words.w;
                return random;
            },
            [range, base] __device__ (uint64_t value) {
                return static_cast<scalar_t>(static_cast<int64_t>(
                    (value % range) + static_cast<uint64_t>(base)));
            },
            std::move(generator));
    } else {
        distribution_nullary_kernel<scalar_t, uint4, 4>(
            data, n,
            [] __device__ (curandStatePhilox4_32_10_t* state) {
                return curand4(state);
            },
            [range, base] __device__ (unsigned int value) {
                return static_cast<scalar_t>(static_cast<int64_t>(
                    (static_cast<uint64_t>(value) % range) +
                    static_cast<uint64_t>(base)));
            },
            std::move(generator));
    }
}

// Raw 64-bit draws, for a range covering every 64-bit value.
template <typename scalar_t>
void launch_random_bits_cuda(scalar_t* data, int64_t n,
                             std::optional<Generator> generator) {
    distribution_nullary_kernel<scalar_t, ulonglong2, 2>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
            ulonglong2 random;
            uint4 words = curand4(state);
            random.x = (static_cast<uint64_t>(words.x) << 32) | words.y;
            random.y = (static_cast<uint64_t>(words.z) << 32) | words.w;
            return random;
        },
        [] __device__ (uint64_t value) {
            return static_cast<scalar_t>(static_cast<int64_t>(value));
        },
        std::move(generator));
}

// Every value in [base, top]; a range that wraps to zero covers all 64-bit
// values and is drawn as raw bits.
template <typename scalar_t>
void launch_random_upto_cuda(scalar_t* data, int64_t n, int64_t base,
                             uint64_t top, std::optional<Generator> generator) {
    const uint64_t range = top - static_cast<uint64_t>(base) + 1;
    if (range == 0) {
        launch_random_bits_cuda(data, n, std::move(generator));
    } else {
        launch_random_range_cuda(data, n, range, base, std::move(generator));
    }
}

static void randint_fill_dispatch(Tensor& t, int64_t low, int64_t high,
                                  std::optional<Generator> generator = std::nullopt) {
    int64_t n = t.numel();
    distribution::check_random_from_to_bounds(low, high, t.dtype());
    if (n == 0) return;
    const uint64_t range = static_cast<uint64_t>(high) -
        static_cast<uint64_t>(low);
    const int64_t base = low;
    distribution::dispatch_dtype(t.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        launch_random_range_cuda(t.data_ptr<scalar_t>(), n, range, base,
                                 std::move(generator));
    });
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

// `random_` without bounds: every value from zero up to the largest one the
// dtype holds exactly.
Tensor& random_kernel_cuda(Tensor& self, std::optional<Generator> generator) {
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return random_kernel_cuda(t, std::move(generator));
        });
    }
    const int64_t n = self.numel();
    if (n == 0) return self;
    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        launch_random_upto_cuda(self.data_ptr<scalar_t>(), n, 0,
                                distribution::unbounded_top<scalar_t>(),
                                std::move(generator));
    });
    return self;
}

// `random_.from`: [from, to), or from `from` up to the largest value the
// dtype holds exactly when no upper bound is given.
Tensor& random_from_kernel_cuda(Tensor& self, int64_t from,
                                std::optional<int64_t> to,
                                std::optional<Generator> generator) {
    TP_THROW_IF(to.has_value() && *to <= from, RuntimeError,
                "random_ expects 'from' to be less than 'to', but got from=",
                from, " >= to=", to.value_or(from));
    if (!self.is_contiguous()) {
        return fill_via_contiguous(self, [&](Tensor& t) {
            return random_from_kernel_cuda(t, from, to, std::move(generator));
        });
    }
    if (to.has_value()) {
        randint_fill_dispatch(self, from, *to, std::move(generator));
        return self;
    }
    const int64_t n = self.numel();
    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const uint64_t top = distribution::unbounded_top<scalar_t>();
        TP_THROW_IF(from >= 0 && static_cast<uint64_t>(from) > top, RuntimeError,
                    "random_ expects 'from' to be at most ", top, " for ",
                    self.dtype(), ", but got from=", from);
        if (n == 0) return;
        launch_random_upto_cuda(self.data_ptr<scalar_t>(), n, from, top,
                                std::move(generator));
    });
    return self;
}

Tensor& random_to_kernel_cuda(Tensor& self, int64_t to,
                              std::optional<Generator> generator) {
    return random_from_kernel_cuda(self, 0, to, std::move(generator));
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
    const Tensor valid_probabilities = Tensor::logical_and(
        probability.ge(Scalar(0)), probability.le(Scalar(1)));
    if (!valid_probabilities.all().item<bool>()) {
        TP_THROW(ValueError,
                 "bernoulli_ expects probability values in [0, 1]");
    }
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

void check_randperm_size(int64_t n, DType dtype) {
    if (n < 0) {
        TP_THROW(RuntimeError, "randperm(): n must be non-negative, got ", n);
    }
    uint64_t max_index;
    switch (dtype) {
        case DType::UInt8: max_index = std::numeric_limits<uint8_t>::max(); break;
        case DType::Int8: max_index = std::numeric_limits<int8_t>::max(); break;
        case DType::Int16: max_index = std::numeric_limits<int16_t>::max(); break;
        case DType::Int32: max_index = std::numeric_limits<int32_t>::max(); break;
        case DType::UInt16: max_index = std::numeric_limits<uint16_t>::max(); break;
        case DType::UInt32: max_index = std::numeric_limits<uint32_t>::max(); break;
        case DType::Float16: max_index = uint64_t{1} << 11; break;
        case DType::BFloat16: max_index = uint64_t{1} << 8; break;
        case DType::Float32: max_index = uint64_t{1} << 24; break;
        case DType::Float64: max_index = uint64_t{1} << 53; break;
        case DType::Bool: max_index = 1; break;
        case DType::Int64:
        case DType::UInt64:
            return;
        default:
            TP_THROW(NotImplementedError,
                     "randperm() does not support this output dtype");
    }
    if (static_cast<uint64_t>(n) > max_index + 1) {
        TP_THROW(RuntimeError,
                 "randperm(): n is too large for the requested output dtype");
    }
}

Tensor randperm_kernel_cuda(int64_t n, DType dtype, Device device) {
    check_randperm_size(n, dtype);
    Tensor idx(std::vector<int64_t>{n}, DType::Int64, device);
    if (n == 0) {
        if (dtype != DType::Int64) return Tensor::empty({n}, dtype, device);
        return idx;
    }
    Tensor keys = Tensor::empty({n}, DType::Int64, device);
    int64_t* data = keys.data_ptr<int64_t>();
    distribution_nullary_kernel<int64_t, ulonglong2, 2>(
        data, n,
        [] __device__ (curandStatePhilox4_32_10_t* state) {
            ulonglong2 random;
            uint4 words = curand4(state);
            random.x = (static_cast<uint64_t>(words.x) << 32) | words.y;
            random.y = (static_cast<uint64_t>(words.z) << 32) | words.w;
            return random;
        },
        [] __device__ (uint64_t value) {
            return static_cast<int64_t>(value);
        });
    extern Tensor argsort_cuda(const Tensor& self, int64_t dim, bool descending);
    idx = argsort_cuda(keys, 0, false);
    if (dtype == DType::Int64) return idx;
    Tensor out = Tensor::empty({n}, dtype, device);
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
    if (layout.has_value() && *layout != 5) {
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
    m.impl("random_.from", random_from_kernel_cuda);
    m.impl("random_.to", random_to_kernel_cuda);
    m.impl("bernoulli", bernoulli_kernel_cuda);
    m.impl("bernoulli.out", bernoulli_out_kernel_cuda);
    m.impl("bernoulli.p", bernoulli_p_kernel_cuda);
    m.impl("bernoulli_.Tensor", bernoulli_tensor_inplace_kernel_cuda);
    m.impl("bernoulli_.float", bernoulli_scalar_inplace_kernel_cuda);
    m.impl("randint", randint_stub_cuda);
    m.impl("randint_like", randint_like_kernel_cuda);
    m.impl("randperm", randperm_stub_cuda);
}

} // namespace cuda
} // namespace tensorplay
